import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import gradio as gr
import h5py
import numpy as np
import requests
import torch

ROOT = Path(__file__).resolve().parent
YOLO_DIR = ROOT / "REAL-TIME_Distance_Estimation_with_YOLOV7"
MAX_FRAME_EDGE = int(os.getenv("MAX_FRAME_EDGE", "640"))

# PyTorch >= 2.6 defaults torch.load(..., weights_only=True), which breaks legacy YOLOv7 checkpoints.
_ORIG_TORCH_LOAD = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_compat

if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

from models.experimental import attempt_load  # noqa: E402
from utils.general import check_img_size, non_max_suppression, scale_coords  # noqa: E402


def letterbox(img, new_shape=640, color=(114, 114, 114), auto=False, scale_fill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


class DistanceMLP:
    def __init__(self, h5_path: Path):
        self.layers = []
        with h5py.File(h5_path, "r") as h5_file:
            for layer_name in ("dense_1", "dense_2", "dense_3", "dense_4"):
                group = h5_file[layer_name][layer_name]
                kernel = np.array(group["kernel:0"], dtype=np.float32)
                bias = np.array(group["bias:0"], dtype=np.float32)
                self.layers.append((kernel, bias))

    def predict(self, features: np.ndarray) -> np.ndarray:
        out = features.astype(np.float32, copy=False)
        for idx, (kernel, bias) in enumerate(self.layers):
            out = (out @ kernel) + bias
            if idx < len(self.layers) - 1:
                out = np.maximum(out, 0.0)
        return out


class RealtimeEngine:
    ORIG_HEIGHT = 375.0
    ORIG_WIDTH = 1242.0
    WEIGHTS_URLS = {
        "yolov7-tiny.pt": [
            "https://huggingface.co/akhaliq/yolov7/resolve/main/yolov7-tiny.pt",
            "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt",
        ],
        "yolov7.pt": [
            "https://huggingface.co/akhaliq/yolov7/resolve/main/yolov7.pt",
            "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
        ],
    }

    def __init__(self):
        requested_threads = int(os.getenv("CPU_THREADS", "2"))
        requested_threads = max(1, requested_threads)
        try:
            torch.set_num_threads(requested_threads)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        self.device = torch.device("cpu")
        self.weights_path = self._resolve_weights_path()
        self.distance_model = DistanceMLP(YOLO_DIR / "model@1535470106.h5")
        self.model = attempt_load(str(self.weights_path), map_location=self.device)
        self.model.eval()
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, "module") else self.model.names
        rng = np.random.default_rng(7)
        self.colors = rng.integers(0, 255, size=(len(self.names), 3), dtype=np.uint8)
        self.distance_cache: Dict[str, float] = {}

    def _resolve_weights_path(self) -> Path:
        env_weight = os.getenv("YOLO_WEIGHTS", "").strip()
        candidates = []
        if env_weight:
            env_path = Path(env_weight)
            candidates.extend([env_path, ROOT / env_path, YOLO_DIR / env_path])
        candidates.extend([ROOT / "yolov7-tiny.pt", ROOT / "yolov7.pt", YOLO_DIR / "yolov7-tiny.pt", YOLO_DIR / "yolov7.pt"])

        for path in candidates:
            if path.is_file():
                return path

        if env_weight.lower().startswith(("http://", "https://")):
            target = ROOT / Path(env_weight).name
            self._download_file(env_weight, target)
            return target

        weight_name = Path(env_weight).name if env_weight else "yolov7-tiny.pt"
        if weight_name not in self.WEIGHTS_URLS:
            raise FileNotFoundError(
                f"Khong tim thay weights local cho '{weight_name}'. "
                "Dat YOLO_WEIGHTS la duong dan ton tai hoac URL hop le."
            )

        target = ROOT / weight_name
        self._download_from_mirrors(weight_name, target)
        return target

    def _download_file(self, url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(tmp, "wb") as file_obj:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_obj.write(chunk)
        if tmp.stat().st_size < 1_000_000:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded file qua nho tu {url}")
        tmp.replace(target)

    def _download_from_mirrors(self, weight_name: str, target: Path) -> None:
        errors = []
        for url in self.WEIGHTS_URLS.get(weight_name, []):
            try:
                self._download_file(url, target)
                return
            except Exception as exc:
                errors.append(f"{url} -> {exc}")
        joined = "\n".join(errors) if errors else "Khong co mirror URL"
        raise RuntimeError(f"Khong the tai weights {weight_name}. Chi tiet:\n{joined}")

    def _predict_distances(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.empty((0,), dtype=np.float32)

        x_mean = features.mean(axis=0, keepdims=True)
        x_std = features.std(axis=0, keepdims=True)
        x_std[x_std < 1e-6] = 1.0
        x_norm = (features - x_mean) / x_std

        y_proxy = ((features[:, 3] - features[:, 1]) / 3.0).reshape(-1, 1)
        y_mean = y_proxy.mean(axis=0, keepdims=True)
        y_std = y_proxy.std(axis=0, keepdims=True)
        y_std[y_std < 1e-6] = 1.0

        pred_norm = self.distance_model.predict(x_norm)
        pred = (pred_norm * y_std) + y_mean
        return np.maximum(pred.reshape(-1), 0.0)

    def infer(
        self,
        frame_rgb: np.ndarray,
        conf_thres: float,
        iou_thres: float,
        img_size: int,
        max_det: int,
        smooth_factor: float,
    ) -> Tuple[np.ndarray, str]:
        start = time.perf_counter()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        img_h, img_w = frame_bgr.shape[:2]
        img_size = check_img_size(int(img_size), s=self.stride)

        padded = letterbox(frame_bgr, new_shape=img_size, stride=self.stride, auto=False)[0]
        padded = padded[:, :, ::-1].transpose(2, 0, 1)
        padded = np.ascontiguousarray(padded)

        tensor = torch.from_numpy(padded).to(self.device).float() / 255.0
        tensor = tensor.unsqueeze(0)

        with torch.inference_mode():
            pred = self.model(tensor, augment=False)[0]
            det = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)[0]

        if len(det):
            det = det.clone()
            det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], frame_bgr.shape).round()
            det = det[: int(max_det)]

            features = []
            parsed = []
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
                x1 = int(np.clip(x1, 0, img_w - 1))
                y1 = int(np.clip(y1, 0, img_h - 1))
                x2 = int(np.clip(x2, 0, img_w - 1))
                y2 = int(np.clip(y2, 0, img_h - 1))

                scaled_x1 = (x1 / img_w) * self.ORIG_WIDTH
                scaled_x2 = (x2 / img_w) * self.ORIG_WIDTH
                scaled_y1 = (y1 / img_h) * self.ORIG_HEIGHT
                scaled_y2 = (y2 / img_h) * self.ORIG_HEIGHT

                features.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])
                parsed.append((x1, y1, x2, y2, float(conf.item()), int(cls.item())))

            distance_values = self._predict_distances(np.asarray(features, dtype=np.float32))
            new_cache: Dict[str, float] = {}

            for idx, (x1, y1, x2, y2, conf, cls_id) in enumerate(parsed):
                key = f"{cls_id}:{(x1 + x2) // 80}:{(y1 + y2) // 80}"
                raw_dist = float(distance_values[idx])
                prev_dist = self.distance_cache.get(key, raw_dist)
                smoothed = smooth_factor * prev_dist + (1.0 - smooth_factor) * raw_dist
                new_cache[key] = smoothed

                label = f"{self.names[cls_id]} {smoothed:.1f}m ({conf:.2f})"
                color = tuple(int(c) for c in self.colors[cls_id])
                self._draw_box(frame_bgr, x1, y1, x2, y2, label, color)

            self.distance_cache = new_cache
            object_count = len(parsed)
        else:
            self.distance_cache = {}
            object_count = 0

        latency_ms = (time.perf_counter() - start) * 1000.0
        fps = 1000.0 / max(latency_ms, 1e-6)
        stats = (
            f"weights={self.weights_path.name} | objects={object_count} | "
            f"latency={latency_ms:.1f}ms | fps~{fps:.1f}"
        )
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), stats

    @staticmethod
    def _draw_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str, color: Tuple[int, int, int]) -> None:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        text_bottom = max(y1, th + 6)
        cv2.rectangle(img, (x1, text_bottom - th - 6), (x1 + tw + 6, text_bottom), color, -1)
        cv2.putText(img, label, (x1 + 3, text_bottom - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


ENGINE = None
ASYNC_WORKER = None


def get_engine() -> RealtimeEngine:
    global ENGINE
    if ENGINE is not None:
        return ENGINE
    ENGINE = RealtimeEngine()
    return ENGINE


class AsyncInferenceWorker:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending_frame = None
        self._pending_params = None
        self._pending_seq = 0
        self._done_seq = 0
        self._latest_frame = None
        self._latest_stats = "Dang tai model..."
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame, conf_thres, iou_thres, img_size, max_det, smooth_factor):
        frame = downscale_frame(frame, MAX_FRAME_EDGE)
        params = (
            float(conf_thres),
            float(iou_thres),
            int(img_size),
            int(max_det),
            float(smooth_factor),
        )
        with self._lock:
            self._pending_frame = frame
            self._pending_params = params
            self._pending_seq += 1
            latest_frame = self._latest_frame
            latest_stats = self._latest_stats

        # Return immediately: no blocking on model inference in request path.
        if latest_frame is None:
            return frame, latest_stats
        return latest_frame, latest_stats

    def _run(self):
        engine = None
        while True:
            with self._lock:
                has_new = self._pending_frame is not None and self._pending_seq > self._done_seq
                if has_new:
                    seq = self._pending_seq
                    frame = self._pending_frame.copy()
                    params = self._pending_params
                else:
                    seq = 0
                    frame = None
                    params = None

            if frame is None:
                time.sleep(0.005)
                continue

            if engine is None:
                try:
                    engine = get_engine()
                except Exception as exc:
                    with self._lock:
                        self._latest_stats = f"Loi khoi tao model: {exc}"
                    time.sleep(0.2)
                    continue

            try:
                out_frame, stats = engine.infer(frame, *params)
            except Exception as exc:
                out_frame = frame
                stats = f"Loi infer: {exc}"

            with self._lock:
                if seq >= self._done_seq:
                    self._done_seq = seq
                    self._latest_frame = out_frame
                    self._latest_stats = stats


def get_async_worker() -> AsyncInferenceWorker:
    global ASYNC_WORKER
    if ASYNC_WORKER is None:
        ASYNC_WORKER = AsyncInferenceWorker()
    return ASYNC_WORKER


def process_frame(frame, conf_thres, iou_thres, img_size, max_det, smooth_factor):
    if frame is None:
        return None, "Dang cho frame webcam..."

    worker = get_async_worker()
    return worker.submit(frame, conf_thres, iou_thres, img_size, max_det, smooth_factor)


DESCRIPTION = (
    "YOLOER V2 tren Hugging Face Spaces (CPU realtime webcam). "
    "Mac dinh uu tien yolov7-tiny.pt de tang FPS; neu khong co se dung yolov7.pt."
)

def downscale_frame(frame: np.ndarray, max_edge: int) -> np.ndarray:
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return frame
    scale = max_edge / float(longest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


with gr.Blocks(title="YOLOER V2 - Realtime Distance Estimation on CPU") as demo:
    gr.Markdown("## YOLOER V2 - Realtime Distance Estimation on CPU")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            webcam = gr.Image(
                label="Webcam",
                type="numpy",
                format="jpeg",
                sources=["webcam"],
                streaming=True,
                height=320,
                webcam_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 360},
                        "frameRate": {"ideal": 24, "max": 30},
                    }
                },
            )
            conf_slider = gr.Slider(0.10, 0.90, value=0.45, step=0.01, label="Confidence threshold")
            iou_slider = gr.Slider(0.10, 0.90, value=0.45, step=0.01, label="IoU threshold")
            size_slider = gr.Slider(192, 512, value=256, step=32, label="Inference image size")
            max_det_slider = gr.Slider(1, 60, value=8, step=1, label="Max detections")
            smooth_slider = gr.Slider(0.0, 0.95, value=0.55, step=0.01, label="Distance smoothing")
        with gr.Column(scale=1):
            result = gr.Image(label="Result", type="numpy", format="jpeg", height=320)
            stats = gr.Textbox(label="Runtime stats")

    webcam.stream(
        process_frame,
        inputs=[webcam, conf_slider, iou_slider, size_slider, max_det_slider, smooth_slider],
        outputs=[result, stats],
        show_progress="hidden",
        queue=False,
        trigger_mode="always_last",
        concurrency_limit=1,
        stream_every=0.033,
    )


if __name__ == "__main__":
    demo.launch()
