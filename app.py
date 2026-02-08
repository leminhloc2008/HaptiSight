import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import gradio as gr
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
YOLO_DIR = ROOT / "REAL-TIME_Distance_Estimation_with_YOLOV7"

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
        candidates.extend([YOLO_DIR / "yolov7-tiny.pt", YOLO_DIR / "yolov7.pt"])

        for path in candidates:
            if path.is_file():
                return path

        # No local checkpoint: return a known filename so YOLOv7's attempt_download() can fetch it.
        if env_weight:
            return ROOT / Path(env_weight).name
        return ROOT / "yolov7-tiny.pt"

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
ENGINE_ERROR = None


def get_engine() -> RealtimeEngine:
    global ENGINE, ENGINE_ERROR
    if ENGINE is not None:
        return ENGINE
    if ENGINE_ERROR is not None:
        raise RuntimeError(ENGINE_ERROR)
    try:
        ENGINE = RealtimeEngine()
        return ENGINE
    except Exception as exc:
        ENGINE_ERROR = str(exc)
        raise


def process_frame(frame, conf_thres, iou_thres, img_size, max_det, smooth_factor):
    if frame is None:
        return None, "Dang cho frame webcam..."

    try:
        engine = get_engine()
    except Exception as exc:
        return frame, f"Loi khoi tao model: {exc}"

    return engine.infer(
        frame_rgb=frame,
        conf_thres=float(conf_thres),
        iou_thres=float(iou_thres),
        img_size=int(img_size),
        max_det=int(max_det),
        smooth_factor=float(smooth_factor),
    )


DESCRIPTION = (
    "YOLOER V2 tren Hugging Face Spaces (CPU realtime webcam). "
    "Mac dinh uu tien yolov7-tiny.pt de tang FPS; neu khong co se dung yolov7.pt."
)

demo = gr.Interface(
    fn=process_frame,
    inputs=[
        gr.Image(
            label="Webcam",
            type="numpy",
            sources=["webcam"],
            streaming=True,
        ),
        gr.Slider(0.10, 0.90, value=0.35, step=0.01, label="Confidence threshold"),
        gr.Slider(0.10, 0.90, value=0.45, step=0.01, label="IoU threshold"),
        gr.Slider(320, 640, value=384, step=32, label="Inference image size"),
        gr.Slider(1, 100, value=20, step=1, label="Max detections"),
        gr.Slider(0.0, 0.95, value=0.55, step=0.01, label="Distance smoothing"),
    ],
    outputs=[
        gr.Image(label="Result", type="numpy"),
        gr.Textbox(label="Runtime stats"),
    ],
    title="YOLOER V2 - Realtime Distance Estimation on CPU",
    description=DESCRIPTION,
    live=True,
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
