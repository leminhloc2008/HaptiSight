import os
import sys
import inspect
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
YOLO_DIR = ROOT / "REAL-TIME_Distance_Estimation_with_YOLOV7"
MAX_FRAME_EDGE = int(os.getenv("MAX_FRAME_EDGE", "512"))
MAX_OUTPUT_EDGE = int(os.getenv("MAX_OUTPUT_EDGE", "416"))
MAX_DEPTH_EDGE = int(os.getenv("MAX_DEPTH_EDGE", "288"))
CAM_FOV_DEG = float(os.getenv("CAM_FOV_DEG", "70.0"))

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


class MidasDepthEstimator:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.transform = None
        self.error_message = None

    @staticmethod
    def _hub_load(repo: str, model_name: str):
        try:
            return torch.hub.load(repo, model_name, trust_repo=True)
        except TypeError:
            return torch.hub.load(repo, model_name)

    def ensure_loaded(self):
        if self.model is not None and self.transform is not None:
            return
        if self.error_message is not None:
            raise RuntimeError(self.error_message)

        try:
            torch.hub.set_dir(str(ROOT / ".torch_hub_cache"))
            self.model = self._hub_load("intel-isl/MiDaS", "MiDaS_small")
            transforms = self._hub_load("intel-isl/MiDaS", "transforms")
            self.transform = transforms.small_transform
            self.model.to(self.device).eval()
        except Exception as exc:
            self.error_message = f"Khong tai duoc MiDaS_small: {exc}"
            raise RuntimeError(self.error_message)

    def predict(self, frame_rgb: np.ndarray, max_edge: int) -> np.ndarray:
        self.ensure_loaded()
        frame_input = downscale_frame(frame_rgb, max_edge)

        with torch.inference_mode():
            inp = self.transform(frame_input).to(self.device)
            pred = self.model(inp)
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=frame_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze(1).squeeze(0)

        depth = pred.detach().cpu().numpy().astype(np.float32)
        lo, hi = np.percentile(depth, [2, 98])
        if hi - lo < 1e-6:
            return np.zeros_like(depth, dtype=np.float32)
        depth = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        return depth


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
        self.xyz_cache: Dict[str, Tuple[float, float, float]] = {}
        self.xyz_filter_state: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        self.depth_estimator = None
        self.depth_last_map = None
        self.depth_last_latency_ms = 0.0
        self.depth_frame_counter = 0

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

    @staticmethod
    def _sample_box_depth(depth_map: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        if depth_map is None:
            return -1.0
        h, w = depth_map.shape[:2]
        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            return float(depth_map[y1, x1])
        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return float(depth_map[y1, x1])
        return float(np.median(roi))

    def _get_depth_map(self, frame_rgb: np.ndarray, depth_enabled: bool, depth_interval: int):
        if not depth_enabled:
            self.depth_last_map = None
            return None, None

        self.depth_frame_counter += 1
        depth_interval = max(1, int(depth_interval))
        refresh = self.depth_last_map is None or (self.depth_frame_counter % depth_interval == 0)
        if not refresh:
            return self.depth_last_map, None

        if self.depth_estimator is None:
            self.depth_estimator = MidasDepthEstimator(self.device)

        t0 = time.perf_counter()
        depth_map = self.depth_estimator.predict(frame_rgb, MAX_DEPTH_EDGE)
        self.depth_last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.depth_last_map = depth_map
        return depth_map, None

    @staticmethod
    def _fuse_distance_with_depth(
        distance_m: np.ndarray,
        depth_vals: np.ndarray,
        conf_vals: np.ndarray,
        area_vals: np.ndarray,
        frame_area: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if distance_m.size == 0 or depth_vals.size == 0:
            return distance_m, distance_m, np.zeros_like(distance_m, dtype=np.float32)

        inv_depth = np.clip(depth_vals.astype(np.float32), 1e-3, None)
        distance_m = np.clip(distance_m.astype(np.float32), 0.05, None)
        conf_vals = np.clip(conf_vals.astype(np.float32), 0.0, 1.0)
        area_ratio = np.clip(area_vals.astype(np.float32) / max(frame_area, 1.0), 1e-4, 1.0)
        area_term = np.sqrt(area_ratio)

        # MiDaS is relative inverse depth. Calibrate per-frame scale using YOLOER metric distance.
        scale = np.median(distance_m * inv_depth)
        if not np.isfinite(scale) or scale <= 1e-6:
            return distance_m, distance_m, np.zeros_like(distance_m, dtype=np.float32)

        depth_metric = scale / inv_depth
        rel_err = np.abs(depth_metric - distance_m) / np.maximum(distance_m, 0.25)
        consistency = np.exp(-1.8 * rel_err).astype(np.float32)

        # Adaptive fusion: trust MiDaS more when detector confidence is lower or object is small/far.
        w_midas = 0.18 + 0.30 * (1.0 - conf_vals) + 0.14 * (1.0 - area_term)
        w_midas = np.clip(w_midas * consistency, 0.05, 0.58)

        fused = (1.0 - w_midas) * distance_m + w_midas * depth_metric
        return np.clip(fused, 0.05, None), depth_metric, w_midas

    def _update_xyz_filter(self, key: str, measurement_xyz: Tuple[float, float, float], conf: float, now_ts: float):
        meas = np.asarray(measurement_xyz, dtype=np.float32)
        state = self.xyz_filter_state.get(key)
        if state is None:
            self.xyz_filter_state[key] = (meas, np.zeros(3, dtype=np.float32), now_ts)
            return float(meas[0]), float(meas[1]), float(meas[2])

        pos, vel, last_ts = state
        dt = float(np.clip(now_ts - last_ts, 1e-3, 0.2))
        pred = pos + vel * dt
        residual = meas - pred
        conf = float(np.clip(conf, 0.0, 1.0))
        alpha = 0.30 + 0.45 * conf
        beta = 0.03 + 0.08 * conf
        pos_new = pred + alpha * residual
        vel_new = vel + (beta / dt) * residual
        self.xyz_filter_state[key] = (pos_new, vel_new, now_ts)
        return float(pos_new[0]), float(pos_new[1]), float(pos_new[2])

    def infer(
        self,
        frame_rgb: np.ndarray,
        conf_thres: float,
        iou_thres: float,
        img_size: int,
        max_det: int,
        smooth_factor: float,
        depth_enabled: bool,
        depth_alpha: float,
        depth_interval: int,
    ) -> Tuple[np.ndarray, str]:
        start = time.perf_counter()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        infer_frame_bgr = frame_bgr.copy()
        img_h, img_w = frame_bgr.shape[:2]
        img_size = check_img_size(int(img_size), s=self.stride)

        depth_map = None
        depth_error = None
        if depth_enabled:
            try:
                depth_map, _ = self._get_depth_map(frame_rgb, depth_enabled, depth_interval)
            except Exception as exc:
                depth_error = str(exc)
                depth_map = None

        if depth_map is not None and depth_alpha > 0:
            depth_uint8 = (np.clip(depth_map, 0.0, 1.0) * 255.0).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
            alpha = float(np.clip(depth_alpha, 0.0, 0.8))
            frame_bgr = cv2.addWeighted(frame_bgr, 1.0 - alpha, depth_color, alpha, 0.0)

        padded = letterbox(infer_frame_bgr, new_shape=img_size, stride=self.stride, auto=False)[0]
        padded = padded[:, :, ::-1].transpose(2, 0, 1)
        padded = np.ascontiguousarray(padded)

        tensor = torch.from_numpy(padded).to(self.device).float() / 255.0
        tensor = tensor.unsqueeze(0)

        with torch.inference_mode():
            pred = self.model(tensor, augment=False)[0]
            det = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)[0]

        nearest_info = None
        fusion_w_mean = None
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
            conf_vals_np = np.asarray([p[4] for p in parsed], dtype=np.float32)
            area_vals_np = np.asarray([max(1.0, float((p[2] - p[0]) * (p[3] - p[1]))) for p in parsed], dtype=np.float32)
            depth_vals = None
            depth_metric_vals = None
            fusion_w_vals = None
            if depth_map is not None:
                depth_vals = np.asarray(
                    [self._sample_box_depth(depth_map, p[0], p[1], p[2], p[3]) for p in parsed],
                    dtype=np.float32,
                )
                distance_values, depth_metric_vals, fusion_w_vals = self._fuse_distance_with_depth(
                    distance_values,
                    depth_vals,
                    conf_vals_np,
                    area_vals_np,
                    float(img_w * img_h),
                )
                if fusion_w_vals is not None and fusion_w_vals.size:
                    fusion_w_mean = float(np.mean(fusion_w_vals))

            fx = (img_w * 0.5) / max(np.tan(np.deg2rad(CAM_FOV_DEG * 0.5)), 1e-3)
            fy = fx
            cx = img_w * 0.5
            cy = img_h * 0.5

            new_cache: Dict[str, float] = {}
            new_xyz_cache: Dict[str, Tuple[float, float, float]] = {}
            now_ts = time.perf_counter()

            for idx, (x1, y1, x2, y2, conf, cls_id) in enumerate(parsed):
                key = f"{cls_id}:{(x1 + x2) // 80}:{(y1 + y2) // 80}"
                raw_dist = float(distance_values[idx])
                prev_dist = self.distance_cache.get(key, raw_dist)
                smoothed = smooth_factor * prev_dist + (1.0 - smooth_factor) * raw_dist
                new_cache[key] = smoothed

                u = 0.5 * (x1 + x2)
                v = 0.5 * (y1 + y2)
                z = max(smoothed, 0.05)
                x3 = ((u - cx) / fx) * z
                y3 = ((v - cy) / fy) * z
                x3, y3, z = self._update_xyz_filter(key, (x3, y3, z), conf, now_ts)
                new_xyz_cache[key] = (x3, y3, z)
                d3 = float(np.sqrt(x3 * x3 + y3 * y3 + z * z))

                obj = {
                    "name": self.names[cls_id],
                    "conf": conf,
                    "x": x3,
                    "y": y3,
                    "z": z,
                    "d": d3,
                }
                if nearest_info is None or obj["d"] < nearest_info["d"]:
                    nearest_info = obj

                if depth_map is not None:
                    depth_val = float(depth_vals[idx]) if depth_vals is not None else -1.0
                    fusion_w = float(fusion_w_vals[idx]) if fusion_w_vals is not None else 0.0
                    label = f"{self.names[cls_id]} {smoothed:.1f}m d{d3:.1f} z{depth_val:.2f} w{fusion_w:.2f}"
                else:
                    label = f"{self.names[cls_id]} {smoothed:.1f}m d{d3:.1f}"
                color = tuple(int(c) for c in self.colors[cls_id])
                self._draw_box(frame_bgr, x1, y1, x2, y2, label, color)

            self.distance_cache = new_cache
            self.xyz_cache = new_xyz_cache
            self.xyz_filter_state = {k: self.xyz_filter_state[k] for k in new_xyz_cache.keys() if k in self.xyz_filter_state}
            object_count = len(parsed)
        else:
            self.distance_cache = {}
            self.xyz_cache = {}
            self.xyz_filter_state = {}
            object_count = 0

        latency_ms = (time.perf_counter() - start) * 1000.0
        fps = 1000.0 / max(latency_ms, 1e-6)
        stats = (
            f"weights={self.weights_path.name} | objects={object_count} | "
            f"latency={latency_ms:.1f}ms | fps~{fps:.1f}"
        )
        if nearest_info is not None:
            direction_h = "center"
            if nearest_info["x"] < -0.25:
                direction_h = "left"
            elif nearest_info["x"] > 0.25:
                direction_h = "right"
            stats += (
                f" | nearest={nearest_info['name']} D={nearest_info['d']:.2f}m"
                f" XYZ=({nearest_info['x']:+.2f},{nearest_info['y']:+.2f},{nearest_info['z']:.2f})"
                f" dir={direction_h}"
            )
        if depth_enabled:
            stats += f" | depth_every={max(1, int(depth_interval))}"
            if self.depth_last_latency_ms > 0:
                stats += f" | depth={self.depth_last_latency_ms:.1f}ms"
            if fusion_w_mean is not None:
                stats += f" | w_depth={fusion_w_mean:.2f}"
            if depth_error:
                stats += f" | depth_err={depth_error}"
        result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result_rgb = downscale_frame(result_rgb, MAX_OUTPUT_EDGE)
        return result_rgb, stats

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
ENGINE_LOCK = threading.Lock()


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

    def submit(self, frame, conf_thres, iou_thres, img_size, max_det, smooth_factor, depth_enabled, depth_alpha, depth_interval):
        frame = downscale_frame(frame, MAX_FRAME_EDGE)
        params = (
            float(conf_thres),
            float(iou_thres),
            int(img_size),
            int(max_det),
            float(smooth_factor),
            bool(depth_enabled),
            float(depth_alpha),
            int(depth_interval),
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
                with ENGINE_LOCK:
                    out_frame, stats = engine.infer(frame, *params)
            except Exception as exc:
                out_frame = frame
                stats = f"Loi infer: {exc}"

            with self._lock:
                if seq >= self._done_seq:
                    self._done_seq = seq
                    self._latest_frame = out_frame
                    self._latest_stats = stats


def process_frame(frame, conf_thres, iou_thres, img_size, max_det, smooth_factor, depth_enabled, depth_alpha, depth_interval, session_worker):
    if frame is None:
        return None, "Dang cho frame webcam...", session_worker

    if session_worker is None:
        session_worker = AsyncInferenceWorker()

    out_frame, out_stats = session_worker.submit(
        frame,
        conf_thres,
        iou_thres,
        img_size,
        max_det,
        smooth_factor,
        depth_enabled,
        depth_alpha,
        depth_interval,
    )
    return out_frame, out_stats, session_worker


PROFILE_PRESETS = {
    "Realtime": {
        "conf": 0.50,
        "iou": 0.45,
        "img_size": 192,
        "max_det": 6,
        "smooth": 0.45,
        "depth_enabled": False,
        "depth_alpha": 0.12,
        "depth_interval": 7,
    },
    "Balanced": {
        "conf": 0.45,
        "iou": 0.45,
        "img_size": 224,
        "max_det": 8,
        "smooth": 0.55,
        "depth_enabled": True,
        "depth_alpha": 0.18,
        "depth_interval": 5,
    },
    "Precision": {
        "conf": 0.35,
        "iou": 0.50,
        "img_size": 288,
        "max_det": 14,
        "smooth": 0.65,
        "depth_enabled": True,
        "depth_alpha": 0.22,
        "depth_interval": 3,
    },
}


def apply_profile(profile_name: str):
    p = PROFILE_PRESETS.get(profile_name, PROFILE_PRESETS["Balanced"])
    return (
        p["conf"],
        p["iou"],
        p["img_size"],
        p["max_det"],
        p["smooth"],
        p["depth_enabled"],
        p["depth_alpha"],
        p["depth_interval"],
    )


DESCRIPTION = (
    "YOLOER V2 tren Hugging Face Spaces (CPU realtime webcam). "
    "Mac dinh uu tien yolov7-tiny.pt de tang FPS; neu khong co se dung yolov7.pt. "
    "Co the bat MiDaS v2.1 Small de phan tich depth va toa do 3D theo thoi gian thuc."
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
            profile = gr.Dropdown(
                choices=["Realtime", "Balanced", "Precision"],
                value="Balanced",
                label="Performance profile",
            )
            webcam = gr.Image(
                label="Webcam",
                type="numpy",
                format="jpeg",
                sources=["webcam"],
                streaming=True,
                height=300,
            )
            conf_slider = gr.Slider(0.10, 0.90, value=0.45, step=0.01, label="Confidence threshold")
            iou_slider = gr.Slider(0.10, 0.90, value=0.45, step=0.01, label="IoU threshold")
            size_slider = gr.Slider(192, 512, value=224, step=32, label="Inference image size")
            max_det_slider = gr.Slider(1, 60, value=8, step=1, label="Max detections")
            smooth_slider = gr.Slider(0.0, 0.95, value=0.55, step=0.01, label="Distance smoothing")
            depth_enabled = gr.Checkbox(value=True, label="Enable MiDaS v2.1 Small depth")
            depth_alpha = gr.Slider(0.0, 0.7, value=0.18, step=0.01, label="Depth overlay alpha")
            depth_interval = gr.Slider(1, 8, value=5, step=1, label="Depth update every N frames")
        with gr.Column(scale=1):
            result = gr.Image(label="Result", type="numpy", format="jpeg", height=300)
            stats = gr.Textbox(label="Runtime stats")
    session_worker = gr.State(value=None)

    profile.change(
        apply_profile,
        inputs=[profile],
        outputs=[
            conf_slider,
            iou_slider,
            size_slider,
            max_det_slider,
            smooth_slider,
            depth_enabled,
            depth_alpha,
            depth_interval,
        ],
    )

    stream_kwargs = {
        "show_progress": "hidden",
        "queue": False,
        "trigger_mode": "always_last",
        "concurrency_limit": 1,
        "stream_every": 0.04,
        "show_api": False,
    }
    supported_stream_args = set(inspect.signature(webcam.stream).parameters.keys())
    stream_kwargs = {k: v for k, v in stream_kwargs.items() if k in supported_stream_args}
    webcam.stream(
        process_frame,
        inputs=[
            webcam,
            conf_slider,
            iou_slider,
            size_slider,
            max_det_slider,
            smooth_slider,
            depth_enabled,
            depth_alpha,
            depth_interval,
            session_worker,
        ],
        outputs=[result, stats, session_worker],
        **stream_kwargs,
    )


if __name__ == "__main__":
    demo.launch()
