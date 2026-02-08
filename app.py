import os
import inspect
import threading
import time
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import cv2
import gradio as gr
import h5py
import numpy as np
import requests
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parent
YOLO_DIR = ROOT / "REAL-TIME_Distance_Estimation_with_YOLOV7"
MAX_FRAME_EDGE = int(os.getenv("MAX_FRAME_EDGE", "512"))
MAX_OUTPUT_EDGE = int(os.getenv("MAX_OUTPUT_EDGE", "416"))
MAX_DEPTH_EDGE = int(os.getenv("MAX_DEPTH_EDGE", "288"))
CAM_FOV_DEG = float(os.getenv("CAM_FOV_DEG", "70.0"))

from smart_agent import GeminiMultiAgentPlanner  # noqa: E402


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
    YOLOE_REPO_ID = "jameslahm/yoloe"
    DEFAULT_MODEL_ID = "yoloe-11s"
    FALLBACK_MODEL_ID = "yoloe-v8s"

    def __init__(self):
        requested_threads = int(os.getenv("CPU_THREADS", "2"))
        requested_threads = max(1, requested_threads)
        try:
            torch.set_num_threads(requested_threads)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        self.torch_device = torch.device("cpu")
        self.device = "cpu"
        self.weights_path = self._resolve_weights_path()
        self.detector_label = Path(self.weights_path).name
        self.distance_model = DistanceMLP(YOLO_DIR / "model@1535470106.h5")
        self.model: Optional[YOLOE] = None
        self.base_names: Dict[int, str] = {}
        self.names: Dict[int, str] = {}
        self.active_prompt_classes: Optional[Tuple[str, ...]] = None
        self.active_prompt_error: Optional[str] = None
        self.color_cache: Dict[str, Tuple[int, int, int]] = {}
        self._load_detector()
        self.distance_cache: Dict[str, float] = {}
        self.xyz_cache: Dict[str, Tuple[float, float, float]] = {}
        self.xyz_filter_state: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        self.depth_estimator = None
        self.depth_last_map = None
        self.depth_last_latency_ms = 0.0
        self.depth_frame_counter = 0

    @staticmethod
    def _model_filename(model_id: str) -> str:
        name = (model_id or "").strip()
        if not name:
            name = RealtimeEngine.DEFAULT_MODEL_ID
        if name.endswith(".pt"):
            return Path(name).name
        if name.endswith("-seg"):
            return f"{name}.pt"
        return f"{name}-seg.pt"

    def _resolve_weights_path(self) -> str:
        env_weight = os.getenv("YOLOE_WEIGHTS", "").strip()
        candidates = []
        if env_weight:
            if env_weight.lower().startswith(("http://", "https://")):
                target = ROOT / Path(env_weight).name
                self._download_file(env_weight, target)
                return str(target)
            env_path = Path(env_weight)
            candidates.extend([env_path, ROOT / env_path, YOLO_DIR / env_path])
        for path in candidates:
            if path.is_file():
                return str(path)

        if env_weight:
            raise FileNotFoundError(
                f"Khong tim thay weights local cho '{env_weight}'. "
                "Dat YOLOE_WEIGHTS la duong dan ton tai hoac URL hop le."
            )

        requested_model = os.getenv("YOLOE_MODEL_ID", self.DEFAULT_MODEL_ID).strip() or self.DEFAULT_MODEL_ID
        requested_file = self._model_filename(requested_model)
        try:
            return hf_hub_download(repo_id=self.YOLOE_REPO_ID, filename=requested_file)
        except Exception as exc:
            fallback_file = self._model_filename(self.FALLBACK_MODEL_ID)
            if requested_file == fallback_file:
                raise RuntimeError(f"Khong tai duoc YOLOE checkpoint '{requested_file}': {exc}") from exc
            try:
                return hf_hub_download(repo_id=self.YOLOE_REPO_ID, filename=fallback_file)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Khong tai duoc YOLOE checkpoint '{requested_file}' va fallback '{fallback_file}'. "
                    f"Chi tiet: {exc} | {fallback_exc}"
                ) from fallback_exc

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

    @staticmethod
    def _names_to_dict(names_obj) -> Dict[int, str]:
        if isinstance(names_obj, dict):
            return {int(k): str(v) for k, v in names_obj.items()}
        if isinstance(names_obj, (list, tuple)):
            return {i: str(v) for i, v in enumerate(names_obj)}
        return {}

    def _load_detector(self) -> None:
        self.model = YOLOE(self.weights_path)
        self.model.to(self.device)
        self.model.eval()
        self.base_names = self._names_to_dict(getattr(self.model, "names", None))
        self.names = dict(self.base_names)
        self.active_prompt_classes = None
        self.active_prompt_error = None

    @staticmethod
    def _parse_prompt_classes(class_prompt: str) -> List[str]:
        raw = (class_prompt or "").replace("\n", ",").replace(";", ",")
        items = [x.strip() for x in raw.split(",") if x and x.strip()]
        unique = []
        seen = set()
        for item in items:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def _apply_prompt_classes_if_needed(self, class_prompt: str) -> Optional[str]:
        classes = self._parse_prompt_classes(class_prompt)
        if not classes:
            if self.active_prompt_classes is not None:
                self._load_detector()
            return None

        cls_tuple = tuple(classes)
        if self.active_prompt_classes == cls_tuple:
            return self.active_prompt_error

        assert self.model is not None
        try:
            embeddings = self.model.get_text_pe(classes)
            self.model.set_classes(classes, embeddings)
            self.names = self._names_to_dict(getattr(self.model, "names", classes))
            self.active_prompt_classes = cls_tuple
            self.active_prompt_error = None
        except Exception as exc:
            self.active_prompt_classes = cls_tuple
            self.active_prompt_error = f"Khong set duoc YOLOE prompt classes: {exc}"
        return self.active_prompt_error

    def _class_name(self, cls_id: int) -> str:
        return self.names.get(int(cls_id), str(int(cls_id)))

    def _color_for_name(self, name: str) -> Tuple[int, int, int]:
        color = self.color_cache.get(name)
        if color is not None:
            return color
        seed = sum(ord(c) for c in name) % (2**32)
        rng = np.random.default_rng(seed)
        color = tuple(int(v) for v in rng.integers(40, 255, size=3))
        self.color_cache[name] = color
        return color

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
            self.depth_estimator = MidasDepthEstimator(self.torch_device)

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
        class_prompt: str,
    ) -> Tuple[np.ndarray, str, Dict[str, object]]:
        start = time.perf_counter()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        infer_frame_bgr = frame_bgr.copy()
        img_h, img_w = frame_bgr.shape[:2]
        img_size = int(np.clip(int(img_size), 128, 640))
        img_size = max(128, (img_size // 32) * 32)

        prompt_error = self._apply_prompt_classes_if_needed(class_prompt)

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

        with torch.inference_mode():
            assert self.model is not None
            results = self.model.predict(
                source=infer_frame_bgr,
                imgsz=img_size,
                conf=float(conf_thres),
                iou=float(iou_thres),
                max_det=int(max_det),
                device=self.device,
                verbose=False,
            )
        pred = results[0] if results else None
        boxes = pred.boxes if pred is not None else None

        nearest_info = None
        fusion_w_mean = None
        scene_objects = []
        if boxes is not None and len(boxes) > 0:
            xyxy_np = boxes.xyxy.detach().cpu().numpy()
            conf_np = boxes.conf.detach().cpu().numpy()
            cls_np = boxes.cls.detach().cpu().numpy().astype(np.int32)

            names_obj = getattr(pred, "names", None)
            names_map = self._names_to_dict(names_obj)
            if names_map:
                self.names = names_map

            features = []
            parsed = []
            for idx in range(len(xyxy_np)):
                x1, y1, x2, y2 = [int(v) for v in xyxy_np[idx]]
                x1 = int(np.clip(x1, 0, img_w - 1))
                y1 = int(np.clip(y1, 0, img_h - 1))
                x2 = int(np.clip(x2, 0, img_w - 1))
                y2 = int(np.clip(y2, 0, img_h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                conf = float(conf_np[idx])
                cls_id = int(cls_np[idx])
                name = self._class_name(cls_id)

                scaled_x1 = (x1 / img_w) * self.ORIG_WIDTH
                scaled_x2 = (x2 / img_w) * self.ORIG_WIDTH
                scaled_y1 = (y1 / img_h) * self.ORIG_HEIGHT
                scaled_y2 = (y2 / img_h) * self.ORIG_HEIGHT

                features.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])
                parsed.append((x1, y1, x2, y2, conf, cls_id, name))

            if not parsed:
                self.distance_cache = {}
                self.xyz_cache = {}
                self.xyz_filter_state = {}
                object_count = 0
                latency_ms = (time.perf_counter() - start) * 1000.0
                fps = 1000.0 / max(latency_ms, 1e-6)
                stats = f"detector={self.detector_label} | objects=0 | latency={latency_ms:.1f}ms | fps~{fps:.1f}"
                result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result_rgb = downscale_frame(result_rgb, MAX_OUTPUT_EDGE)
                scene_state = {
                    "timestamp": time.time(),
                    "image_size": [int(img_w), int(img_h)],
                    "profile_fov_deg": CAM_FOV_DEG,
                    "objects": [],
                    "nearest": None,
                    "depth_enabled": bool(depth_enabled),
                    "prompt_classes": list(self.active_prompt_classes) if self.active_prompt_classes else [],
                }
                if prompt_error:
                    stats += f" | prompt_err={prompt_error}"
                if depth_enabled and depth_error:
                    stats += f" | depth_err={depth_error}"
                return result_rgb, stats, scene_state

            distance_values = self._predict_distances(np.asarray(features, dtype=np.float32))
            conf_vals_np = np.asarray([p[4] for p in parsed], dtype=np.float32)
            area_vals_np = np.asarray([max(1.0, float((p[2] - p[0]) * (p[3] - p[1]))) for p in parsed], dtype=np.float32)
            depth_vals = None
            fusion_w_vals = None
            if depth_map is not None:
                depth_vals = np.asarray(
                    [self._sample_box_depth(depth_map, p[0], p[1], p[2], p[3]) for p in parsed],
                    dtype=np.float32,
                )
                distance_values, _, fusion_w_vals = self._fuse_distance_with_depth(
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

            for idx, (x1, y1, x2, y2, conf, cls_id, name) in enumerate(parsed):
                key = f"{name}:{(x1 + x2) // 80}:{(y1 + y2) // 80}"
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
                    "name": name,
                    "conf": conf,
                    "x": x3,
                    "y": y3,
                    "z": z,
                    "d": d3,
                }
                scene_objects.append(
                    {
                        "name": name,
                        "conf": round(float(conf), 4),
                        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                        "distance_m": round(float(smoothed), 3),
                        "distance3d_m": round(float(d3), 3),
                        "xyz_m": [round(float(x3), 3), round(float(y3), 3), round(float(z), 3)],
                        "depth_rel": round(float(depth_vals[idx]), 4) if depth_vals is not None else None,
                        "fusion_w": round(float(fusion_w_vals[idx]), 4) if fusion_w_vals is not None else None,
                    }
                )
                if nearest_info is None or obj["d"] < nearest_info["d"]:
                    nearest_info = obj

                if depth_map is not None:
                    depth_val = float(depth_vals[idx]) if depth_vals is not None else -1.0
                    fusion_w = float(fusion_w_vals[idx]) if fusion_w_vals is not None else 0.0
                    label = f"{name} {smoothed:.1f}m d{d3:.1f} z{depth_val:.2f} w{fusion_w:.2f}"
                else:
                    label = f"{name} {smoothed:.1f}m d{d3:.1f}"
                color = self._color_for_name(name)
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
            f"detector={self.detector_label} | objects={object_count} | "
            f"latency={latency_ms:.1f}ms | fps~{fps:.1f}"
        )
        if self.active_prompt_classes:
            stats += f" | prompt_cls={len(self.active_prompt_classes)}"
        if prompt_error:
            stats += f" | prompt_err={prompt_error}"
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
        scene_state = {
            "timestamp": time.time(),
            "image_size": [int(img_w), int(img_h)],
            "profile_fov_deg": CAM_FOV_DEG,
            "objects": scene_objects,
            "nearest": nearest_info,
            "depth_enabled": bool(depth_enabled),
            "prompt_classes": list(self.active_prompt_classes) if self.active_prompt_classes else [],
        }
        return result_rgb, stats, scene_state

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
        self._latest_scene = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(
        self,
        frame,
        conf_thres,
        iou_thres,
        img_size,
        max_det,
        smooth_factor,
        depth_enabled,
        depth_alpha,
        depth_interval,
        class_prompt,
    ):
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
            str(class_prompt or ""),
        )
        with self._lock:
            busy_before_submit = self._done_seq < self._pending_seq
            self._pending_frame = frame
            self._pending_params = params
            self._pending_seq += 1
            latest_frame = self._latest_frame
            latest_stats = self._latest_stats

        # Return immediately: never block request path on model inference.
        if busy_before_submit:
            return frame, f"{latest_stats} | live_preview=1"
        if latest_frame is None:
            return frame, latest_stats
        return latest_frame, latest_stats

    def latest_scene(self):
        with self._lock:
            return self._latest_scene

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
                    out_frame, stats, scene_state = engine.infer(frame, *params)
            except Exception as exc:
                out_frame = frame
                stats = f"Loi infer: {exc}"
                scene_state = None

            with self._lock:
                if seq >= self._done_seq:
                    self._done_seq = seq
                    self._latest_frame = out_frame
                    self._latest_stats = stats
                    self._latest_scene = scene_state


def process_frame(
    frame,
    conf_thres,
    iou_thres,
    img_size,
    max_det,
    smooth_factor,
    depth_enabled,
    depth_alpha,
    depth_interval,
    class_prompt,
    session_worker,
):
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
        class_prompt,
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


def _build_local_fallback_plan(user_query: str, scene_state: Dict[str, object]) -> Dict[str, object]:
    objects = (scene_state or {}).get("objects") or []
    nearest = (scene_state or {}).get("nearest") or {}
    target_name = nearest.get("name") if isinstance(nearest, dict) else "target object"
    if objects and not target_name:
        target_name = objects[0].get("name", "target object")

    if not objects:
        return {
            "intent_agent": {"task": user_query or "Reach requested object", "target_object": "unknown", "confidence": 0.2},
            "spatial_agent": {"target_visible": False, "target_xyz_m": [0.0, 0.0, 0.0], "target_distance_m": -1.0, "recommended_approach": "Scan left-center-right slowly."},
            "safety_agent": {
                "risk_level": "medium",
                "hazards": ["Target not visible."],
                "collision_objects": [],
                "safety_rules": ["Move slowly.", "Stop if contact is uncertain."],
            },
            "path_agent": {
                "micro_steps": ["Raise camera slightly.", "Sweep scene slowly.", "Stop when target appears and re-plan."],
                "stop_conditions": ["No target detected for 5 seconds."],
                "fallback_actions": ["Ask nearby person for scene repositioning support."],
            },
            "final_guidance": {
                "summary": "Target is not visible now.",
                "speakable_guidance": ["Target not visible.", "Scan left to right slowly.", "Stop and retry after target appears."],
            },
        }

    x = float(nearest.get("x", 0.0))
    y = float(nearest.get("y", 0.0))
    z = float(nearest.get("z", 0.0))
    d = float(nearest.get("d", z))
    horizontal = "left" if x < -0.2 else ("right" if x > 0.2 else "center")
    vertical = "up" if y < -0.15 else ("down" if y > 0.15 else "center")

    steps = [
        f"Orient hand toward {horizontal} side.",
        f"Adjust hand vertically {vertical}.",
        f"Move forward about {max(0.1, d - 0.35):.2f} meters slowly.",
        "Open fingers and grasp gently.",
    ]
    return {
        "intent_agent": {"task": user_query or f"Reach {target_name}", "target_object": target_name, "confidence": 0.7},
        "spatial_agent": {
            "target_visible": True,
            "target_xyz_m": [round(x, 3), round(y, 3), round(z, 3)],
            "target_distance_m": round(d, 3),
            "recommended_approach": f"Approach from {horizontal}-{vertical} with slow straight-line reach.",
        },
        "safety_agent": {
            "risk_level": "medium" if d < 0.7 else "low",
            "hazards": ["Potential clutter near target."],
            "collision_objects": [o.get("name") for o in objects[:3] if o.get("name") != target_name],
            "safety_rules": ["Slow speed.", "Pause every 10-15 cm.", "Do not sweep sideways quickly."],
        },
        "path_agent": {
            "micro_steps": steps,
            "stop_conditions": ["Target leaves center view.", "Unexpected contact before reaching target."],
            "fallback_actions": ["Pull hand back 10 cm.", "Re-center camera and retry."],
        },
        "final_guidance": {
            "summary": f"Nearest target is {target_name} at about {d:.2f} m.",
            "speakable_guidance": [
                f"{target_name} detected.",
                f"Move {horizontal}, then {vertical}.",
                f"Reach forward about {max(0.1, d - 0.35):.2f} meters slowly.",
            ],
        },
    }


def _render_plan_markdown(plan: Dict[str, object], model_used: str, latency_ms: float, fallback_used: bool) -> str:
    final = plan.get("final_guidance", {}) if isinstance(plan, dict) else {}
    path = plan.get("path_agent", {}) if isinstance(plan, dict) else {}
    safety = plan.get("safety_agent", {}) if isinstance(plan, dict) else {}
    speak = final.get("speakable_guidance", []) if isinstance(final, dict) else []
    steps = path.get("micro_steps", []) if isinstance(path, dict) else []
    hazards = safety.get("hazards", []) if isinstance(safety, dict) else []

    lines = [
        f"Model: `{model_used}` | latency: `{latency_ms:.1f} ms` | fallback: `{fallback_used}`",
        "",
        f"Summary: {final.get('summary', 'N/A') if isinstance(final, dict) else 'N/A'}",
        "",
        "Speakable guidance:",
    ]
    for s in speak[:5]:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("Micro steps:")
    for s in steps[:7]:
        lines.append(f"- {s}")
    if hazards:
        lines.append("")
        lines.append("Hazards:")
        for h in hazards[:5]:
            lines.append(f"- {h}")
    return "\n".join(lines)


def run_smart_planner(user_query: str, profile_name: str, api_key_input: str, session_worker):
    if session_worker is None:
        msg = "Chua co scene state. Bat webcam truoc, doi 1-2s roi plan lai."
        return msg, "{}", session_worker

    scene_state = session_worker.latest_scene()
    if not scene_state:
        msg = "Scene state trong. Hay de webcam chay them mot chut roi bam plan."
        return msg, "{}", session_worker

    planner = GeminiMultiAgentPlanner(api_key=api_key_input.strip() if api_key_input else None)
    result = planner.plan(
        user_query=user_query.strip() if user_query else "Reach the requested object safely.",
        scene_state=scene_state,
        profile_name=profile_name,
        fov_deg=CAM_FOV_DEG,
    )

    fallback_used = False
    if result.ok:
        plan = result.output
        model_used = result.model
        latency_ms = result.latency_ms
    else:
        fallback_used = True
        plan = _build_local_fallback_plan(user_query, scene_state)
        model_used = "local-fallback"
        latency_ms = 0.0

    md = _render_plan_markdown(plan, model_used, latency_ms, fallback_used)
    js = json.dumps(plan, ensure_ascii=False, indent=2)
    return md, js, session_worker


DESCRIPTION = (
    "YOLOER V2 tren Hugging Face Spaces (CPU realtime webcam). "
    "Object detection da chuyen sang YOLOE (THU-MIG) de tang do chinh xac. "
    "Co the nhap prompt classes de tap trung vao nhom vat the muc tieu. "
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
            task_query = gr.Textbox(
                label="Task query for smart agent",
                value="Help me safely reach the nearest cup on the table.",
                lines=2,
            )
            gemini_api_key = gr.Textbox(
                label="Gemini API key (optional, session only)",
                type="password",
                placeholder="Leave empty to use GEMINI_API_KEY env",
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
            class_prompt = gr.Textbox(
                label="YOLOE prompt classes (optional, comma-separated)",
                value="",
                lines=2,
                placeholder="cup, bottle, apple, cell phone",
            )
            depth_enabled = gr.Checkbox(value=True, label="Enable MiDaS v2.1 Small depth")
            depth_alpha = gr.Slider(0.0, 0.7, value=0.18, step=0.01, label="Depth overlay alpha")
            depth_interval = gr.Slider(1, 8, value=5, step=1, label="Depth update every N frames")
        with gr.Column(scale=1):
            result = gr.Image(label="Result", type="numpy", format="jpeg", height=300)
            stats = gr.Textbox(label="Runtime stats")
            plan_btn = gr.Button("Generate Smart Guidance (Gemini)")
            plan_md = gr.Markdown("Guidance plan will appear here.")
            plan_json = gr.Code(label="Planner JSON", language="json")
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
            class_prompt,
            session_worker,
        ],
        outputs=[result, stats, session_worker],
        **stream_kwargs,
    )

    plan_btn.click(
        run_smart_planner,
        inputs=[task_query, profile, gemini_api_key, session_worker],
        outputs=[plan_md, plan_json, session_worker],
        queue=False,
    )


if __name__ == "__main__":
    demo.launch()
