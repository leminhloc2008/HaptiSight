import os
import inspect
import threading
import time
import json
import base64
import re
import html
from difflib import SequenceMatcher
from collections import deque
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import cv2
import gradio as gr
import h5py
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLOE
try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None
try:
    from mediapipe.python.solutions import hands as mp_hands  # type: ignore
except Exception:
    mp_hands = None

ROOT = Path(__file__).resolve().parent
YOLO_DIR = ROOT / "distance_estimation_core"
MAX_FRAME_EDGE = int(os.getenv("MAX_FRAME_EDGE", "448"))
MAX_OUTPUT_EDGE = int(os.getenv("MAX_OUTPUT_EDGE", "320"))
MAX_DEPTH_EDGE = int(os.getenv("MAX_DEPTH_EDGE", "256"))
CAM_FOV_DEG = float(os.getenv("CAM_FOV_DEG", "70.0"))
WEBCAM_CAPTURE_W = int(os.getenv("WEBCAM_CAPTURE_W", "640"))
WEBCAM_CAPTURE_H = int(os.getenv("WEBCAM_CAPTURE_H", "360"))
WEBCAM_CAPTURE_FPS = int(os.getenv("WEBCAM_CAPTURE_FPS", "24"))
OUTPUT_JPEG_QUALITY = int(os.getenv("OUTPUT_JPEG_QUALITY", "58"))
STREAM_EVERY_SEC = float(os.getenv("STREAM_EVERY_SEC", "0.08"))
PROMPT_CONF_CAP = float(os.getenv("PROMPT_CONF_CAP", "0.28"))
PROMPT_CONF_RETRY = float(os.getenv("PROMPT_CONF_RETRY", "0.20"))
PROMPT_IMG_SIZE = int(os.getenv("PROMPT_IMG_SIZE", "640"))
PROMPT_FORCE_FP32 = os.getenv("PROMPT_FORCE_FP32", "1").strip().lower() in {"1", "true", "yes"}
ACCURACY_RETRY_ENABLED = os.getenv("ACCURACY_RETRY_ENABLED", "1").strip().lower() in {"1", "true", "yes"}
ACCURACY_RETRY_CONF = float(os.getenv("ACCURACY_RETRY_CONF", "0.16"))
ACCURACY_RETRY_IMG = int(os.getenv("ACCURACY_RETRY_IMG", "768"))
APP_BUILD = os.getenv("APP_BUILD", "2026-02-13-enact11-mediapipe-handonly")
GUIDE_MIN_INTERVAL_SEC = float(os.getenv("GUIDE_MIN_INTERVAL_SEC", "0.8"))
GUIDE_MAX_INTERVAL_SEC = float(os.getenv("GUIDE_MAX_INTERVAL_SEC", "6.0"))
GUIDE_MAX_TEXT_CHARS = int(os.getenv("GUIDE_MAX_TEXT_CHARS", "260"))
GUIDE_GEMINI_MIN_INTERVAL_SEC = float(os.getenv("GUIDE_GEMINI_MIN_INTERVAL_SEC", "2.8"))
GUIDE_GEMINI_BACKOFF_BASE_SEC = float(os.getenv("GUIDE_GEMINI_BACKOFF_BASE_SEC", "8.0"))
GUIDE_GEMINI_BACKOFF_MAX_SEC = float(os.getenv("GUIDE_GEMINI_BACKOFF_MAX_SEC", "120.0"))
GUIDE_GEMINI_RPM_LIMIT = int(os.getenv("GUIDE_GEMINI_RPM_LIMIT", "5"))
GUIDE_GEMINI_HOURLY_LIMIT = int(os.getenv("GUIDE_GEMINI_HOURLY_LIMIT", "120"))
GUIDE_GEMINI_SCENE_FORCE_SEC = float(os.getenv("GUIDE_GEMINI_SCENE_FORCE_SEC", "20.0"))
GUIDE_GEMINI_ALWAYS_ON = os.getenv("GUIDE_GEMINI_ALWAYS_ON", "1").strip().lower() in {"1", "true", "yes"}
GUIDE_MIN_VOICE_EMIT_SEC = float(os.getenv("GUIDE_MIN_VOICE_EMIT_SEC", "2.4"))
GUIDE_REPEAT_SIM_THRESHOLD = float(os.getenv("GUIDE_REPEAT_SIM_THRESHOLD", "0.90"))
GUIDE_TARGET_LOCK_SEC = float(os.getenv("GUIDE_TARGET_LOCK_SEC", "4.0"))
GUIDE_SAY_HISTORY = max(3, int(os.getenv("GUIDE_SAY_HISTORY", "8")))
TARGET_QUERY_MAX_CLASSES = int(os.getenv("TARGET_QUERY_MAX_CLASSES", "6"))
GUIDE_DEPTH_HAZARD_X_M = float(os.getenv("GUIDE_DEPTH_HAZARD_X_M", "0.24"))
GUIDE_DEPTH_HAZARD_Y_M = float(os.getenv("GUIDE_DEPTH_HAZARD_Y_M", "0.20"))
GUIDE_DEPTH_HAZARD_FRONT_M = float(os.getenv("GUIDE_DEPTH_HAZARD_FRONT_M", "0.34"))
GUIDE_DEPTH_HAZARD_BEHIND_M = float(os.getenv("GUIDE_DEPTH_HAZARD_BEHIND_M", "0.24"))
HAND_DETECT_ENABLED = os.getenv("HAND_DETECT_ENABLED", "1").strip().lower() in {"1", "true", "yes"}
HAND_DETECT_EVERY_N = max(1, int(os.getenv("HAND_DETECT_EVERY_N", "3")))
HAND_MAX_EDGE = int(os.getenv("HAND_MAX_EDGE", "320"))
HAND_MIN_SCORE = float(os.getenv("HAND_MIN_SCORE", "0.35"))
HAND_SMOOTH_ALPHA = float(os.getenv("HAND_SMOOTH_ALPHA", "0.55"))
HAND_TARGET_REACH_M = float(os.getenv("HAND_TARGET_REACH_M", "0.12"))
HAND_CONTACT_DIST_M = float(os.getenv("HAND_CONTACT_DIST_M", "0.06"))
HAND_CONTACT_IOU = float(os.getenv("HAND_CONTACT_IOU", "0.18"))
HAND_DETECT_MODE = str(os.getenv("HAND_DETECT_MODE", "mediapipe")).strip().lower()
HAND_YOLOE_ENABLED = os.getenv("HAND_YOLOE_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
HAND_YOLOE_EVERY_N = max(1, int(os.getenv("HAND_YOLOE_EVERY_N", "5")))
HAND_YOLOE_IMG_SIZE = max(224, int(os.getenv("HAND_YOLOE_IMG_SIZE", "320")))
HAND_YOLOE_CONF = float(os.getenv("HAND_YOLOE_CONF", "0.22"))
HAND_YOLOE_MODEL_ID = str(os.getenv("HAND_YOLOE_MODEL_ID", "yoloe-v8s")).strip() or "yoloe-v8s"

from smart_agent import GeminiMultiAgentPlanner  # noqa: E402

DEFAULT_COCO80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

TARGET_ALIASES: Dict[str, List[str]] = {
    "cup": ["cup", "mug", "glass", "water cup", "coffee cup"],
    "bottle": ["bottle", "water bottle", "drink bottle", "flask"],
    "apple": ["apple"],
    "banana": ["banana"],
    "orange": ["orange"],
    "cell phone": ["phone", "cell phone", "mobile phone", "smartphone", "iphone"],
    "remote": ["remote", "controller", "tv remote"],
    "book": ["book", "notebook"],
    "keyboard": ["keyboard"],
    "mouse": ["mouse"],
    "laptop": ["laptop", "notebook computer"],
    "person": ["person", "human", "man", "woman"],
    "chair": ["chair", "seat", "stool"],
    "backpack": ["backpack", "bag", "school bag"],
}

DANGEROUS_OBJECT_ALIASES: Dict[str, List[str]] = {
    "knife": ["knife", "kitchen knife", "blade"],
    "scissors": ["scissors", "shears"],
    "oven": ["oven", "stove", "cooktop"],
    "microwave": ["microwave"],
    "toaster": ["toaster"],
    "fire": ["fire", "flame", "stove flame"],
    "hot cup": ["hot cup", "hot mug", "coffee cup"],
    "glass": ["wine glass", "glass"],
}

TARGET_QUERY_PATTERNS = [
    r"(?:reach|grab|get|take|pick up|find|locate)\s+(?:the\s+|a\s+|an\s+)?([a-z][a-z0-9\-\s]{1,40})",
    r"(?:target|object)\s*(?:is|:)?\s*([a-z][a-z0-9\-\s]{1,40})",
]
DISALLOWED_PROMPT_CLASSES = {"hand", "hands", "palm", "human hand"}


def _norm_label(text: str) -> str:
    s = str(text or "").strip().lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def _canonical_target_name(name: str) -> str:
    query = _norm_label(name)
    if not query:
        return ""
    for canonical, aliases in TARGET_ALIASES.items():
        cand = {_norm_label(canonical)}
        cand.update(_norm_label(a) for a in aliases)
        if query in cand:
            return canonical
    for canonical, aliases in TARGET_ALIASES.items():
        cand = {_norm_label(canonical)}
        cand.update(_norm_label(a) for a in aliases)
        if any(re.search(rf"\b{re.escape(c)}\b", query) for c in cand if c):
            return canonical
    return query


def _cleanup_target_phrase(text: str) -> str:
    s = _norm_label(text)
    if not s:
        return ""
    s = re.sub(r"^(the|a|an)\s+", "", s).strip()
    s = re.split(r"\b(on|in|at|near|with|from|to|for|and)\b", s, maxsplit=1)[0].strip()
    words = [w for w in s.split() if w]
    if len(words) > 3:
        words = words[:3]
    return " ".join(words)


def _target_alias_set(name: str) -> set:
    canonical = _canonical_target_name(name)
    if not canonical:
        return set()
    out = {_norm_label(canonical)}
    out.update(_norm_label(a) for a in TARGET_ALIASES.get(canonical, []))
    return {x for x in out if x}


def _target_matches_name(target_name: str, obj_name: str) -> bool:
    t = _norm_label(target_name)
    o = _norm_label(obj_name)
    if not t or not o:
        return False
    alias = _target_alias_set(t)
    if o in alias:
        return True
    if t in _target_alias_set(o):
        return True
    return (t in o and len(t) >= 4) or (o in t and len(o) >= 4)


def _dangerous_object_type(name: str) -> str:
    n = _norm_label(name)
    if not n:
        return ""
    for kind, aliases in DANGEROUS_OBJECT_ALIASES.items():
        cands = {_norm_label(kind)}
        cands.update(_norm_label(a) for a in aliases)
        if any((c and ((n == c) or (c in n and len(c) >= 4))) for c in cands):
            return kind
    return ""


def _is_dangerous_object_name(name: str) -> bool:
    return bool(_dangerous_object_type(name))


RISK_SCORE_PRIOR: Dict[str, float] = {
    "knife": 1.00,
    "scissors": 0.92,
    "glass": 0.82,
    "oven": 0.90,
    "microwave": 0.72,
    "toaster": 0.68,
    "fire": 1.00,
    "hot cup": 0.78,
}


def _risk_score_for_object(name: str, danger_type: str = "") -> float:
    dt = _norm_label(danger_type)
    nm = _norm_label(name)
    if dt in RISK_SCORE_PRIOR:
        return float(RISK_SCORE_PRIOR[dt])
    for k, v in RISK_SCORE_PRIOR.items():
        if k in nm:
            return float(v)
    return 0.18


def _parse_class_prompt_text(class_prompt: str) -> List[str]:
    raw = str(class_prompt or "").replace("\n", ",").replace(";", ",")
    items = [x.strip() for x in raw.split(",") if x and x.strip()]
    out = []
    seen = set()
    for item in items:
        c = _canonical_target_name(item)
        if _norm_label(c) in DISALLOWED_PROMPT_CLASSES:
            continue
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _extract_requested_targets(user_query: str, class_prompt: str = "") -> List[str]:
    requested: List[str] = []
    seen = set()

    for cls in _parse_class_prompt_text(class_prompt):
        if cls not in seen:
            seen.add(cls)
            requested.append(cls)

    norm_q = _norm_label(user_query)
    if norm_q:
        # First, parse explicit target phrase patterns to prioritize intended object.
        for pattern in TARGET_QUERY_PATTERNS:
            m = re.search(pattern, norm_q, flags=re.IGNORECASE)
            if not m:
                continue
            phrase = _canonical_target_name(_cleanup_target_phrase(m.group(1)))
            if phrase and phrase not in seen:
                seen.add(phrase)
                requested.append(phrase)

        # Then add other mentioned classes by first appearance in query text.
        alias_hits: List[Tuple[int, str]] = []
        for canonical, aliases in TARGET_ALIASES.items():
            cands = {_norm_label(canonical)}
            cands.update(_norm_label(a) for a in aliases)
            idxs = [norm_q.find(c) for c in cands if c and norm_q.find(c) >= 0]
            if idxs:
                alias_hits.append((min(idxs), canonical))
        alias_hits.sort(key=lambda x: x[0])
        for _, canonical in alias_hits:
            if canonical not in seen:
                seen.add(canonical)
                requested.append(canonical)

    return requested[: max(1, int(TARGET_QUERY_MAX_CLASSES))]


def _merge_prompt_classes(class_prompt: str, user_query: str, auto_target_prompt: bool) -> str:
    manual = _parse_class_prompt_text(class_prompt)
    if not auto_target_prompt:
        return ", ".join(manual)
    merged: List[str] = []
    seen = set()
    for name in manual + _extract_requested_targets(user_query, ""):
        c = _canonical_target_name(name)
        if c and c not in seen:
            seen.add(c)
            merged.append(c)
    return ", ".join(merged[: max(1, int(TARGET_QUERY_MAX_CLASSES))])


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
    def __init__(self, device: torch.device, model_name: str = "MiDaS_small", use_half: bool = False):
        self.device = device
        self.model_name = model_name
        self.use_half = bool(use_half and device.type == "cuda")
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
            self.model = self._hub_load("intel-isl/MiDaS", self.model_name)
            transforms = self._hub_load("intel-isl/MiDaS", "transforms")
            if self.model_name == "MiDaS_small":
                self.transform = transforms.small_transform
            else:
                self.transform = transforms.dpt_transform
            self.model.to(self.device).eval()
            if self.use_half:
                self.model.half()
        except Exception as exc:
            self.error_message = f"Failed to load {self.model_name}: {exc}"
            raise RuntimeError(self.error_message)

    def predict(self, frame_rgb: np.ndarray, max_edge: int) -> np.ndarray:
        self.ensure_loaded()
        frame_input = downscale_frame(frame_rgb, max_edge)

        with torch.inference_mode():
            inp = self.transform(frame_input).to(self.device)
            if self.use_half:
                inp = inp.half()
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


class HandTracker:
    def __init__(self):
        self.error_message: Optional[str] = None
        self._hands = None
        self.backend = "unavailable"
        self._last_bbox: Optional[Tuple[int, int, int, int]] = None
        if mp_hands is not None:
            self.backend = "mediapipe"
        elif mp is not None and getattr(getattr(mp, "solutions", None), "hands", None) is not None:
            self.backend = "mediapipe"
        else:
            self.error_message = "mediapipe_unavailable"

    def ensure_loaded(self):
        if self.backend != "mediapipe":
            return
        if self._hands is not None:
            return
        if self.error_message is not None:
            raise RuntimeError(self.error_message)
        try:
            hands_api = mp_hands
            if hands_api is None:
                hands_api = getattr(getattr(mp, "solutions", None), "hands", None)
            if hands_api is None:
                raise RuntimeError("mediapipe hands solution unavailable")
            self._hands = hands_api.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.50,
            )
        except Exception as exc:
            self.error_message = f"hand_tracker_init_err={exc}"
            raise RuntimeError(self.error_message)

    @staticmethod
    def _detect_skin_hand(frame_rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        h, w = frame_rgb.shape[:2]
        if h < 4 or w < 4:
            return None
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask_y = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 180, 135], dtype=np.uint8))
        mask_h1 = cv2.inRange(hsv, np.array([0, 20, 55], dtype=np.uint8), np.array([25, 185, 255], dtype=np.uint8))
        mask_h2 = cv2.inRange(hsv, np.array([160, 20, 55], dtype=np.uint8), np.array([180, 185, 255], dtype=np.uint8))
        mask = cv2.bitwise_and(mask_y, cv2.bitwise_or(mask_h1, mask_h2))
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        img_area = float(h * w)
        best = None
        best_area = 0.0
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 0.008 * img_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 18 or bh < 18:
                continue
            aspect = bw / max(float(bh), 1.0)
            if aspect < 0.35 or aspect > 2.8:
                continue
            if area > best_area:
                best_area = area
                best = (x, y, x + bw, y + bh)
        if best is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in best]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        score = float(np.clip(best_area / max(img_area * 0.22, 1.0), 0.20, 0.75))
        return {
            "visible": True,
            "bbox_xyxy": [x1, y1, x2, y2],
            "center_xy": [cx, cy],
            "score": score,
            "handedness": "",
            "source": "opencv_skin",
        }

    def detect(self, frame_rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        if self.backend != "mediapipe":
            return None
        self.ensure_loaded()
        assert self._hands is not None
        small = downscale_frame(frame_rgb, max(160, int(HAND_MAX_EDGE)))
        sh, sw = small.shape[:2]
        oh, ow = frame_rgb.shape[:2]
        sx = ow / max(sw, 1)
        sy = oh / max(sh, 1)

        result = self._hands.process(small)
        if not result or not result.multi_hand_landmarks:
            return None

        lm = result.multi_hand_landmarks[0]
        handed = ""
        handed_score = 0.0
        try:
            handed_info = (result.multi_handedness or [None])[0]
            if handed_info and handed_info.classification:
                handed = str(handed_info.classification[0].label)
                handed_score = float(handed_info.classification[0].score)
        except Exception:
            handed = ""
            handed_score = 0.0

        pts = [(float(p.x) * sw, float(p.y) * sh) for p in lm.landmark]
        if not pts:
            return None
        xs = np.asarray([p[0] for p in pts], dtype=np.float32)
        ys = np.asarray([p[1] for p in pts], dtype=np.float32)
        x1 = max(0.0, float(np.min(xs) - 8.0))
        y1 = max(0.0, float(np.min(ys) - 8.0))
        x2 = min(float(sw - 1), float(np.max(xs) + 8.0))
        y2 = min(float(sh - 1), float(np.max(ys) + 8.0))
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        score = max(float(handed_score), 0.5)
        if score < float(HAND_MIN_SCORE):
            return None

        return {
            "visible": True,
            "bbox_xyxy": [int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy))],
            "center_xy": [float(cx * sx), float(cy * sy)],
            "score": float(np.clip(score, 0.0, 1.0)),
            "handedness": handed,
            "thumb_tip_xy": [float(pts[4][0] * sx), float(pts[4][1] * sy)] if len(pts) > 8 else None,
            "index_tip_xy": [float(pts[8][0] * sx), float(pts[8][1] * sy)] if len(pts) > 8 else None,
            "pinch_ratio": (
                float(
                    np.clip(
                        np.sqrt((pts[8][0] - pts[4][0]) ** 2 + (pts[8][1] - pts[4][1]) ** 2)
                        / max(8.0, np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)),
                        0.0,
                        2.0,
                    )
                )
                if len(pts) > 8
                else None
            ),
            "source": "mediapipe",
        }


class RealtimeEngine:
    ORIG_HEIGHT = 375.0
    ORIG_WIDTH = 1242.0
    YOLOE_REPO_ID = "jameslahm/yoloe"
    CPU_DEFAULT_MODEL_ID = "yoloe-11s"
    GPU_DEFAULT_MODEL_ID = "yoloe-11m"
    FALLBACK_MODEL_ID = "yoloe-v8s"

    def __init__(self):
        force_cpu = os.getenv("FORCE_CPU", "0").strip().lower() in {"1", "true", "yes"}
        self.use_cuda = (not force_cpu) and torch.cuda.is_available()
        self.device = "cuda:0" if self.use_cuda else "cpu"
        self.torch_device = torch.device(self.device)
        self.use_half = bool(self.use_cuda and os.getenv("USE_FP16", "1").strip().lower() in {"1", "true", "yes"})

        env_threads = os.getenv("CPU_THREADS", "").strip()
        if env_threads:
            requested_threads = max(1, int(env_threads))
        else:
            # Default auto-tuning for HF CPU: leave 1 core for system/Gradio threads.
            cpu_count = max(2, (os.cpu_count() or 4))
            requested_threads = max(2, min(8, cpu_count - 1))
        if not self.use_cuda:
            try:
                torch.set_num_threads(requested_threads)
                torch.set_num_interop_threads(1)
                torch.set_flush_denormal(True)
            except RuntimeError:
                pass
        else:
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        self.cpu_threads = requested_threads
        self.weights_path = self._resolve_weights_path()
        self.detector_label = Path(self.weights_path).name
        self.distance_model = DistanceMLP(YOLO_DIR / "model@1535470106.h5")
        self.model: Optional[YOLOE] = None
        self.base_names: Dict[int, str] = {}
        self.names: Dict[int, str] = {}
        self.default_vocab_active = False
        self.default_vocab_error: Optional[str] = None
        self.active_prompt_classes: Optional[Tuple[str, ...]] = None
        self.active_prompt_error: Optional[str] = None
        self.prompt_fp32_active = False
        self.prompt_fail_streak = 0
        self.prompt_retry_after_ts = 0.0
        self.color_cache: Dict[str, Tuple[int, int, int]] = {}
        self._load_detector()
        self.distance_cache: Dict[str, float] = {}
        self.xyz_cache: Dict[str, Tuple[float, float, float]] = {}
        self.xyz_filter_state: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        self.obj_track_state: Dict[str, Dict[str, float]] = {}
        self.depth_estimator = None
        self.depth_last_map = None
        self.depth_last_shape = None
        self.depth_last_latency_ms = 0.0
        self.depth_frame_counter = 0
        self.depth_last_update_ts = 0.0
        self.depth_last_error: Optional[str] = None
        self.depth_job_busy = False
        self.depth_lock = threading.Lock()
        self.det_frame_counter = 0
        self.miss_streak = 0
        self.hires_refresh_every = max(2, int(os.getenv("HIRES_REFRESH_EVERY", "6")))
        self.fast_size_delta = max(0, int(os.getenv("FAST_SIZE_DELTA", "64")))
        self.recovery_img_boost = max(0, int(os.getenv("RECOVERY_IMG_BOOST", "64")))
        self.recovery_miss_threshold = max(1, int(os.getenv("RECOVERY_MISS_THRESHOLD", "2")))
        default_depth_model = "DPT_Hybrid" if self.use_cuda else "MiDaS_small"
        self.depth_model_name = os.getenv("DEPTH_MODEL", default_depth_model).strip() or default_depth_model
        self.depth_use_half = bool(self.use_half and os.getenv("DEPTH_FP16", "1").strip().lower() in {"1", "true", "yes"})
        self.hand_tracker = HandTracker() if HAND_DETECT_ENABLED else None
        self.hand_frame_counter = 0
        self.hand_last_state: Dict[str, Any] = {"visible": False, "source": "disabled"}
        self.hand_last_error: Optional[str] = None
        self.hand_xyz_cache: Optional[Tuple[float, float, float]] = None
        self.hand_detect_mode = "mediapipe"
        self.hand_mode_forced = HAND_DETECT_MODE not in {"", "mediapipe"}
        self.hand_yolo_enabled = False
        self.hand_yolo_model: Optional[YOLOE] = None
        self.hand_yolo_error: Optional[str] = None
        self.hand_yolo_frame_counter = 0
        if self.hand_mode_forced:
            self.hand_yolo_error = "hand_mode_forced=mediapipe"
        if self.hand_tracker is not None and getattr(self.hand_tracker, "backend", "") != "mediapipe":
            self.hand_last_error = "hand_backend_unavailable=mediapipe"

    def _select_adaptive_infer_settings(self, base_img_size: int, base_conf: float) -> Tuple[int, float, str]:
        self.det_frame_counter += 1
        base_img_size = int(np.clip(base_img_size, 128, 640))
        base_img_size = max(128, (base_img_size // 32) * 32)
        fast_delta = 0 if self.use_cuda else self.fast_size_delta
        fast_img = max(128, ((base_img_size - fast_delta) // 32) * 32)

        periodic_hires = (self.det_frame_counter % self.hires_refresh_every) == 0
        recovery_hires = (
            self.miss_streak >= self.recovery_miss_threshold
            and (self.det_frame_counter % 4 == 0)
        )
        use_hires = periodic_hires or recovery_hires

        run_img = base_img_size if use_hires else fast_img
        run_conf = float(base_conf)
        mode = "fast"
        if use_hires:
            mode = "hires"
        if self.miss_streak >= self.recovery_miss_threshold:
            run_conf = max(0.16 if self.use_cuda else 0.20, float(base_conf) - 0.06)
        if recovery_hires:
            run_img = min(640, max(run_img, base_img_size + self.recovery_img_boost))
            run_img = max(128, (run_img // 32) * 32)
            run_conf = max(0.14 if self.use_cuda else 0.18, float(base_conf) - 0.10)
            mode = "recover"
        return int(run_img), float(run_conf), mode

    @staticmethod
    def _model_filename(model_id: str) -> str:
        name = (model_id or "").strip()
        if not name:
            name = RealtimeEngine.CPU_DEFAULT_MODEL_ID
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
                f"Cannot find local weights for '{env_weight}'. "
                "Set YOLOE_WEIGHTS to a valid local path or URL."
            )

        default_model = self.GPU_DEFAULT_MODEL_ID if self.use_cuda else self.CPU_DEFAULT_MODEL_ID
        requested_model = os.getenv("YOLOE_MODEL_ID", default_model).strip() or default_model
        requested_file = self._model_filename(requested_model)
        try:
            return hf_hub_download(repo_id=self.YOLOE_REPO_ID, filename=requested_file)
        except Exception as exc:
            fallback_file = self._model_filename(self.FALLBACK_MODEL_ID)
            if requested_file == fallback_file:
                raise RuntimeError(f"Cannot download YOLOE checkpoint '{requested_file}': {exc}") from exc
            try:
                return hf_hub_download(repo_id=self.YOLOE_REPO_ID, filename=fallback_file)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Cannot download YOLOE checkpoint '{requested_file}' and fallback '{fallback_file}'. "
                    f"Details: {exc} | {fallback_exc}"
                ) from fallback_exc

    def _resolve_weights_for_model_id(self, model_id: str) -> str:
        filename = self._model_filename(model_id)
        return hf_hub_download(repo_id=self.YOLOE_REPO_ID, filename=filename)

    def _load_hand_yolo_detector(self) -> None:
        if not self.hand_yolo_enabled or self.hand_detect_mode not in {"yoloe", "hybrid"}:
            return
        if self.hand_yolo_model is not None:
            return
        try:
            weights = self._resolve_weights_for_model_id(HAND_YOLOE_MODEL_ID)
            model = YOLOE(weights)
            model.to(self.device).eval()
            # Keep fp32 for prompt embeddings to avoid dtype mismatch in prompt setup.
            if hasattr(model, "model"):
                try:
                    model.model.float()  # type: ignore[attr-defined]
                except Exception:
                    pass
            hand_classes = ["hand", "palm", "human hand"]
            embeddings = model.get_text_pe(hand_classes)
            embeddings = embeddings.to(device=self.torch_device, dtype=torch.float32)
            model.set_classes(hand_classes, embeddings)
            self.hand_yolo_model = model
            self.hand_yolo_error = None
        except Exception as exc:
            self.hand_yolo_error = f"hand_yolo_load_err={exc}"
            self.hand_yolo_model = None
            raise RuntimeError(self.hand_yolo_error)

    def _detect_hand_yoloe(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        # Hand detection is intentionally Mediapipe-only for realtime stability.
        # YOLOE remains object-only in this app.
        return None

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

    @staticmethod
    def _names_are_numeric_placeholders(names_map: Dict[int, str]) -> bool:
        if not names_map:
            return True
        values = [str(v).strip() for v in names_map.values()]
        if not values:
            return True
        numeric_count = sum(v.isdigit() for v in values)
        return numeric_count >= max(5, int(0.8 * len(values)))

    def _ensure_default_vocab(self) -> None:
        assert self.model is not None
        self.default_vocab_active = False
        self.default_vocab_error = None
        if not self._names_are_numeric_placeholders(self.base_names):
            return
        classes = list(DEFAULT_COCO80_CLASSES)
        switched_to_fp32 = False
        try:
            if self.use_half and self.use_cuda and PROMPT_FORCE_FP32 and hasattr(self.model, "model"):
                try:
                    self.model.model.float()
                    switched_to_fp32 = True
                except Exception:
                    switched_to_fp32 = False
            embeddings = self.model.get_text_pe(classes)
            model_dtype = torch.float16 if self.use_half else torch.float32
            try:
                model_dtype = next(self.model.model.parameters()).dtype  # type: ignore[attr-defined]
            except Exception:
                pass
            embeddings = embeddings.to(device=self.torch_device, dtype=model_dtype)
            self.model.set_classes(classes, embeddings)
            self.base_names = self._names_to_dict(getattr(self.model, "names", None))
            self.names = dict(self.base_names)
            self.default_vocab_active = True
        except Exception as exc:
            self.default_vocab_error = f"default_vocab_err={exc}"
        finally:
            if switched_to_fp32 and self.use_half and hasattr(self.model, "model"):
                try:
                    self.model.model.half()
                except Exception:
                    pass

    def _load_detector(self) -> None:
        self.model = YOLOE(self.weights_path)
        self.model.to(self.device)
        self.model.eval()
        if self.use_half and hasattr(self.model, "model"):
            try:
                self.model.model.half()
            except Exception:
                self.use_half = False
        # One tiny warmup pass to reduce first-frame latency spikes.
        try:
            warm = np.zeros((160, 160, 3), dtype=np.uint8)
            self.model.predict(
                source=warm,
                imgsz=160,
                conf=0.5,
                iou=0.45,
                max_det=1,
                device=self.device,
                half=self.use_half,
                verbose=False,
            )
        except Exception:
            pass
        self.base_names = self._names_to_dict(getattr(self.model, "names", None))
        self.names = dict(self.base_names)
        self._ensure_default_vocab()
        self.active_prompt_classes = None
        self.active_prompt_error = None
        self.prompt_fp32_active = False
        self.prompt_fail_streak = 0
        self.prompt_retry_after_ts = 0.0

    @staticmethod
    def _parse_prompt_classes(class_prompt: str) -> List[str]:
        raw_input = class_prompt
        if isinstance(raw_input, dict):
            raw_input = raw_input.get("value") or raw_input.get("text") or ""
        elif isinstance(raw_input, (list, tuple)):
            raw_input = ",".join(str(x) for x in raw_input if x is not None)
        raw = str(raw_input or "").replace("\n", ",").replace(";", ",")
        items = [x.strip() for x in raw.split(",") if x and x.strip()]
        unique = []
        seen = set()
        for item in items:
            normalized = _canonical_target_name(item)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
        return unique

    def _apply_prompt_classes_if_needed(self, class_prompt: str) -> Optional[str]:
        classes = self._parse_prompt_classes(class_prompt)
        if not classes:
            if self.active_prompt_classes is not None:
                self._load_detector()
            self.prompt_fail_streak = 0
            self.prompt_retry_after_ts = 0.0
            return None

        cls_tuple = tuple(classes)
        now_ts = time.time()
        if self.active_prompt_classes == cls_tuple and self.active_prompt_error is None:
            return self.active_prompt_error
        if (
            self.active_prompt_classes == cls_tuple
            and self.active_prompt_error is not None
            and now_ts < float(self.prompt_retry_after_ts)
        ):
            return self.active_prompt_error

        assert self.model is not None
        try:
            # YOLOE prompt projection can fail on mixed dtypes in CUDA fp16 mode.
            # Run prompt setup in fp32 for stability, then infer in fp32 while prompt is active.
            if self.use_half and self.use_cuda and PROMPT_FORCE_FP32 and hasattr(self.model, "model"):
                try:
                    self.model.model.float()
                    self.prompt_fp32_active = True
                except Exception:
                    self.prompt_fp32_active = False
            else:
                self.prompt_fp32_active = False

            embeddings = self.model.get_text_pe(classes)
            model_dtype = torch.float16 if self.use_half else torch.float32
            try:
                model_dtype = next(self.model.model.parameters()).dtype  # type: ignore[attr-defined]
            except Exception:
                pass
            if self.prompt_fp32_active:
                model_dtype = torch.float32
            embeddings = embeddings.to(device=self.torch_device, dtype=model_dtype)
            self.model.set_classes(classes, embeddings)
            self.names = self._names_to_dict(getattr(self.model, "names", classes))
            self.active_prompt_classes = cls_tuple
            self.active_prompt_error = None
            self.prompt_fail_streak = 0
            self.prompt_retry_after_ts = 0.0
        except Exception as exc:
            # Preserve error for UI stats, and allow retry on next calls.
            err_msg = f"Cannot set YOLOE prompt classes: {exc}"
            self.active_prompt_classes = cls_tuple
            self.active_prompt_error = err_msg
            self.prompt_fail_streak += 1
            cooldown = min(12.0, 2.0 + 1.6 * float(self.prompt_fail_streak))
            self.prompt_retry_after_ts = time.time() + cooldown
            # Keep detector usable even if prompt embedding setup fails.
            try:
                self._load_detector()
                self.active_prompt_classes = cls_tuple
                self.active_prompt_error = err_msg
            except Exception:
                pass
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

    def _depth_job(self, frame_rgb: np.ndarray) -> None:
        try:
            if self.depth_estimator is None:
                self.depth_estimator = MidasDepthEstimator(
                    self.torch_device,
                    model_name=self.depth_model_name,
                    use_half=self.depth_use_half,
                )
            t0 = time.perf_counter()
            depth_map = self.depth_estimator.predict(frame_rgb, MAX_DEPTH_EDGE)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            with self.depth_lock:
                self.depth_last_map = depth_map
                self.depth_last_shape = tuple(int(v) for v in frame_rgb.shape[:2])
                self.depth_last_latency_ms = latency_ms
                self.depth_last_update_ts = time.time()
                self.depth_last_error = None
        except Exception as exc:
            with self.depth_lock:
                self.depth_last_error = str(exc)
        finally:
            with self.depth_lock:
                self.depth_job_busy = False

    def _get_depth_map(self, frame_rgb: np.ndarray, depth_enabled: bool, depth_interval: int):
        if not depth_enabled:
            with self.depth_lock:
                self.depth_last_map = None
                self.depth_last_error = None
            return None, None

        self.depth_frame_counter += 1
        depth_interval = max(1, int(depth_interval))
        with self.depth_lock:
            frame_shape = tuple(int(v) for v in frame_rgb.shape[:2])
            now_ts = time.time()
            age_s = (now_ts - self.depth_last_update_ts) if self.depth_last_update_ts > 0 else 1e9
            max_age_s = max(2.0, 0.5 * float(depth_interval))
            has_map = (
                (self.depth_last_map is not None)
                and (self.depth_last_shape == frame_shape)
                and (age_s <= max_age_s)
            )
            refresh = (not has_map) or (self.depth_frame_counter % depth_interval == 0)
            if refresh and (not self.depth_job_busy):
                self.depth_job_busy = True
                job_frame = frame_rgb.copy()
                threading.Thread(target=self._depth_job, args=(job_frame,), daemon=True).start()
            depth_map = self.depth_last_map if has_map else None
            depth_err = self.depth_last_error
        return depth_map, depth_err

    def _detect_hand(self, frame_rgb: np.ndarray) -> Dict[str, Any]:
        if self.hand_tracker is None:
            self.hand_last_state = {"visible": False, "source": "disabled"}
            return dict(self.hand_last_state)

        self.hand_frame_counter += 1
        must_refresh = (self.hand_frame_counter % max(1, int(HAND_DETECT_EVERY_N))) == 0
        if (not must_refresh) and bool(self.hand_last_state.get("visible", False)):
            return dict(self.hand_last_state)

        det = None
        try:
            det = self.hand_tracker.detect(frame_rgb)
        except Exception as exc:
            self.hand_last_error = str(exc)

        if det is not None:
            self.hand_last_state = det
            self.hand_last_error = None
        else:
            self.hand_last_state = {"visible": False, "source": "mediapipe"}
        return dict(self.hand_last_state)

    @staticmethod
    def _smooth_hand_xyz(prev_xyz: Optional[Tuple[float, float, float]], new_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if prev_xyz is None:
            return new_xyz
        a = float(np.clip(HAND_SMOOTH_ALPHA, 0.0, 0.95))
        px, py, pz = prev_xyz
        nx, ny, nz = new_xyz
        return (
            a * float(px) + (1.0 - a) * float(nx),
            a * float(py) + (1.0 - a) * float(ny),
            a * float(pz) + (1.0 - a) * float(nz),
        )

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
        profile_name: str = "",
    ) -> Tuple[np.ndarray, str, Dict[str, object]]:
        start = time.perf_counter()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        infer_frame_bgr = frame_bgr
        img_h, img_w = frame_bgr.shape[:2]
        base_img_size = int(np.clip(int(img_size), 128, 640))
        base_img_size = max(128, (base_img_size // 32) * 32)
        run_img_size, run_conf, run_mode = self._select_adaptive_infer_settings(base_img_size, float(conf_thres))
        profile_label = str(profile_name or "").strip() or "Custom"

        prompt_error = self._apply_prompt_classes_if_needed(class_prompt)
        prompt_active = bool(self.active_prompt_classes)
        if prompt_active and not prompt_error:
            run_img_size = max(run_img_size, max(320, (PROMPT_IMG_SIZE // 32) * 32))
            run_img_size = min(768, run_img_size)
            # Keep prompt mode strict enough to avoid false positives from open-vocabulary ambiguity.
            run_conf = max(run_conf, float(np.clip(PROMPT_CONF_CAP, 0.12, 0.55)))
            run_mode = f"{run_mode}+prompt"

        depth_map = None
        depth_error = None
        if depth_enabled:
            try:
                depth_map, depth_error = self._get_depth_map(frame_rgb, depth_enabled, depth_interval)
            except Exception as exc:
                depth_error = str(exc)
                depth_map = None

        if depth_map is not None and depth_alpha > 0:
            try:
                depth_arr = np.asarray(depth_map, dtype=np.float32)
                if depth_arr.ndim == 3:
                    depth_arr = depth_arr[..., 0]
                if depth_arr.ndim != 2:
                    raise ValueError(f"depth_ndim={depth_arr.ndim}")
                if depth_arr.shape[:2] != frame_bgr.shape[:2]:
                    depth_arr = cv2.resize(
                        depth_arr,
                        (img_w, img_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                depth_uint8 = (np.clip(depth_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
                if depth_color.shape[:2] != frame_bgr.shape[:2]:
                    depth_color = cv2.resize(
                        depth_color,
                        (img_w, img_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                if depth_color.dtype != frame_bgr.dtype:
                    depth_color = depth_color.astype(frame_bgr.dtype, copy=False)
                alpha = float(np.clip(depth_alpha, 0.0, 0.8))
                frame_bgr = cv2.addWeighted(frame_bgr, 1.0 - alpha, depth_color, alpha, 0.0)
            except Exception as exc:
                overlay_err = f"depth_overlay_err={exc}"
                depth_error = f"{depth_error}; {overlay_err}" if depth_error else overlay_err

        with torch.inference_mode():
            assert self.model is not None
            infer_half = bool(self.use_half and not self.prompt_fp32_active)
            results = self.model.predict(
                source=infer_frame_bgr,
                imgsz=run_img_size,
                conf=run_conf,
                iou=float(iou_thres),
                max_det=int(max_det),
                device=self.device,
                half=infer_half,
                verbose=False,
            )
        pred = results[0] if results else None
        boxes = pred.boxes if pred is not None else None

        if prompt_active and not prompt_error and (boxes is None or len(boxes) == 0):
            retry_conf = float(np.clip(PROMPT_CONF_RETRY, 0.10, max(0.30, run_conf)))
            retry_img = max(run_img_size, max(320, (PROMPT_IMG_SIZE // 32) * 32))
            with torch.inference_mode():
                results_retry = self.model.predict(
                    source=infer_frame_bgr,
                    imgsz=retry_img,
                    conf=retry_conf,
                    iou=float(iou_thres),
                    max_det=int(max_det),
                    device=self.device,
                    half=infer_half,
                    verbose=False,
                )
            if results_retry:
                pred_retry = results_retry[0]
                boxes_retry = pred_retry.boxes
                if boxes_retry is not None and len(boxes_retry) > 0:
                    results = results_retry
                    pred = pred_retry
                    boxes = boxes_retry
                    run_mode = f"{run_mode}+retry"
                    run_conf = retry_conf
                    run_img_size = retry_img

        # Accuracy rescue pass: if nothing is detected, re-run at larger size and lower conf.
        # This improves hard cases without slowing normal frames that already have detections.
        if ACCURACY_RETRY_ENABLED and (boxes is None or len(boxes) == 0):
            retry_conf = float(np.clip(ACCURACY_RETRY_CONF, 0.12, max(0.24, run_conf)))
            retry_img = int(np.clip(max(run_img_size, ACCURACY_RETRY_IMG), 320, 960))
            retry_img = max(320, (retry_img // 32) * 32)
            with torch.inference_mode():
                results_retry = self.model.predict(
                    source=infer_frame_bgr,
                    imgsz=retry_img,
                    conf=retry_conf,
                    iou=float(iou_thres),
                    max_det=max(int(max_det), 32),
                    device=self.device,
                    half=infer_half,
                    verbose=False,
                )
            if results_retry:
                pred_retry = results_retry[0]
                boxes_retry = pred_retry.boxes
                if boxes_retry is not None and len(boxes_retry) > 0:
                    results = results_retry
                    pred = pred_retry
                    boxes = boxes_retry
                    run_mode = f"{run_mode}+accretry"
                    run_conf = retry_conf
                    run_img_size = retry_img

        hand_state = self._detect_hand(frame_rgb)
        hand_scene: Dict[str, Any] = {
            "visible": bool(hand_state.get("visible", False)),
            "source": str(hand_state.get("source", "none")),
            "confidence": float(hand_state.get("score", 0.0)),
            "handedness": str(hand_state.get("handedness", "")),
            "bbox_xyxy": hand_state.get("bbox_xyxy", None),
            "center_xy": hand_state.get("center_xy", None),
            "thumb_tip_xy": hand_state.get("thumb_tip_xy", None),
            "index_tip_xy": hand_state.get("index_tip_xy", None),
            "pinch_ratio": hand_state.get("pinch_ratio", None),
            "xyz_m": None,
            "distance_m": None,
        }
        dangerous_objects: List[Dict[str, Any]] = []

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
                bw = float(x2 - x1)
                bh = float(y2 - y1)
                area_ratio = (bw * bh) / float(max(1, img_w * img_h))
                if area_ratio < 0.00035 and conf < max(0.44, run_conf + 0.06):
                    continue
                if area_ratio > 0.88 and conf < 0.60:
                    continue
                aspect = bw / max(bh, 1.0)
                if (aspect > 7.0 or aspect < 0.14) and conf < 0.58:
                    continue

                scaled_x1 = (x1 / img_w) * self.ORIG_WIDTH
                scaled_x2 = (x2 / img_w) * self.ORIG_WIDTH
                scaled_y1 = (y1 / img_h) * self.ORIG_HEIGHT
                scaled_y2 = (y2 / img_h) * self.ORIG_HEIGHT

                features.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])
                parsed.append((x1, y1, x2, y2, conf, cls_id, name))

            if parsed and prompt_active and self.active_prompt_classes:
                prompt_classes = [str(x) for x in self.active_prompt_classes if str(x).strip()]
                keep_idx: List[int] = []
                for i, item in enumerate(parsed):
                    nm = str(item[6])
                    if any(_target_matches_name(cls_name, nm) for cls_name in prompt_classes):
                        keep_idx.append(i)
                if keep_idx:
                    features = [features[i] for i in keep_idx]
                    parsed = [parsed[i] for i in keep_idx]
                    run_mode = f"{run_mode}+focus"
                elif prompt_classes:
                    focus_conf = float(np.clip(max(0.24, 0.90 * run_conf), 0.16, 0.42))
                    focus_img = max(run_img_size, 640 if self.use_cuda else 416)
                    focus_img = int(np.clip(focus_img, 320, 960))
                    focus_img = max(320, (focus_img // 32) * 32)
                    with torch.inference_mode():
                        focus_results = self.model.predict(
                            source=infer_frame_bgr,
                            imgsz=focus_img,
                            conf=focus_conf,
                            iou=float(iou_thres),
                            max_det=max(int(max_det), 32),
                            device=self.device,
                            half=infer_half,
                            verbose=False,
                        )
                    focus_pred = focus_results[0] if focus_results else None
                    focus_boxes = focus_pred.boxes if focus_pred is not None else None
                    if focus_boxes is not None and len(focus_boxes) > 0:
                        xyxy_np = focus_boxes.xyxy.detach().cpu().numpy()
                        conf_np = focus_boxes.conf.detach().cpu().numpy()
                        cls_np = focus_boxes.cls.detach().cpu().numpy().astype(np.int32)
                        names_obj = getattr(focus_pred, "names", None)
                        names_map = self._names_to_dict(names_obj)
                        if names_map:
                            self.names = names_map
                        f2 = []
                        p2 = []
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
                            if not any(_target_matches_name(cls_name, name) for cls_name in prompt_classes):
                                continue
                            scaled_x1 = (x1 / img_w) * self.ORIG_WIDTH
                            scaled_x2 = (x2 / img_w) * self.ORIG_WIDTH
                            scaled_y1 = (y1 / img_h) * self.ORIG_HEIGHT
                            scaled_y2 = (y2 / img_h) * self.ORIG_HEIGHT
                            f2.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])
                            p2.append((x1, y1, x2, y2, conf, cls_id, name))
                        if p2:
                            features = f2
                            parsed = p2
                            run_mode = f"{run_mode}+focusretry"
                            run_conf = focus_conf
                            run_img_size = focus_img

            if parsed:
                prompt_classes = [str(x) for x in (self.active_prompt_classes or []) if str(x).strip()]
                temporal_features: List[List[float]] = []
                temporal_parsed: List[Tuple[int, int, int, int, float, int, str]] = []
                next_track_state: Dict[str, Dict[str, float]] = {}
                for i, item in enumerate(parsed):
                    x1, y1, x2, y2, conf, cls_id, name = item
                    key = f"{name}:{(x1 + x2) // 96}:{(y1 + y2) // 96}"
                    prev = self.obj_track_state.get(key, {})
                    prev_streak = int(prev.get("streak", 0.0) or 0)
                    prev_ema = float(prev.get("conf_ema", conf) or conf)
                    streak = min(12, prev_streak + 1)
                    conf_ema = 0.62 * prev_ema + 0.38 * float(conf)
                    next_track_state[key] = {
                        "streak": float(streak),
                        "conf_ema": float(conf_ema),
                        "last_seen": float(time.time()),
                    }

                    is_prompt_match = bool(prompt_classes) and any(_target_matches_name(cls_name, str(name)) for cls_name in prompt_classes)
                    keep = False
                    if prompt_active and prompt_classes:
                        strict_conf = max(float(run_conf), 0.30)
                        stable_conf = max(0.24, 0.86 * float(run_conf))
                        keep = (float(conf) >= strict_conf) or (streak >= 2 and conf_ema >= stable_conf)
                        if is_prompt_match and (float(conf) >= max(0.24, 0.82 * float(run_conf))):
                            keep = True
                    else:
                        strict_conf = max(float(run_conf) + 0.08, 0.50)
                        stable_conf = max(0.34, float(run_conf))
                        keep = (float(conf) >= strict_conf) or (streak >= 2 and conf_ema >= stable_conf)

                    if keep:
                        temporal_features.append(features[i])
                        temporal_parsed.append(item)

                self.obj_track_state = next_track_state
                if temporal_parsed:
                    features = temporal_features
                    parsed = temporal_parsed
                    run_mode = f"{run_mode}+stable"
                else:
                    parsed = []
                    features = []

            if not parsed:
                self.distance_cache = {}
                self.xyz_cache = {}
                self.xyz_filter_state = {}
                self.obj_track_state = {}
                self.miss_streak += 1
                object_count = 0
                if hand_scene.get("visible"):
                    bx = hand_scene.get("bbox_xyxy")
                    if isinstance(bx, (list, tuple)) and len(bx) >= 4:
                        x1h, y1h, x2h, y2h = [int(v) for v in bx[:4]]
                        self._draw_box(frame_bgr, x1h, y1h, x2h, y2h, "hand", (80, 220, 255))
                latency_ms = (time.perf_counter() - start) * 1000.0
                fps = 1000.0 / max(latency_ms, 1e-6)
                stats = (
                    f"detector={self.detector_label} | objects=0 | latency={latency_ms:.1f}ms | fps~{fps:.1f}"
                    f" | profile={profile_label} | infer={run_mode} | imgsz={run_img_size} | conf={run_conf:.2f} | miss={self.miss_streak}"
                    f" | dev={'cuda' if self.use_cuda else 'cpu'} | fp16={1 if self.use_half else 0} | cpu_t={self.cpu_threads}"
                )
                stats += f" | hand_mode={self.hand_detect_mode}"
                if self.hand_tracker is not None:
                    stats += f" | hand_backend={getattr(self.hand_tracker, 'backend', 'none')}"
                if hand_scene.get("visible"):
                    stats += " | hand=1"
                result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result_rgb = downscale_frame(result_rgb, MAX_OUTPUT_EDGE)
                scene_state = {
                    "timestamp": time.time(),
                    "image_size": [int(img_w), int(img_h)],
                    "profile_fov_deg": CAM_FOV_DEG,
                    "objects": [],
                    "nearest": None,
                    "hand": hand_scene,
                    "dangerous_objects": [],
                    "depth_enabled": bool(depth_enabled),
                    "prompt_classes": list(self.active_prompt_classes) if self.active_prompt_classes else [],
                }
                if prompt_error:
                    stats += f" | prompt_err={prompt_error}"
                if self.default_vocab_active:
                    stats += " | vocab=coco80"
                if self.default_vocab_error:
                    stats += f" | {self.default_vocab_error}"
                if depth_enabled and depth_error:
                    stats += f" | depth_err={depth_error}"
                return result_rgb, stats, scene_state

            distance_values_raw = self._predict_distances(np.asarray(features, dtype=np.float32))
            distance_values = np.asarray(distance_values_raw, dtype=np.float32).copy()
            conf_vals_np = np.asarray([p[4] for p in parsed], dtype=np.float32)
            area_vals_np = np.asarray([max(1.0, float((p[2] - p[0]) * (p[3] - p[1]))) for p in parsed], dtype=np.float32)
            depth_vals = None
            fusion_w_vals = None
            depth_scale = None
            if depth_map is not None:
                depth_vals = np.asarray(
                    [self._sample_box_depth(depth_map, p[0], p[1], p[2], p[3]) for p in parsed],
                    dtype=np.float32,
                )
                if depth_vals.size and distance_values_raw.size:
                    scale_try = float(np.median(np.clip(distance_values_raw, 0.05, None) * np.clip(depth_vals, 1e-3, None)))
                    if np.isfinite(scale_try) and scale_try > 1e-6:
                        depth_scale = scale_try
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
                danger_type = _dangerous_object_type(name)
                scene_objects.append(
                    {
                        "name": name,
                        "danger_type": danger_type,
                        "is_dangerous": bool(danger_type),
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

            if hand_scene.get("visible"):
                hb = hand_scene.get("bbox_xyxy")
                hc = hand_scene.get("center_xy")
                hand_dist = None
                if isinstance(hb, (list, tuple)) and len(hb) >= 4 and depth_map is not None:
                    try:
                        hx1, hy1, hx2, hy2 = [int(v) for v in hb[:4]]
                        hand_depth_rel = self._sample_box_depth(depth_map, hx1, hy1, hx2, hy2)
                        if (depth_scale is not None) and np.isfinite(hand_depth_rel) and hand_depth_rel > 1e-3:
                            hand_dist = float(np.clip(depth_scale / hand_depth_rel, 0.08, 2.5))
                    except Exception:
                        hand_dist = None
                if hand_dist is None and nearest_info is not None:
                    hand_dist = float(np.clip(float(nearest_info["z"]), 0.12, 2.5))

                if isinstance(hc, (list, tuple)) and len(hc) >= 2 and hand_dist is not None:
                    hu = float(hc[0])
                    hv = float(hc[1])
                    hx3 = ((hu - cx) / fx) * hand_dist
                    hy3 = ((hv - cy) / fy) * hand_dist
                    hx3, hy3, hz3 = self._smooth_hand_xyz(self.hand_xyz_cache, (hx3, hy3, hand_dist))
                    self.hand_xyz_cache = (hx3, hy3, hz3)
                    hand_scene["xyz_m"] = [round(float(hx3), 3), round(float(hy3), 3), round(float(hz3), 3)]
                    hand_scene["distance_m"] = round(float(np.sqrt(hx3 * hx3 + hy3 * hy3 + hz3 * hz3)), 3)
                else:
                    self.hand_xyz_cache = None

                if isinstance(hb, (list, tuple)) and len(hb) >= 4:
                    hx1, hy1, hx2, hy2 = [int(v) for v in hb[:4]]
                    hand_lbl = "hand"
                    if hand_scene.get("distance_m") is not None:
                        hand_lbl = f"hand {float(hand_scene['distance_m']):.2f}m"
                    pinch_ratio = hand_scene.get("pinch_ratio", None)
                    if pinch_ratio is not None:
                        try:
                            hand_lbl += f" p{float(pinch_ratio):.2f}"
                        except Exception:
                            pass
                    self._draw_box(frame_bgr, hx1, hy1, hx2, hy2, hand_lbl, (80, 220, 255))
            else:
                self.hand_xyz_cache = None

            if hand_scene.get("visible"):
                hb = hand_scene.get("bbox_xyxy")
                hc = hand_scene.get("center_xy")
                hand_interaction_candidates: List[Tuple[float, Dict[str, Any]]] = []
                if isinstance(hb, (list, tuple)) and len(hb) >= 4:
                    for obj in scene_objects:
                        ob = obj.get("bbox_xyxy", None)
                        if not isinstance(ob, (list, tuple)) or len(ob) < 4:
                            continue
                        iou = _bbox_iou_xyxy(hb, ob)
                        overlap_small = _bbox_overlap_on_smaller(hb, ob)
                        hc_hit = False
                        oc_hit = False
                        if isinstance(hc, (list, tuple)) and len(hc) >= 2:
                            hc_hit = _point_in_bbox(float(hc[0]), float(hc[1]), ob)
                        oc = _bbox_center_xyxy(ob)
                        if oc is not None:
                            oc_hit = _point_in_bbox(float(oc[0]), float(oc[1]), hb)

                        depth_delta = None
                        if isinstance(hand_scene.get("xyz_m"), list) and len(hand_scene.get("xyz_m")) >= 3:
                            try:
                                hz = float(hand_scene["xyz_m"][2])
                                oz = float(obj.get("xyz_m", [0.0, 0.0, 0.0])[2])
                                depth_delta = abs(hz - oz)
                            except Exception:
                                depth_delta = None
                        depth_term = 0.0
                        if depth_delta is not None:
                            depth_term = float(np.clip(1.0 - depth_delta / 0.18, 0.0, 1.0))

                        score = (
                            0.44 * float(np.clip(overlap_small / 0.45, 0.0, 1.0))
                            + 0.24 * float(np.clip(iou / 0.28, 0.0, 1.0))
                            + 0.18 * (1.0 if (hc_hit or oc_hit) else 0.0)
                            + 0.14 * depth_term
                        )
                        info = {
                            "score": round(float(np.clip(score, 0.0, 1.0)), 3),
                            "iou": round(float(iou), 3),
                            "overlap_small": round(float(overlap_small), 3),
                            "hand_center_in_obj": bool(hc_hit),
                            "obj_center_in_hand": bool(oc_hit),
                            "depth_delta_m": (round(float(depth_delta), 3) if depth_delta is not None else None),
                        }
                        obj["hand_interaction"] = info
                        if float(info["score"]) >= 0.25:
                            hand_interaction_candidates.append((float(info["score"]), {"name": str(obj.get("name", "")), **info}))
                if hand_interaction_candidates:
                    hand_interaction_candidates.sort(key=lambda x: x[0], reverse=True)
                    hand_scene["interaction_top"] = hand_interaction_candidates[0][1]
                else:
                    hand_scene["interaction_top"] = None

            for obj in scene_objects:
                if not bool(obj.get("is_dangerous", False)):
                    continue
                ox, oy, oz = [float(v) for v in obj.get("xyz_m", [0.0, 0.0, 0.0])]
                d3 = float(obj.get("distance3d_m", obj.get("distance_m", 9.9)))
                level = "high" if d3 < 0.80 else ("medium" if d3 < 1.30 else "low")
                danger_item = {
                    "name": str(obj.get("name", "object")),
                    "danger_type": str(obj.get("danger_type", "")),
                    "level": level,
                    "distance_m": round(d3, 3),
                    "xyz_m": [round(ox, 3), round(oy, 3), round(oz, 3)],
                    "conf": float(obj.get("conf", 0.0)),
                }
                if isinstance(hand_scene.get("xyz_m"), list) and len(hand_scene.get("xyz_m")) >= 3:
                    hx, hy, hz = [float(v) for v in hand_scene.get("xyz_m", [0.0, 0.0, 0.0])]
                    dh = float(np.sqrt((ox - hx) ** 2 + (oy - hy) ** 2 + (oz - hz) ** 2))
                    danger_item["distance_to_hand_m"] = round(dh, 3)
                    if dh < 0.28:
                        danger_item["level"] = "high"
                dangerous_objects.append(danger_item)
            if dangerous_objects:
                dangerous_objects.sort(key=lambda x: (0 if x.get("level") == "high" else 1, float(x.get("distance_m", 9.9))))

            self.distance_cache = new_cache
            self.xyz_cache = new_xyz_cache
            self.xyz_filter_state = {k: self.xyz_filter_state[k] for k in new_xyz_cache.keys() if k in self.xyz_filter_state}
            object_count = len(parsed)
            self.miss_streak = 0
        else:
            self.distance_cache = {}
            self.xyz_cache = {}
            self.xyz_filter_state = {}
            self.obj_track_state = {}
            if hand_scene.get("visible"):
                hb = hand_scene.get("bbox_xyxy")
                if isinstance(hb, (list, tuple)) and len(hb) >= 4:
                    hx1, hy1, hx2, hy2 = [int(v) for v in hb[:4]]
                    self._draw_box(frame_bgr, hx1, hy1, hx2, hy2, "hand", (80, 220, 255))
            self.hand_xyz_cache = None
            object_count = 0
            self.miss_streak += 1

        latency_ms = (time.perf_counter() - start) * 1000.0
        fps = 1000.0 / max(latency_ms, 1e-6)
        stats = (
            f"detector={self.detector_label} | objects={object_count} | "
            f"latency={latency_ms:.1f}ms | fps~{fps:.1f}"
        )
        stats += f" | profile={profile_label} | infer={run_mode} | imgsz={run_img_size} | conf={run_conf:.2f} | miss={self.miss_streak}"
        if self.obj_track_state:
            stats += f" | tracks={len(self.obj_track_state)}"
        if self.active_prompt_classes:
            stats += f" | prompt_cls={len(self.active_prompt_classes)}"
        if prompt_error:
            stats += f" | prompt_err={prompt_error}"
        if self.default_vocab_active:
            stats += " | vocab=coco80"
        if self.default_vocab_error:
            stats += f" | {self.default_vocab_error}"
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
        if hand_scene.get("visible"):
            stats += " | hand=1"
            stats += f" | hand_src={str(hand_scene.get('source', 'unknown'))}"
            if isinstance(hand_scene.get("xyz_m"), list) and len(hand_scene.get("xyz_m")) >= 3:
                hx, hy, hz = [float(v) for v in hand_scene.get("xyz_m", [0.0, 0.0, 0.0])]
                stats += f" | hand_xyz=({hx:+.2f},{hy:+.2f},{hz:.2f})"
            pinch_ratio = hand_scene.get("pinch_ratio", None)
            if pinch_ratio is not None:
                try:
                    stats += f" | pinch={float(pinch_ratio):.2f}"
                except Exception:
                    pass
            itop = hand_scene.get("interaction_top", None)
            if isinstance(itop, dict):
                iname = str(itop.get("name", ""))
                iscore = itop.get("score", None)
                if iname:
                    stats += f" | hand_obj={iname}"
                if iscore is not None:
                    try:
                        stats += f"({float(iscore):.2f})"
                    except Exception:
                        pass
        else:
            stats += " | hand=0"
        if self.hand_last_error:
            stats += f" | hand_err={self.hand_last_error}"
        if dangerous_objects:
            high_cnt = sum(1 for x in dangerous_objects if str(x.get("level", "")) == "high")
            stats += f" | danger={len(dangerous_objects)}"
            if high_cnt > 0:
                stats += f"({high_cnt} high)"
        if depth_enabled:
            stats += f" | depth_every={max(1, int(depth_interval))}"
            if self.depth_last_latency_ms > 0:
                stats += f" | depth={self.depth_last_latency_ms:.1f}ms"
            if fusion_w_mean is not None:
                stats += f" | w_depth={fusion_w_mean:.2f}"
            if depth_error:
                stats += f" | depth_err={depth_error}"
            with self.depth_lock:
                depth_busy = self.depth_job_busy
                depth_age = time.time() - self.depth_last_update_ts if self.depth_last_update_ts > 0 else -1.0
            stats += f" | depth_async={'1' if depth_busy else '0'}"
            if depth_age >= 0:
                stats += f" | depth_age={depth_age:.1f}s"
        stats += f" | dev={'cuda' if self.use_cuda else 'cpu'} | fp16={1 if infer_half else 0} | cpu_t={self.cpu_threads}"
        stats += f" | hand_mode={self.hand_detect_mode}"
        if self.hand_tracker is not None:
            stats += f" | hand_backend={getattr(self.hand_tracker, 'backend', 'none')}"
        stats += f" | build={APP_BUILD}"
        if depth_enabled:
            stats += f" | depth_model={self.depth_model_name}"
        result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result_rgb = downscale_frame(result_rgb, MAX_OUTPUT_EDGE)
        scene_state = {
            "timestamp": time.time(),
            "image_size": [int(img_w), int(img_h)],
            "profile_fov_deg": CAM_FOV_DEG,
            "objects": scene_objects,
            "nearest": nearest_info,
            "hand": hand_scene,
            "dangerous_objects": dangerous_objects,
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
WORKER = None
WORKER_LOCK = threading.Lock()
GUIDANCE_WORKER = None
GUIDANCE_WORKER_LOCK = threading.Lock()


def get_engine() -> RealtimeEngine:
    global ENGINE
    if ENGINE is not None:
        return ENGINE
    ENGINE = RealtimeEngine()
    return ENGINE


def get_worker():
    global WORKER
    if WORKER is not None:
        return WORKER
    with WORKER_LOCK:
        if WORKER is None:
            WORKER = AsyncInferenceWorker()
    return WORKER


def get_guidance_worker():
    global GUIDANCE_WORKER
    if GUIDANCE_WORKER is not None:
        return GUIDANCE_WORKER
    with GUIDANCE_WORKER_LOCK:
        if GUIDANCE_WORKER is None:
            GUIDANCE_WORKER = RealtimeGuidanceWorker()
    return GUIDANCE_WORKER


class AsyncInferenceWorker:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending_frame = None
        self._pending_params = None
        self._pending_seq = 0
        self._done_seq = 0
        self._latest_frame = None
        self._latest_stats = "Loading model..."
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
        prompt_enabled,
        class_prompt,
        profile_name,
    ):
        frame = normalize_frame(frame)
        if frame is None:
            return None, "Cannot decode webcam frame."
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
            str(class_prompt or "") if bool(prompt_enabled) else "",
            str(profile_name or ""),
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

    def latest_output(self):
        with self._lock:
            return self._latest_frame, self._latest_stats

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
                        self._latest_stats = f"Model initialization error: {exc}"
                    time.sleep(0.2)
                    continue

            try:
                with ENGINE_LOCK:
                    out_frame, stats, scene_state = engine.infer(frame, *params)
            except Exception as exc:
                out_frame = normalize_frame(frame)
                stats = f"Inference error: {exc}"
                scene_state = None

            with self._lock:
                if seq >= self._done_seq:
                    self._done_seq = seq
                    self._latest_frame = out_frame
                    self._latest_stats = stats
                    self._latest_scene = scene_state


class RealtimeGuidanceWorker:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending_scene = None
        self._pending_config = None
        self._pending_seq = 0
        self._done_seq = 0
        self._last_run_ts = 0.0
        self._latest_md = "Realtime guidance idle."
        self._latest_payload = json.dumps(
            {"enabled": False, "token": "", "text": "", "lang": "en-US", "rate": 1.0, "pitch": 1.0},
            ensure_ascii=False,
        )
        self._latest_hint = ""
        self._last_spoken_text = ""
        self._last_speech_token = ""
        self._last_voice_emit_ts = 0.0
        self._last_voice_norm = ""
        self._planner_cache: Dict[str, GeminiMultiAgentPlanner] = {}
        self._last_scene_sig = ""
        self._last_gemini_scene_sig = ""
        self._last_gemini_call_ts = 0.0
        self._target_lock_name = ""
        self._target_lock_until_ts = 0.0
        self._gemini_backoff_until_ts = 0.0
        self._gemini_fail_streak = 0
        self._gemini_last_error = ""
        self._minute_calls = deque()  # timestamps
        self._hour_calls = deque()  # timestamps
        self._task_complete_until_ts = 0.0
        self._hint_history = deque(maxlen=max(3, int(GUIDE_SAY_HISTORY)))
        self._norm_history = deque(maxlen=max(3, int(GUIDE_SAY_HISTORY)))
        self._scene_memory_layout: Dict[str, Dict[str, float]] = {}
        self._scene_memory_last_note = ""
        self._scene_memory_last_ts = 0.0
        self._timeline_events = deque(maxlen=24)
        self._latest_timeline_html = _render_guidance_timeline_html([])
        self._touch_streak = 0
        self._grasp_streak = 0
        self._grasp_target = ""
        self._grasp_hold_until_ts = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _speech_payload(
        text: str,
        enabled: bool,
        lang: str,
        rate: float,
        pitch: float,
        token: str,
    ) -> str:
        payload = {
            "enabled": bool(enabled),
            "token": str(token or ""),
            "text": _sanitize_guide_text(text),
            "lang": str(lang or "en-US"),
            "rate": float(np.clip(rate, 0.6, 1.6)),
            "pitch": float(np.clip(pitch, 0.6, 1.6)),
        }
        return json.dumps(payload, ensure_ascii=False)

    def submit(
        self,
        scene_state: Optional[Dict[str, Any]],
        user_query: str,
        profile_name: str,
        api_key_input: str,
        enabled: bool,
        interval_sec: float,
        voice_enabled: bool,
        voice_lang: str,
        voice_rate: float,
        voice_pitch: float,
        depth_informed: bool,
    ) -> Tuple[str, str, str]:
        interval_sec = float(np.clip(float(interval_sec), GUIDE_MIN_INTERVAL_SEC, GUIDE_MAX_INTERVAL_SEC))
        query = (user_query or "").strip() or "Help me safely reach the target object."
        profile = (profile_name or "").strip() or "Realtime"
        api_key = (api_key_input or "").strip()
        cfg = {
            "query": query,
            "profile": profile,
            "api_key": api_key,
            "enabled": bool(enabled),
            "interval_sec": interval_sec,
            "voice_enabled": bool(voice_enabled),
            "voice_lang": str(voice_lang or "en-US"),
            "voice_rate": float(voice_rate),
            "voice_pitch": float(voice_pitch),
            "depth_informed": bool(depth_informed),
        }
        with self._lock:
            if not enabled:
                self._pending_scene = None
                self._pending_config = None
                self._done_seq = self._pending_seq
                self._task_complete_until_ts = 0.0
                self._latest_md = "Realtime guidance paused."
                self._latest_payload = self._speech_payload(
                    text="",
                    enabled=False,
                    lang=cfg["voice_lang"],
                    rate=cfg["voice_rate"],
                    pitch=cfg["voice_pitch"],
                    token=self._last_speech_token,
                )
                return self._latest_md, self._latest_payload, self._latest_timeline_html
            if scene_state is None:
                self._latest_md = "Realtime guidance waiting for camera scene..."
                self._latest_payload = self._speech_payload(
                    text="",
                    enabled=False,
                    lang=cfg["voice_lang"],
                    rate=cfg["voice_rate"],
                    pitch=cfg["voice_pitch"],
                    token=self._last_speech_token,
                )
                return self._latest_md, self._latest_payload, self._latest_timeline_html
            self._pending_scene = scene_state
            self._pending_config = cfg
            self._pending_seq += 1
            return self._latest_md, self._latest_payload, self._latest_timeline_html

    def latest_output(self) -> Tuple[str, str, str]:
        with self._lock:
            return self._latest_md, self._latest_payload, self._latest_timeline_html

    def _get_planner(self, api_key: str) -> GeminiMultiAgentPlanner:
        key = api_key if api_key else "__env__"
        planner = self._planner_cache.get(key)
        if planner is not None:
            return planner
        planner = GeminiMultiAgentPlanner(api_key=api_key or None)
        self._planner_cache[key] = planner
        return planner

    @staticmethod
    def _scene_signature(scene_state: Dict[str, Any]) -> str:
        def q(value: Any, step: float) -> float:
            try:
                v = float(value)
            except Exception:
                v = 0.0
            if step <= 1e-6:
                return round(v, 3)
            return round(round(v / step) * step, 3)

        if not isinstance(scene_state, dict):
            return "none"
        nearest = scene_state.get("nearest") or {}
        if not isinstance(nearest, dict):
            nearest = {}
        hand = scene_state.get("hand") or {}
        if not isinstance(hand, dict):
            hand = {}
        dangers = scene_state.get("dangerous_objects") or []
        if not isinstance(dangers, list):
            dangers = []
        objs = scene_state.get("objects") or []
        obj_names = []
        fusion_vals: List[float] = []
        if isinstance(objs, list):
            for obj in objs[:4]:
                if isinstance(obj, dict):
                    obj_names.append(str(obj.get("name", "")))
                    try:
                        fw = obj.get("fusion_w", None)
                        if fw is not None:
                            fusion_vals.append(float(fw))
                    except Exception:
                        pass
        fusion_mean = round(float(np.mean(fusion_vals)), 2) if fusion_vals else 0.0
        hand_xyz = hand.get("xyz_m", None)
        if not isinstance(hand_xyz, (list, tuple)) or len(hand_xyz) < 3:
            hand_xyz = [0.0, 0.0, 0.0]
        top_danger = dangers[0] if dangers and isinstance(dangers[0], dict) else {}
        if not isinstance(top_danger, dict):
            top_danger = {}
        sig = {
            "n": str(nearest.get("name", "")),
            "x": q(nearest.get("x", 0.0), 0.3),
            "y": q(nearest.get("y", 0.0), 0.3),
            "z": q(nearest.get("z", 0.0), 0.5),
            "d": q(nearest.get("d", 0.0), 0.5),
            "c": len(objs) if isinstance(objs, list) else 0,
            "o": obj_names,
            "de": bool(scene_state.get("depth_enabled", False)),
            "dw": q(fusion_mean, 0.1),
            "hv": bool(hand.get("visible", False)),
            "hx": q(hand_xyz[0], 0.10),
            "hy": q(hand_xyz[1], 0.10),
            "hz": q(hand_xyz[2], 0.12),
            "dc": len(dangers),
            "dn": str(top_danger.get("name", "")),
            "dl": str(top_danger.get("level", "")),
        }
        return json.dumps(sig, ensure_ascii=True, sort_keys=True)

    def _update_scene_memory(self, scene_state: Dict[str, Any]) -> str:
        if not isinstance(scene_state, dict):
            return ""
        objects = scene_state.get("objects") or []
        if not isinstance(objects, list) or not objects:
            return ""
        now = time.time()
        current: Dict[str, Dict[str, float]] = {}
        for obj in objects[:12]:
            if not isinstance(obj, dict):
                continue
            name_raw = str(obj.get("name", "")).strip()
            if not name_raw:
                continue
            name = _canonical_target_name(name_raw) or _norm_label(name_raw) or name_raw.lower()
            x, y, z, d, conf = _obj_pose(obj)
            entry = {"x": float(x), "y": float(y), "z": float(z), "d": float(d), "conf": float(conf)}
            prev = current.get(name)
            if prev is None or entry["d"] < prev["d"]:
                current[name] = entry
        if not current:
            return ""

        note = ""
        if not self._scene_memory_layout:
            self._scene_memory_layout = {k: dict(v) for k, v in current.items()}
            self._scene_memory_last_note = "Scene baseline saved for memory."
            self._scene_memory_last_ts = now
            return self._scene_memory_last_note

        moved: List[Tuple[str, float]] = []
        added: List[str] = []
        removed: List[str] = []
        for name, cur in current.items():
            prev = self._scene_memory_layout.get(name)
            if prev is None:
                added.append(name)
            else:
                delta = float(
                    np.sqrt(
                        (cur["x"] - prev.get("x", 0.0)) ** 2
                        + (cur["y"] - prev.get("y", 0.0)) ** 2
                        + (cur["z"] - prev.get("z", 0.0)) ** 2
                    )
                )
                if delta >= 0.18:
                    moved.append((name, delta))
        for name in list(self._scene_memory_layout.keys())[:16]:
            if name not in current:
                removed.append(name)

        for name, cur in current.items():
            prev = self._scene_memory_layout.get(name)
            if prev is None:
                self._scene_memory_layout[name] = dict(cur)
            else:
                a = 0.72
                self._scene_memory_layout[name] = {
                    "x": a * prev.get("x", cur["x"]) + (1.0 - a) * cur["x"],
                    "y": a * prev.get("y", cur["y"]) + (1.0 - a) * cur["y"],
                    "z": a * prev.get("z", cur["z"]) + (1.0 - a) * cur["z"],
                    "d": a * prev.get("d", cur["d"]) + (1.0 - a) * cur["d"],
                    "conf": max(prev.get("conf", 0.0), cur["conf"]),
                }

        if moved:
            moved.sort(key=lambda x: x[1], reverse=True)
            note = f"Scene update: {moved[0][0]} moved."
        elif added:
            note = f"Scene update: new {added[0]} detected."
        elif removed:
            note = f"Scene update: {removed[0]} is no longer visible."

        if note:
            self._scene_memory_last_note = note
            self._scene_memory_last_ts = now
            return note
        if now - self._scene_memory_last_ts <= 6.0:
            return self._scene_memory_last_note
        return ""

    @staticmethod
    def _is_rate_limited_error(err_text: str) -> bool:
        e = str(err_text or "").lower()
        markers = ["429", "too many request", "too many requests", "quota", "rate limit", "resource exhausted"]
        return any(m in e for m in markers)

    def _trim_budget_windows(self, now: float) -> None:
        while self._minute_calls and (now - float(self._minute_calls[0])) > 60.0:
            self._minute_calls.popleft()
        while self._hour_calls and (now - float(self._hour_calls[0])) > 3600.0:
            self._hour_calls.popleft()

    def _check_budget(self, now: float) -> Tuple[bool, str]:
        self._trim_budget_windows(now)
        rpm_limit = max(1, int(GUIDE_GEMINI_RPM_LIMIT))
        hourly_limit = max(rpm_limit, int(GUIDE_GEMINI_HOURLY_LIMIT))
        used_min = len(self._minute_calls)
        used_hour = len(self._hour_calls)
        ok = used_min < rpm_limit and used_hour < hourly_limit
        return ok, f"{used_min}/{rpm_limit} rpm | {used_hour}/{hourly_limit} h"

    def _record_budget_call(self, now: float) -> None:
        self._trim_budget_windows(now)
        self._minute_calls.append(now)
        self._hour_calls.append(now)

    def _is_over_repeated(self, norm_text: str) -> bool:
        if not norm_text:
            return False
        cnt = sum(1 for x in self._norm_history if _text_similarity(norm_text, x) >= 0.93)
        return cnt >= 2

    @staticmethod
    def _diversify_phrase(phase: str, text: str) -> str:
        base = _sanitize_guide_text(text)
        phase_key = str(phase or "").strip().lower()
        if not base:
            return base
        openers = {
            "orientation": [
                "Reference update.",
                "Direction refresh.",
                "Spatial cue update.",
            ],
            "approach": [
                "Approach update.",
                "Small correction.",
                "Keep steady.",
            ],
            "fine-tuning": [
                "Fine control now.",
                "Micro adjustment.",
                "Precision step.",
            ],
            "grasp": [
                "Final grasp cue.",
                "Grip timing update.",
                "Close-range step.",
            ],
            "complete-touch": [
                "Contact confirmed.",
                "Touch reached.",
                "Reach complete.",
            ],
            "complete-grasp": [
                "Grasp confirmed.",
                "Object secured.",
                "Hold stable.",
            ],
        }
        choices = openers.get(phase_key, ["Update."])
        idx = int(time.time() * 10) % max(1, len(choices))
        prefix = choices[idx]
        if _normalize_guide_text(base).startswith(_normalize_guide_text(prefix)):
            return base
        return _sanitize_guide_text(f"{prefix} {base}")

    def _push_timeline(self, text: str, phase: str, model_used: str, now: float) -> None:
        msg = _sanitize_guide_text(text)
        if not msg:
            return
        norm = _normalize_guide_text(msg)
        if self._timeline_events:
            last = self._timeline_events[-1]
            last_norm = str(last.get("norm", ""))
            last_phase = str(last.get("phase", ""))
            last_ts = float(last.get("ts", 0.0) or 0.0)
            if (
                _text_similarity(norm, last_norm) >= 0.91
                and str(phase or "") == last_phase
                and (now - last_ts) < 2.2
            ):
                return
        self._timeline_events.append(
            {
                "ts": float(now),
                "phase": str(phase or "guidance"),
                "model": str(model_used or "local"),
                "text": msg,
                "norm": norm,
            }
        )
        events = [{k: v for k, v in item.items() if k != "norm"} for item in self._timeline_events]
        self._latest_timeline_html = _render_guidance_timeline_html(events)

    def _run(self):
        while True:
            with self._lock:
                has_new = self._pending_scene is not None and self._pending_seq > self._done_seq
                if has_new:
                    seq = self._pending_seq
                    scene_state = self._pending_scene
                    cfg = dict(self._pending_config or {})
                    last_run_ts = self._last_run_ts
                else:
                    seq = 0
                    scene_state = None
                    cfg = None
                    last_run_ts = self._last_run_ts

            if scene_state is None or cfg is None:
                time.sleep(0.05)
                continue

            now = time.time()
            interval_sec = float(cfg.get("interval_sec", 1.4))
            if now - last_run_ts < interval_sec:
                time.sleep(0.03)
                continue

            scene_sig = self._scene_signature(scene_state)
            scene_state_runtime = dict(scene_state) if isinstance(scene_state, dict) else scene_state
            if isinstance(scene_state_runtime, dict):
                mem_note = self._update_scene_memory(scene_state_runtime)
                if mem_note:
                    scene_state_runtime["scene_memory_note"] = mem_note
            gemini_interval_sec = max(float(interval_sec), float(GUIDE_GEMINI_MIN_INTERVAL_SEC))
            guidance = None
            model_used = "local-realtime"
            latency_ms = 0.0
            fallback_used = True
            err = ""
            prev_hint = " || ".join(list(self._hint_history)[-2:]) if self._hint_history else self._latest_hint
            env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
            has_api_key = bool(str(cfg.get("api_key", "")).strip() or str(env_key).strip())
            should_call_gemini = bool(has_api_key)
            budget_ok, budget_note = self._check_budget(now)

            if now < self._gemini_backoff_until_ts:
                remain = max(0.0, self._gemini_backoff_until_ts - now)
                should_call_gemini = False
                err = f"Gemini cooldown {remain:.1f}s after rate limit."
            elif not budget_ok:
                should_call_gemini = False
                err = "Gemini skipped: budget guard active."
            else:
                since_last = now - self._last_gemini_call_ts
                min_gap = gemini_interval_sec if GUIDE_GEMINI_ALWAYS_ON else max(8.0, 2.2 * gemini_interval_sec)
                if since_last < float(min_gap):
                    should_call_gemini = False
                    err = "Gemini skipped: waiting interval."

            if should_call_gemini:
                try:
                    planner = self._get_planner(str(cfg.get("api_key", "")))
                    t0 = time.perf_counter()
                    result = planner.guide_realtime(
                        user_query=str(cfg.get("query", "")),
                        scene_state=scene_state_runtime,
                        profile_name=str(cfg.get("profile", "Realtime")),
                        fov_deg=CAM_FOV_DEG,
                        previous_hint=prev_hint,
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    self._record_budget_call(now)
                    self._last_gemini_call_ts = now
                    self._last_gemini_scene_sig = scene_sig
                    if result.ok:
                        guidance = dict(result.output or {})
                        guidance["say"] = result.say
                        model_used = result.model
                        fallback_used = False
                        self._gemini_fail_streak = 0
                        self._gemini_backoff_until_ts = 0.0
                        self._gemini_last_error = ""
                    else:
                        err = str(result.error or "Gemini realtime guidance failed.")
                except Exception as exc:
                    err = str(exc)

                if guidance is None and err:
                    self._gemini_last_error = err
                    if self._is_rate_limited_error(err):
                        self._gemini_fail_streak += 1
                        backoff = min(
                            float(GUIDE_GEMINI_BACKOFF_MAX_SEC),
                            float(GUIDE_GEMINI_BACKOFF_BASE_SEC) * (2 ** max(0, self._gemini_fail_streak - 1)),
                        )
                        self._gemini_backoff_until_ts = now + backoff
                        err = f"{err} | backoff={backoff:.0f}s"

            if guidance is None:
                locked_target = self._target_lock_name if now < self._target_lock_until_ts else ""
                guidance = _build_local_realtime_guidance(
                    user_query=str(cfg.get("query", "")),
                    scene_state=scene_state_runtime,
                    previous_hint=prev_hint,
                    locked_target=locked_target,
                    depth_informed=bool(cfg.get("depth_informed", True)),
                    language=str(cfg.get("voice_lang", "en-US")),
                )
            scene_summary, scene_lines = _scene_detail_for_ui(scene_state_runtime)
            guidance["scene_summary"] = guidance.get("scene_summary") or scene_summary
            guidance["scene_lines"] = guidance.get("scene_lines") or scene_lines
            guidance["budget_note"] = budget_note
            phase_raw = str(guidance.get("phase", "")).strip().lower()
            current_target_raw = _canonical_target_name(str(guidance.get("target", "")))
            if not current_target_raw:
                current_target_raw = self._grasp_target
            raw_contact = bool(guidance.get("contact_detected", False))
            raw_grasp = bool(guidance.get("grasp_detected", False))
            if current_target_raw and current_target_raw != "unknown":
                if self._grasp_target and self._grasp_target != current_target_raw:
                    self._touch_streak = 0
                    self._grasp_streak = 0
                self._grasp_target = current_target_raw
            if raw_contact:
                self._touch_streak = min(8, self._touch_streak + 1)
            else:
                self._touch_streak = max(0, self._touch_streak - 1)
            if raw_grasp:
                self._grasp_streak = min(8, self._grasp_streak + 1)
            else:
                self._grasp_streak = max(0, self._grasp_streak - 1)

            stable_contact = bool(raw_contact or self._touch_streak >= 2)
            stable_grasp = bool(raw_grasp or self._grasp_streak >= 2)
            if stable_grasp and self._grasp_target:
                self._grasp_hold_until_ts = now + 1.8
            if (
                (not stable_grasp)
                and self._grasp_target
                and current_target_raw == self._grasp_target
                and (now < self._grasp_hold_until_ts)
                and stable_contact
            ):
                stable_grasp = True

            if phase_raw in {"search", "search-target"} and self._grasp_target and now < self._grasp_hold_until_ts:
                guidance["target"] = self._grasp_target
                stable_contact = True
                stable_grasp = True
                guidance["phase"] = "complete-grasp"
                guidance["say"] = _sanitize_guide_text(
                    f"{self._grasp_target} is likely secured in your hand. Keep your wrist steady and move slowly."
                )

            guidance["contact_detected"] = bool(stable_contact)
            guidance["grasp_detected"] = bool(stable_grasp)
            guidance["touch_streak"] = int(self._touch_streak)
            guidance["grasp_streak"] = int(self._grasp_streak)
            if stable_grasp:
                guidance["phase"] = "complete-grasp"
            elif stable_contact and phase_raw in {"fine-tuning", "grasp"}:
                guidance["phase"] = "complete-touch"

            say = _sanitize_guide_text(str(guidance.get("say", "")))
            phase = str(guidance.get("phase", ""))
            say_norm = _normalize_guide_text(say)
            if self._is_over_repeated(say_norm):
                say = self._diversify_phrase(phase, say)
                say_norm = _normalize_guide_text(say)
            guidance["say"] = say
            md = _render_realtime_guidance_markdown(guidance, model_used, latency_ms, fallback_used, err)

            with self._lock:
                prev_scene_sig_state = self._last_scene_sig
                self._last_run_ts = now
                self._last_scene_sig = scene_sig
                self._done_seq = max(self._done_seq, seq)
                self._latest_hint = say
                current_target = _canonical_target_name(str(guidance.get("target", "")))
                if current_target and current_target != "unknown":
                    if self._target_lock_name == current_target or now >= self._target_lock_until_ts or not self._target_lock_name:
                        self._target_lock_name = current_target
                        self._target_lock_until_ts = now + float(GUIDE_TARGET_LOCK_SEC)
                contact_detected = bool(guidance.get("contact_detected", False))
                grasp_detected = bool(guidance.get("grasp_detected", False))
                curr_norm = _normalize_guide_text(say)
                sim_prev = _text_similarity(curr_norm, self._last_voice_norm)
                elapsed_voice = now - self._last_voice_emit_ts
                major_scene_change = scene_sig != prev_scene_sig_state
                should_emit_voice = bool(cfg.get("voice_enabled", True))
                if contact_detected:
                    self._task_complete_until_ts = max(self._task_complete_until_ts, now + 12.0)
                if grasp_detected:
                    self._task_complete_until_ts = max(self._task_complete_until_ts, now + 18.0)
                if should_emit_voice:
                    if elapsed_voice < float(GUIDE_MIN_VOICE_EMIT_SEC) and sim_prev >= float(GUIDE_REPEAT_SIM_THRESHOLD):
                        should_emit_voice = False
                    if sim_prev >= 0.96 and phase == "approach-mid" and elapsed_voice < 5.0 and (not major_scene_change):
                        should_emit_voice = False
                    if phase == "complete-touch" and elapsed_voice < 10.0 and sim_prev >= 0.86:
                        should_emit_voice = False
                    if phase == "complete-touch" and now < self._task_complete_until_ts and elapsed_voice < 3.5:
                        should_emit_voice = False
                    if phase == "complete-grasp" and elapsed_voice < 12.0 and sim_prev >= 0.84:
                        should_emit_voice = False
                    if phase == "complete-grasp" and now < self._task_complete_until_ts and elapsed_voice < 4.5:
                        should_emit_voice = False
                if should_emit_voice and say:
                    self._last_spoken_text = say
                    self._last_speech_token = f"{int(now * 1000)}-{self._done_seq}"
                    self._last_voice_emit_ts = now
                    self._last_voice_norm = curr_norm
                if say:
                    self._hint_history.append(say)
                    self._norm_history.append(curr_norm)
                    self._push_timeline(say, str(guidance.get("phase", "")), model_used, now)
                self._latest_md = md
                self._latest_payload = self._speech_payload(
                    text=say,
                    enabled=bool(should_emit_voice),
                    lang=str(cfg.get("voice_lang", "en-US")),
                    rate=float(cfg.get("voice_rate", 1.0)),
                    pitch=float(cfg.get("voice_pitch", 1.0)),
                    token=self._last_speech_token,
                )


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
    prompt_enabled,
    class_prompt,
    auto_target_prompt,
    target_object,
    task_query,
    profile_name,
    gemini_api_key,
    rt_guidance_enabled,
    rt_guidance_interval,
    depth_informed_guidance,
    voice_enabled,
    voice_lang,
    voice_rate,
    voice_pitch,
):
    guidance_worker = get_guidance_worker()
    combined_query = _build_user_query(target_object, task_query)
    frame = normalize_frame(frame)
    if frame is None:
        worker = get_worker()
        latest_frame, latest_stats = worker.latest_output()
        latest_scene = worker.latest_scene()
        guide_md, speech_payload, guide_timeline = guidance_worker.submit(
            scene_state=latest_scene,
            user_query=combined_query,
            profile_name=profile_name,
            api_key_input=gemini_api_key,
            enabled=bool(rt_guidance_enabled),
            interval_sec=float(rt_guidance_interval),
            voice_enabled=bool(voice_enabled),
            voice_lang=str(voice_lang),
            voice_rate=float(voice_rate),
            voice_pitch=float(voice_pitch),
            depth_informed=bool(depth_informed_guidance),
        )
        if latest_frame is not None:
            latest_frame = normalize_frame(latest_frame)
            if latest_frame is not None:
                return frame_to_html(latest_frame), f"{latest_stats} | webcam_decode=retry", guide_md, speech_payload, guide_timeline
        placeholder = np.zeros((WEBCAM_CAPTURE_H, WEBCAM_CAPTURE_W, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Waiting for webcam frame...",
            (12, max(24, WEBCAM_CAPTURE_H // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        return frame_to_html(placeholder), "Waiting for webcam frame...", guide_md, speech_payload, guide_timeline

    worker = get_worker()
    seed_prompt = str(class_prompt or "").strip()
    target_seed = _cleanup_target_phrase(str(target_object or ""))
    target_seed = _canonical_target_name(target_seed) if target_seed else ""
    if target_seed:
        seed_prompt = f"{seed_prompt}, {target_seed}" if seed_prompt else target_seed
    effective_prompt = _merge_prompt_classes(str(seed_prompt or ""), str(combined_query or ""), bool(auto_target_prompt))
    prompt_focus_enabled = bool(prompt_enabled)
    if (not prompt_focus_enabled) and bool(auto_target_prompt):
        # Auto-focus detector on query targets to reduce clutter/false positives.
        prompt_focus_enabled = bool(str(effective_prompt or "").strip())
    out_frame, out_stats = worker.submit(
        frame,
        conf_thres,
        iou_thres,
        img_size,
        max_det,
        smooth_factor,
        depth_enabled,
        depth_alpha,
        depth_interval,
        prompt_focus_enabled,
        effective_prompt,
        profile_name,
    )
    scene_state = worker.latest_scene()
    guide_md, speech_payload, guide_timeline = guidance_worker.submit(
        scene_state=scene_state,
        user_query=combined_query,
        profile_name=profile_name,
        api_key_input=gemini_api_key,
        enabled=bool(rt_guidance_enabled),
        interval_sec=float(rt_guidance_interval),
        voice_enabled=bool(voice_enabled),
        voice_lang=str(voice_lang),
        voice_rate=float(voice_rate),
        voice_pitch=float(voice_pitch),
        depth_informed=bool(depth_informed_guidance),
    )
    if out_frame is not None:
        out_frame = normalize_frame(out_frame)
    if out_frame is None:
        out_frame = frame
    out_frame = downscale_frame(out_frame, MAX_OUTPUT_EDGE)
    return frame_to_html(out_frame), out_stats, guide_md, speech_payload, guide_timeline


def _build_profile_presets() -> Dict[str, Dict[str, float]]:
    force_cpu = os.getenv("FORCE_CPU", "0").strip().lower() in {"1", "true", "yes"}
    gpu_mode = (not force_cpu) and torch.cuda.is_available()
    if gpu_mode:
        return {
            "Fast": {
                "conf": 0.38,
                "iou": 0.50,
                "img_size": 416,
                "max_det": 16,
                "smooth": 0.34,
                "depth_enabled": True,
                "depth_alpha": 0.04,
                "depth_interval": 10,
            },
            "30fps-stable": {
                "conf": 0.34,
                "iou": 0.52,
                "img_size": 480,
                "max_det": 20,
                "smooth": 0.36,
                "depth_enabled": True,
                "depth_alpha": 0.05,
                "depth_interval": 8,
            },
            "Realtime": {
                "conf": 0.32,
                "iou": 0.54,
                "img_size": 544,
                "max_det": 24,
                "smooth": 0.36,
                "depth_enabled": True,
                "depth_alpha": 0.06,
                "depth_interval": 7,
            },
            "Balanced": {
                "conf": 0.30,
                "iou": 0.55,
                "img_size": 608,
                "max_det": 28,
                "smooth": 0.40,
                "depth_enabled": True,
                "depth_alpha": 0.10,
                "depth_interval": 6,
            },
            "High Accuracy": {
                "conf": 0.42,
                "iou": 0.58,
                "img_size": 640,
                "max_det": 36,
                "smooth": 0.44,
                "depth_enabled": True,
                "depth_alpha": 0.10,
                "depth_interval": 5,
            },
            "Ultra Accuracy": {
                "conf": 0.46,
                "iou": 0.62,
                "img_size": 704,
                "max_det": 48,
                "smooth": 0.48,
                "depth_enabled": True,
                "depth_alpha": 0.11,
                "depth_interval": 4,
            },
            "Precision": {
                "conf": 0.24,
                "iou": 0.60,
                "img_size": 704,
                "max_det": 48,
                "smooth": 0.50,
                "depth_enabled": True,
                "depth_alpha": 0.14,
                "depth_interval": 2,
            },
        }

    return {
        "Fast": {
            "conf": 0.56,
            "iou": 0.44,
            "img_size": 224,
            "max_det": 10,
            "smooth": 0.40,
            "depth_enabled": True,
            "depth_alpha": 0.06,
            "depth_interval": 11,
        },
        "30fps-stable": {
            "conf": 0.48,
            "iou": 0.45,
            "img_size": 256,
            "max_det": 12,
            "smooth": 0.44,
            "depth_enabled": True,
            "depth_alpha": 0.08,
            "depth_interval": 10,
        },
        "Realtime": {
            "conf": 0.50,
            "iou": 0.45,
            "img_size": 288,
            "max_det": 14,
            "smooth": 0.42,
            "depth_enabled": True,
            "depth_alpha": 0.09,
            "depth_interval": 9,
        },
        "Balanced": {
            "conf": 0.42,
            "iou": 0.50,
            "img_size": 320,
            "max_det": 18,
            "smooth": 0.52,
            "depth_enabled": True,
            "depth_alpha": 0.15,
            "depth_interval": 6,
        },
        "High Accuracy": {
            "conf": 0.52,
            "iou": 0.54,
            "img_size": 352,
            "max_det": 22,
            "smooth": 0.56,
            "depth_enabled": True,
            "depth_alpha": 0.16,
            "depth_interval": 5,
        },
        "Ultra Accuracy": {
            "conf": 0.56,
            "iou": 0.58,
            "img_size": 384,
            "max_det": 28,
            "smooth": 0.60,
            "depth_enabled": True,
            "depth_alpha": 0.18,
            "depth_interval": 4,
        },
        "Precision": {
            "conf": 0.34,
            "iou": 0.55,
            "img_size": 384,
            "max_det": 28,
            "smooth": 0.60,
            "depth_enabled": True,
            "depth_alpha": 0.18,
            "depth_interval": 4,
        },
    }


PROFILE_PRESETS = _build_profile_presets()


def apply_profile(profile_name: str):
    p = PROFILE_PRESETS.get(profile_name, PROFILE_PRESETS["Realtime"])
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


def _sanitize_guide_text(text: str) -> str:
    clean = " ".join(str(text or "").replace("\n", " ").split())
    if len(clean) > GUIDE_MAX_TEXT_CHARS:
        clean = clean[: GUIDE_MAX_TEXT_CHARS - 3].rstrip() + "..."
    return clean


def _normalize_guide_text(text: str) -> str:
    s = str(text or "").lower().strip()
    s = re.sub(r"\d+(\.\d+)?", "#", s)
    s = re.sub(r"[^a-z# ]+", " ", s)
    s = " ".join(s.split())
    return s


def _text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def _build_user_query(target_object: str, task_query: str) -> str:
    target = _cleanup_target_phrase(str(target_object or ""))
    target = _canonical_target_name(target) if target else ""
    detail = " ".join(str(task_query or "").split()).strip()
    if target:
        if detail:
            return f"I want to reach and grasp the {target}. {detail}"
        return f"I want to reach and grasp the {target} safely."
    return detail or "Help me safely reach the nearest target object."


def _render_guidance_timeline_html(events: List[Dict[str, Any]]) -> str:
    if not events:
        return (
            "<div class='lyrics-shell'>"
            "<div class='lyrics-empty'>Realtime guidance timeline will appear here.</div>"
            "</div>"
        )
    lines: List[str] = ["<div class='lyrics-shell'>"]
    ordered = list(events)[::-1]
    for idx, ev in enumerate(ordered):
        text = html.escape(str(ev.get("text", "")).strip())
        if not text:
            continue
        phase = html.escape(str(ev.get("phase", "")).strip() or "guidance")
        model = html.escape(str(ev.get("model", "")).strip() or "local")
        ts = float(ev.get("ts", 0.0) or 0.0)
        tm = time.localtime(ts if ts > 1 else time.time())
        tlabel = f"{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}"
        active = " lyric-active" if idx == 0 else ""
        lines.append(
            "<div class='lyric-row"
            f"{active}'>"
            f"<div class='lyric-meta'><span class='lyric-time'>{tlabel}</span>"
            f"<span class='lyric-phase'>{phase}</span>"
            f"<span class='lyric-model'>{model}</span></div>"
            f"<div class='lyric-text'>{text}</div>"
            "</div>"
        )
    lines.append("</div>")
    return "".join(lines)


def _scene_detail_for_ui(scene_state: Dict[str, Any]) -> Tuple[str, List[str]]:
    objects = (scene_state or {}).get("objects") or []
    if not isinstance(objects, list) or not objects:
        return "No object snapshot yet.", []

    def _obj_d(o: Dict[str, Any]) -> float:
        try:
            return float(o.get("distance3d_m", o.get("distance_m", 1e9)))
        except Exception:
            return 1e9

    ordered = sorted((o for o in objects if isinstance(o, dict)), key=_obj_d)
    top = ordered[:5]
    lines: List[str] = []
    for obj in top:
        name = str(obj.get("name", "obj"))
        d = float(obj.get("distance3d_m", obj.get("distance_m", -1.0)))
        xyz = obj.get("xyz_m", [0.0, 0.0, 0.0])
        if not isinstance(xyz, list) or len(xyz) < 3:
            xyz = [0.0, 0.0, d]
        x = float(xyz[0])
        y = float(xyz[1])
        z = float(xyz[2])
        hdir = "left" if x < -0.2 else ("right" if x > 0.2 else "center")
        lines.append(
            f"{name}: D={d:.2f}m XYZ=({x:+.2f},{y:+.2f},{z:.2f}) dir={hdir}"
        )

    nearest_name = str(top[0].get("name", "unknown")) if top else "unknown"
    nearest_d = _obj_d(top[0]) if top else -1.0
    summary = f"{len(ordered)} objects | nearest={nearest_name} {nearest_d:.2f}m"
    return summary, lines


def _obj_pose(obj: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    xyz = obj.get("xyz_m", [0.0, 0.0, 0.0]) if isinstance(obj, dict) else [0.0, 0.0, 0.0]
    if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
        xyz = [obj.get("x", 0.0), obj.get("y", 0.0), obj.get("z", 0.0)] if isinstance(obj, dict) else [0.0, 0.0, 0.0]
    x = float(xyz[0])
    y = float(xyz[1])
    z = float(xyz[2])
    d = float(obj.get("distance3d_m", obj.get("distance_m", obj.get("d", z)))) if isinstance(obj, dict) else float(z)
    conf = float(obj.get("conf", 0.0)) if isinstance(obj, dict) else 0.0
    return x, y, z, d, conf


def _obj_depth_values(obj: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(obj, dict):
        return None, None
    depth_rel = obj.get("depth_rel", None)
    fusion_w = obj.get("fusion_w", None)
    try:
        d_rel = float(depth_rel) if depth_rel is not None else None
    except Exception:
        d_rel = None
    try:
        w = float(fusion_w) if fusion_w is not None else None
    except Exception:
        w = None
    return d_rel, w


def _depth_reliability_score(conf: float, fusion_w: Optional[float], depth_rel: Optional[float]) -> float:
    if depth_rel is None:
        return 0.0
    score = 0.48 + 0.40 * float(np.clip(conf, 0.0, 1.0))
    if fusion_w is not None:
        # Fusion around ~0.3 usually means depth contributes but does not dominate.
        score += 0.22 * (1.0 - min(1.0, abs(float(fusion_w) - 0.30) / 0.30))
    return float(np.clip(score, 0.0, 1.0))


def _clock_direction_from_xy(x: float, y: float) -> str:
    # Camera plane: x right(+), y down(+), so 12 o'clock is -y.
    ang = (float(np.degrees(np.arctan2(y, x))) + 90.0) % 360.0
    idx = int(np.floor((ang + 15.0) / 30.0)) % 12
    hour = 12 if idx == 0 else idx
    return f"{hour} o'clock"


def _distance_phrase_human(d_m: float) -> str:
    d = float(max(0.0, d_m))
    if d >= 1.20:
        return "about one and a half arm lengths away"
    if d >= 0.75:
        return "about one arm length away"
    if d >= 0.45:
        return "about half an arm length away"
    if d >= 0.25:
        return "about one palm length away"
    if d >= 0.12:
        return "about two finger lengths away"
    if d >= 0.06:
        return "about one finger length away"
    return "very close"


def _cm_to_body_step(cm: float) -> str:
    v = float(max(0.0, cm))
    if v <= 1.2:
        return "a tiny bit"
    if v <= 2.5:
        return "about one fingertip"
    if v <= 5.0:
        return "about one finger width"
    if v <= 8.0:
        return "about two finger widths"
    if v <= 14.0:
        return "about one palm width"
    if v <= 24.0:
        return "about half a forearm"
    return "about one forearm"


def _bbox_iou_xyxy(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = max(1e-6, area_a + area_b - inter)
    return float(inter / union)


def _bbox_area_xyxy(bbox: Optional[List[float]]) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _bbox_center_xyxy(bbox: Optional[List[float]]) -> Optional[Tuple[float, float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return float(0.5 * (x1 + x2)), float(0.5 * (y1 + y2))


def _bbox_overlap_on_smaller(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = _bbox_area_xyxy(a)
    area_b = _bbox_area_xyxy(b)
    base = max(1e-6, min(area_a, area_b))
    return float(inter / base)


def _point_in_bbox(px: float, py: float, bbox: Optional[List[float]]) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return (x1 <= float(px) <= x2) and (y1 <= float(py) <= y2)


def _distance_point_to_segment_xy(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom < 1e-8:
        return float(np.sqrt((px - ax) ** 2 + (py - ay) ** 2))
    t = float(np.clip((apx * abx + apy * aby) / denom, 0.0, 1.0))
    qx = ax + t * abx
    qy = ay + t * aby
    return float(np.sqrt((px - qx) ** 2 + (py - qy) ** 2))


def _danger_blocks_reach_path(danger_xyz: Tuple[float, float, float], hand_xyz: Tuple[float, float, float], target_xyz: Tuple[float, float, float]) -> bool:
    dx, dy, dz = danger_xyz
    hx, hy, hz = hand_xyz
    tx, ty, tz = target_xyz
    lateral = _distance_point_to_segment_xy(dx, dy, hx, hy, tx, ty)
    if lateral > 0.13:
        return False
    z_min = min(hz, tz) - 0.08
    z_max = max(hz, tz) + 0.10
    return z_min <= dz <= z_max


def _pick_primary_danger(
    dangerous_objects: List[Dict[str, Any]],
    hand_xyz: Optional[Tuple[float, float, float]],
    target_xyz: Optional[Tuple[float, float, float]],
) -> Optional[Dict[str, Any]]:
    if not dangerous_objects:
        return None
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for obj in dangerous_objects:
        xyz = obj.get("xyz_m", [0.0, 0.0, 0.0])
        if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
            continue
        ox, oy, oz = [float(v) for v in xyz[:3]]
        d = float(obj.get("distance_m", 9.9))
        level = str(obj.get("level", "low")).lower()
        s = (2.0 if level == "high" else (1.2 if level == "medium" else 0.6)) + (1.0 / max(d, 0.2))
        if hand_xyz is not None and target_xyz is not None and _danger_blocks_reach_path((ox, oy, oz), hand_xyz, target_xyz):
            s += 2.0
        if "distance_to_hand_m" in obj:
            s += 1.0 / max(float(obj.get("distance_to_hand_m", 9.9)), 0.08)
        scored.append((s, obj))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _pick_depth_blocking_hazard(focus_obj: Dict[str, Any], objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    fx, fy, _, fd, _ = _obj_pose(focus_obj)
    fname = str(focus_obj.get("name", "")).strip().lower()
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name", "")).strip().lower()
        if not name or name == fname:
            continue
        x, y, _, d, conf = _obj_pose(obj)
        dx = abs(x - fx)
        dy = abs(y - fy)
        dd = d - fd
        if (
            dx <= float(GUIDE_DEPTH_HAZARD_X_M)
            and dy <= float(GUIDE_DEPTH_HAZARD_Y_M)
            and (-float(GUIDE_DEPTH_HAZARD_FRONT_M)) <= dd <= float(GUIDE_DEPTH_HAZARD_BEHIND_M)
            and conf >= 0.15
        ):
            front_bonus = 0.45 if dd < -0.03 else 0.0
            risk = (1.0 / max(0.05, dx + 0.04)) + (0.8 / max(0.08, abs(dd) + 0.08)) + front_bonus
            candidates.append((risk, obj))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _select_focus_object(
    scene_state: Dict[str, Any],
    requested_targets: List[str],
    locked_target: str = "",
) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    objects = (scene_state or {}).get("objects") or []
    objs = [o for o in objects if isinstance(o, dict)]
    if not objs:
        return None, (requested_targets[0] if requested_targets else "unknown"), False

    target_order: List[str] = []
    seen = set()
    for t in [locked_target] + list(requested_targets):
        c = _canonical_target_name(t)
        if c and c not in seen:
            seen.add(c)
            target_order.append(c)

    def _score(obj: Dict[str, Any]) -> float:
        x, y, _, d, conf = _obj_pose(obj)
        center_bonus = 1.0 - min(1.0, np.sqrt(x * x + y * y) / 0.8)
        dist_bonus = 1.0 / max(d + 0.25, 0.35)
        return 0.70 * float(conf) + 0.20 * center_bonus + 0.10 * dist_bonus

    if target_order:
        for tgt in target_order:
            matches = [o for o in objs if _target_matches_name(tgt, str(o.get("name", "")))]
            if matches:
                matches.sort(key=lambda o: (_score(o), -_obj_pose(o)[3]), reverse=True)
                return matches[0], tgt, True
        return None, target_order[0], False

    objs.sort(key=lambda o: (_score(o), -_obj_pose(o)[3]), reverse=True)
    best = objs[0]
    return best, str(best.get("name", "target")), True


def _pick_primary_hazard(focus_obj: Dict[str, Any], objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    fx, fy, _, fd, _ = _obj_pose(focus_obj)
    fname = str(focus_obj.get("name", "")).strip().lower()
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name", "")).strip().lower()
        if not name or name == fname:
            continue
        x, y, _, d, conf = _obj_pose(obj)
        lateral = abs(x - fx)
        vertical = abs(y - fy)
        if lateral <= 0.24 and vertical <= 0.20 and d <= fd + 0.35 and conf >= 0.18:
            risk = (1.0 / max(d, 0.25)) + (0.5 / max(lateral + 0.02, 0.05))
            candidates.append((risk, obj))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _build_local_realtime_guidance(
    user_query: str,
    scene_state: Dict[str, Any],
    previous_hint: str = "",
    locked_target: str = "",
    depth_informed: bool = True,
    language: str = "en-US",
) -> Dict[str, Any]:
    scene_summary, scene_lines = _scene_detail_for_ui(scene_state)
    objects = [o for o in ((scene_state or {}).get("objects") or []) if isinstance(o, dict)]
    hand_state = (scene_state or {}).get("hand") or {}
    dangerous_objects = (scene_state or {}).get("dangerous_objects") or []
    if not isinstance(dangerous_objects, list):
        dangerous_objects = []
    scene_memory_note = str((scene_state or {}).get("scene_memory_note", "")).strip()
    is_vi = str(language or "").lower().startswith("vi")

    def _txt(en_text: str, vi_text: str) -> str:
        return vi_text if is_vi else en_text

    hand_visible = bool(isinstance(hand_state, dict) and hand_state.get("visible", False))
    hand_xyz: Optional[Tuple[float, float, float]] = None
    hand_distance_m = None
    if isinstance(hand_state, dict):
        hxyz = hand_state.get("xyz_m", None)
        if isinstance(hxyz, list) and len(hxyz) >= 3:
            try:
                hand_xyz = (float(hxyz[0]), float(hxyz[1]), float(hxyz[2]))
                hand_distance_m = float(np.sqrt(hand_xyz[0] ** 2 + hand_xyz[1] ** 2 + hand_xyz[2] ** 2))
            except Exception:
                hand_xyz = None
                hand_distance_m = None

    def _base_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        out["guidance_style"] = str(out.get("guidance_style", "enact"))
        out["clock_direction"] = str(out.get("clock_direction", "12 o'clock"))
        out["distance_human"] = str(out.get("distance_human", "unknown"))
        out["hand_visible"] = bool(hand_visible)
        out["hand_xyz_m"] = [round(float(v), 3) for v in hand_xyz] if hand_xyz is not None else None
        out["hand_distance_m"] = round(float(hand_distance_m), 3) if hand_distance_m is not None else None
        out["hand_to_target_m"] = out.get("hand_to_target_m", None)
        out["contact_detected"] = bool(out.get("contact_detected", False))
        out["grasp_detected"] = bool(out.get("grasp_detected", False))
        out["target_xyz_m"] = out.get("target_xyz_m", None)
        out["danger_count"] = len(dangerous_objects)
        out["primary_danger"] = out.get("primary_danger", None)
        out["corridor_blocked"] = bool(out.get("corridor_blocked", False))
        out["scene_memory_note"] = scene_memory_note
        return out

    requested_targets = _extract_requested_targets(user_query, "")
    focus_obj, target_hint, target_visible = _select_focus_object(scene_state, requested_targets, locked_target=locked_target)
    if not objects or focus_obj is None:
        requested_txt = target_hint if target_hint and target_hint != "unknown" else "target"
        variants = [
            _txt(
                f"I cannot see {requested_txt} yet. Keep your hand still and scan the camera slowly from left to right.",
                f"Chua thay {requested_txt}. Giu tay yen va quet camera cham tu trai sang phai.",
            ),
            _txt(
                f"{requested_txt} is not visible now. Pause at center for one second, then continue a gentle side scan.",
                f"Van chua thay {requested_txt}. Dung 1 giay o giua roi quet nhe hai ben.",
            ),
            _txt(
                f"I am still searching for {requested_txt}. Keep your wrist steady and sweep horizontally at a slow pace.",
                f"He thong dang tim {requested_txt}. Giu co tay on dinh va lia ngang that cham.",
            ),
        ]
        if hand_visible:
            variants.append(
                _txt(
                    f"{requested_txt} is not visible yet. Keep your hand inside the camera view and scan slowly left to right.",
                    f"Chua thay {requested_txt}. Giu tay trong khung hinh va quet cham tu trai sang phai.",
                )
            )
        idx = int(time.time()) % len(variants)
        say = variants[idx]
        if previous_hint and _text_similarity(_normalize_guide_text(previous_hint), _normalize_guide_text(say)) > 0.92:
            say = variants[(idx + 1) % len(variants)]
        if scene_memory_note:
            say = _sanitize_guide_text(f"{say} {scene_memory_note}")
        return _base_fields({
            "say": say,
            "target": requested_txt,
            "distance_m": -1.0,
            "direction": "center",
            "clock_direction": "12 o'clock",
            "distance_human": "unknown",
            "confidence": 0.35 if requested_targets else 0.2,
            "safety_note": _txt("Stop if uncertain. Keep fingers open while searching.", "Neu khong chac, dung lai. Mo cac ngon tay trong luc tim."),
            "scene_summary": scene_summary,
            "scene_lines": scene_lines,
            "phase": "search",
            "depth_mode": "off",
            "depth_reliability": 0.0,
            "depth_note": "Depth guidance waiting for target visibility.",
            "intent_agent": {"task": user_query or "Reach target", "target_object": requested_txt, "confidence": 0.28},
            "spatial_agent": {"target_visible": False, "target_xyz_m": [0.0, 0.0, 0.0], "target_distance_m": -1.0, "recommended_approach": "Slow scan left-center-right."},
            "safety_agent": {"risk_level": "medium", "hazards": ["Target not visible."], "collision_objects": [], "safety_score": 0.5},
            "path_agent": {
                "phase": "search",
                "micro_steps": [
                    _txt("Center camera.", "Canh giua camera."),
                    _txt("Sweep slowly left to right.", "Quet cham trai sang phai."),
                    _txt("Stop and re-center when target appears.", "Dung lai va canh giua khi vat the xuat hien."),
                ],
                "adaptive_voice_cadence_sec": 2.2,
            },
        })

    if requested_targets and not target_visible:
        requested_txt = target_hint if target_hint else requested_targets[0]
        visible_names = [str(o.get("name", "")) for o in objects[:4] if str(o.get("name", "")).strip()]
        seen_note = f" I currently see: {', '.join(visible_names[:3])}." if visible_names else ""
        say = _txt(
            (
                f"I cannot confirm {requested_txt} yet.{seen_note} "
                f"Scan slowly and keep the camera centered on your workspace."
            ).strip(),
            (
                f"Toi chua xac nhan duoc {requested_txt}.{seen_note} "
                f"Hay quet cham va giu camera o giua ban."
            ).strip(),
        )
        return _base_fields({
            "say": _sanitize_guide_text(say),
            "target": requested_txt,
            "distance_m": -1.0,
            "direction": "center",
            "clock_direction": "12 o'clock",
            "distance_human": "unknown",
            "confidence": 0.45,
            "safety_note": _txt("Requested object is not visible. Avoid reaching forward until it appears.", "Vat the yeu cau chua xuat hien. Khong voi tay toi truoc khi thay ro."),
            "scene_summary": scene_summary,
            "scene_lines": scene_lines,
            "phase": "search-target",
            "depth_mode": "off",
            "depth_reliability": 0.0,
            "depth_note": "Depth-informed path is paused until requested target appears.",
            "intent_agent": {"task": user_query or "Reach target", "target_object": requested_txt, "confidence": 0.42},
            "spatial_agent": {"target_visible": False, "target_xyz_m": [0.0, 0.0, 0.0], "target_distance_m": -1.0, "recommended_approach": "Search while keeping hand still."},
            "safety_agent": {"risk_level": "medium", "hazards": ["Requested target missing."], "collision_objects": visible_names[:2], "safety_score": 0.52},
            "path_agent": {
                "phase": "search-target",
                "micro_steps": [
                    _txt("Keep hand still.", "Giu tay dung yen."),
                    _txt("Scan workspace left-center-right.", "Quet khong gian trai-giua-phai."),
                    _txt("Resume reach only when target is visible.", "Chi tiep tuc voi tay khi thay ro muc tieu."),
                ],
                "adaptive_voice_cadence_sec": 2.0,
            },
        })

    target = str(focus_obj.get("name") or target_hint or "target")
    x, y, z_t, d, conf = _obj_pose(focus_obj)
    d = max(0.05, float(d))
    z_t = max(0.05, float(z_t if np.isfinite(z_t) else d))
    target_xyz = (float(x), float(y), float(z_t))

    hand_to_target_m = None
    dxh = dyh = dzh = 0.0
    if hand_xyz is not None:
        dxh = float(target_xyz[0] - hand_xyz[0])
        dyh = float(target_xyz[1] - hand_xyz[1])
        dzh = float(target_xyz[2] - hand_xyz[2])
        hand_to_target_m = float(np.sqrt(dxh * dxh + dyh * dyh + dzh * dzh))
    clock_direction = _clock_direction_from_xy(x, y)
    distance_human = _distance_phrase_human(d)

    target_bbox = focus_obj.get("bbox_xyxy", None) if isinstance(focus_obj, dict) else None
    hand_bbox = hand_state.get("bbox_xyxy", None) if isinstance(hand_state, dict) else None
    hand_interaction = focus_obj.get("hand_interaction", {}) if isinstance(focus_obj, dict) else {}
    if not isinstance(hand_interaction, dict):
        hand_interaction = {}
    interaction_score = float(hand_interaction.get("score", 0.0) or 0.0)
    interaction_overlap = float(hand_interaction.get("overlap_small", 0.0) or 0.0)
    pinch_ratio = hand_state.get("pinch_ratio", None) if isinstance(hand_state, dict) else None
    try:
        pinch_ratio = float(pinch_ratio) if pinch_ratio is not None else None
    except Exception:
        pinch_ratio = None
    pinch_closed = bool(pinch_ratio is not None and pinch_ratio <= 0.30)
    hand_target_iou = _bbox_iou_xyxy(hand_bbox, target_bbox)
    hand_target_overlap = _bbox_overlap_on_smaller(hand_bbox, target_bbox)
    target_center = _bbox_center_xyxy(target_bbox)
    hand_center = _bbox_center_xyxy(hand_bbox)
    center_hit = False
    center_norm = 10.0
    if hand_center is not None and target_bbox is not None:
        center_hit = _point_in_bbox(hand_center[0], hand_center[1], target_bbox)
    if target_center is not None and hand_bbox is not None:
        center_hit = bool(center_hit or _point_in_bbox(target_center[0], target_center[1], hand_bbox))
    if hand_center is not None and target_center is not None:
        center_dist_px = float(np.sqrt((hand_center[0] - target_center[0]) ** 2 + (hand_center[1] - target_center[1]) ** 2))
        target_diag = float(np.sqrt(max(_bbox_area_xyxy(target_bbox), 1.0)))
        center_norm = center_dist_px / max(8.0, 0.55 * target_diag)

    hand_cover_target = 0.0
    if hand_bbox is not None and target_bbox is not None:
        ax1, ay1, ax2, ay2 = [float(v) for v in hand_bbox[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in target_bbox[:4]]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_t = max(1e-6, _bbox_area_xyxy(target_bbox))
        hand_cover_target = float(np.clip(inter / area_t, 0.0, 1.0))

    depth_near = bool(hand_to_target_m is not None and hand_to_target_m <= max(float(HAND_CONTACT_DIST_M) * 1.35, 0.075))
    reach_near = bool(hand_to_target_m is not None and hand_to_target_m <= max(float(HAND_TARGET_REACH_M) * 1.30, 0.16))
    contact_score = 0.0
    if depth_near:
        contact_score += 0.42
    if center_hit:
        contact_score += 0.27
    contact_score += 0.24 * float(np.clip(hand_target_overlap / 0.35, 0.0, 1.0))
    contact_score += 0.18 * float(np.clip(hand_target_iou / 0.28, 0.0, 1.0))
    contact_score += 0.32 * float(np.clip(interaction_score, 0.0, 1.0))
    contact_score += 0.14 * float(np.clip(interaction_overlap / 0.45, 0.0, 1.0))
    if center_norm < 0.65:
        contact_score += 0.13
    if hand_cover_target >= 0.34:
        contact_score += 0.12
    if pinch_closed:
        contact_score += 0.11
    if reach_near:
        contact_score += 0.08
    if hand_to_target_m is not None and hand_to_target_m > 0.22:
        contact_score -= 0.18
    contact_score = float(np.clip(contact_score, 0.0, 1.0))
    contact_detected = bool(hand_visible and contact_score >= 0.56)
    grasp_score = 0.0
    grasp_score += 0.46 * contact_score
    grasp_score += 0.25 * float(np.clip(hand_cover_target / 0.52, 0.0, 1.0))
    grasp_score += 0.16 * float(np.clip(interaction_score, 0.0, 1.0))
    if pinch_closed:
        grasp_score += 0.18
    if hand_to_target_m is not None and hand_to_target_m <= max(0.080, float(HAND_TARGET_REACH_M) * 0.95):
        grasp_score += 0.11
    if hand_to_target_m is not None and hand_to_target_m > 0.20:
        grasp_score -= 0.18
    grasp_score = float(np.clip(grasp_score, 0.0, 1.0))
    grasp_detected = bool(hand_visible and grasp_score >= 0.64)

    depth_enabled_flag = bool((scene_state or {}).get("depth_enabled", False))
    depth_rel, fusion_w = _obj_depth_values(focus_obj)
    depth_available = bool(depth_enabled_flag and depth_rel is not None)
    depth_rel_score = _depth_reliability_score(conf, fusion_w, depth_rel) if (depth_informed and depth_available) else 0.0
    depth_mode = "off"
    depth_note = "Depth guidance disabled."
    if depth_informed and depth_enabled_flag and depth_available:
        depth_mode = "midas-fused"
        if depth_rel_score >= 0.78:
            depth_note = f"Depth lock strong (rel={depth_rel:.2f}, w={float(fusion_w or 0.0):.2f})."
        elif depth_rel_score >= 0.58:
            depth_note = f"Depth lock moderate (rel={depth_rel:.2f}, w={float(fusion_w or 0.0):.2f})."
        else:
            depth_note = f"Depth lock weak; keep smaller motion (rel={depth_rel:.2f}, w={float(fusion_w or 0.0):.2f})."
    elif depth_informed and depth_enabled_flag:
        depth_mode = "warmup"
        depth_note = "Depth model warming up or unavailable for target box."

    horiz = "left" if x < -0.16 else ("right" if x > 0.16 else "center")
    vert = "up" if y < -0.12 else ("down" if y > 0.12 else "center")
    side_cm = int(round(abs(x) * 100.0))
    vertical_cm = int(round(abs(y) * 100.0))
    forward_cm = int(round(max(0.0, d - 0.28) * 100.0))
    if depth_informed and depth_available:
        if d > 0.95:
            forward_cm = int(round(max(0.0, d - 0.32) * 100.0))
        elif d < 0.55:
            forward_cm = int(round(max(0.0, d - 0.24) * 100.0))
    lateral_cmd = f"move {horiz}" if horiz != "center" else "keep center alignment"
    direction_phrase = "center"
    if horiz != "center" and vert != "center":
        direction_phrase = f"{horiz} and slightly {vert}"
    elif horiz != "center":
        direction_phrase = horiz
    elif vert != "center":
        direction_phrase = f"slightly {vert}"

    h_horiz = "left" if dxh < -0.04 else ("right" if dxh > 0.04 else "center")
    h_vert = "up" if dyh < -0.04 else ("down" if dyh > 0.04 else "center")
    h_forward = "forward" if dzh > 0.035 else ("back a little" if dzh < -0.035 else "hold depth")
    hand_side_cm = abs(dxh) * 100.0
    hand_vertical_cm = abs(dyh) * 100.0
    hand_forward_cm = abs(dzh) * 100.0
    side_step = _cm_to_body_step(hand_side_cm)
    vert_step = _cm_to_body_step(hand_vertical_cm)
    fwd_step = _cm_to_body_step(hand_forward_cm)

    if grasp_detected:
        phase = "complete-grasp"
    elif contact_detected:
        phase = "complete-touch"
    elif hand_visible and hand_to_target_m is not None:
        if hand_to_target_m > 0.42:
            phase = "orientation"
        elif hand_to_target_m > 0.18:
            phase = "approach"
        elif hand_to_target_m > max(float(HAND_CONTACT_DIST_M) * 1.2, 0.08):
            phase = "fine-tuning"
        else:
            phase = "grasp"
    else:
        if d > 0.80:
            phase = "orientation"
        elif d > 0.26:
            phase = "approach"
        else:
            phase = "fine-tuning"

    key = f"{target}:{phase}:{horiz}:{vert}:{int(round(d * 10))}:{int(round(conf * 10))}"
    primary_danger = _pick_primary_danger(dangerous_objects, hand_xyz, target_xyz)
    hazard = primary_danger
    if hazard is None:
        hazard = _pick_depth_blocking_hazard(focus_obj, objects) if (depth_informed and depth_available) else None
        if hazard is None:
            hazard = _pick_primary_hazard(focus_obj, objects)
    collision_note = ""
    hazard_prefix = ""
    hazard_name = ""
    hazard_level = "low"
    hazard_penalty = 0.0
    corridor_blocked = False
    if hazard is not None:
        hname = str(hazard.get("name", "object"))
        hazard_name = hname
        hx, _, _, hd, _ = _obj_pose(hazard)
        hxyz = hazard.get("xyz_m", [0.0, 0.0, 0.0]) if isinstance(hazard, dict) else [0.0, 0.0, 0.0]
        if isinstance(hxyz, (list, tuple)) and len(hxyz) >= 3 and hand_xyz is not None:
            try:
                corridor_blocked = _danger_blocks_reach_path(
                    (float(hxyz[0]), float(hxyz[1]), float(hxyz[2])),
                    hand_xyz,
                    target_xyz,
                )
            except Exception:
                corridor_blocked = False
        side = "left" if hx < x else "right"
        depth_gap_cm = int(round((hd - d) * 100.0))
        danger_level = str(hazard.get("level", "")).lower()
        hazard_level = danger_level or "medium"
        hazard_penalty = float(_risk_score_for_object(hname, str(hazard.get("danger_type", ""))))
        if corridor_blocked:
            hazard_prefix = _txt(
                (
                    f"Caution, dangerous {hname} is on your current reach corridor. "
                    "Pause, move your hand slightly away from it, then continue with a narrow straight path. "
                ),
                (
                    f"Canh bao, {hname} nguy hiem dang nam tren duong voi tay. "
                    "Tam dung, ne tay sang ben nhe roi di tiep theo duong hep va thang. "
                ),
            )
        elif depth_gap_cm <= -4:
            hazard_prefix = _txt(
                (
                    f"Caution, {hname} is in front of {target} by about {abs(depth_gap_cm)} centimeters. "
                    "Narrow your approach path. "
                ),
                (
                    f"Canh bao, {hname} nam truoc {target} khoang {abs(depth_gap_cm)} cm. "
                    "Thu hep duong di cua tay. "
                ),
            )
        elif depth_gap_cm >= 4:
            hazard_prefix = _txt(
                f"Caution, {hname} is just behind {target}. Slow down before final reach. ",
                f"Canh bao, {hname} nam sau {target}. Giam toc truoc buoc cuoi.",
            )
        else:
            hazard_prefix = _txt(
                f"Caution, {hname} is on your {side} side near the reach path. ",
                f"Canh bao, {hname} nam ben {side} gan duong voi tay.",
            )
        if danger_level == "high":
            hazard_prefix = _txt(
                f"{hazard_prefix}This is a high-risk object. ",
                f"{hazard_prefix}Day la vat co rui ro cao. ",
            )
        collision_note = (
            _txt(
                f" Obstacle alert: {hname} is near your {side} path at about {hd:.2f} meters. Keep motion narrow and avoid sweeping.",
                f" Canh bao va cham: {hname} dang gan duong {side} cua tay, cach khoang {hd:.2f} m. Di tay hep va tranh quat ngang.",
            )
        )

    cadence_sec = 2.2
    if phase == "orientation":
        cadence_sec = 2.2
    elif phase == "approach":
        cadence_sec = 1.6
    elif phase == "fine-tuning":
        cadence_sec = 1.0
    elif phase in {"grasp", "complete-touch"}:
        cadence_sec = 0.85
    elif phase == "complete-grasp":
        cadence_sec = 2.5

    variants: List[str] = []
    if phase == "complete-grasp":
        variants = [
            _txt(
                f"{target} is secured in your hand. Hold steady, keep it upright, and move back slowly.",
                f"Ban da nam chac {target}. Giu on dinh, giu vat dung va rut tay lai cham.",
            ),
            _txt(
                f"Grasp confirmed on {target}. Keep your wrist stable and lift straight up gently.",
                f"Da xac nhan nam {target}. Giu co tay on dinh va nhac len thang nhe nhang.",
            ),
        ]
    elif phase == "complete-touch":
        variants = [
            _txt(
                f"Contact confirmed on {target}. Close your fingers gently now, then lift a little.",
                f"Da cham vao {target}. Khep ngon tay nhe ngay bay gio roi nhac len mot chut.",
            ),
            _txt(
                f"You touched {target}. Keep your palm vertical, wrap fingers around it, then hold.",
                f"Ban da cham {target}. Giu long ban tay dung, om quanh vat roi giu chac.",
            ),
        ]
    elif phase == "grasp":
        variants = [
            _txt(
                f"Final micro-step. Move {h_horiz} {side_step}, {h_vert} {vert_step}, then nudge {h_forward} and feel contact.",
                f"Buoc vi chinh cuoi. Dua tay {h_horiz} {side_step}, {h_vert} {vert_step}, roi nhich {h_forward} de cham vat.",
            ),
            _txt(
                f"{target} is in grasp range. Keep palm vertical like a handshake, tiny forward nudge, then close fingers.",
                f"{target} da vao tam nam. Giu long ban tay dung nhu bat tay, nhich toi rat nhe roi khep ngon tay.",
            ),
        ]
    elif phase == "fine-tuning":
        variants = [
            _txt(
                f"Very close now. Shift {h_horiz} {side_step}, adjust {h_vert} {vert_step}, and move {h_forward} only a tiny bit.",
                f"Da rat gan. Dich {h_horiz} {side_step}, chinh {h_vert} {vert_step}, va di {h_forward} that nhe.",
            ),
            _txt(
                f"Precision step: keep your wrist soft, tiny correction to center, then advance fingertip by fingertip.",
                f"Buoc chinh xac: tha long co tay, chinh nhe ve giua, roi tien toi tung chut theo dau ngon tay.",
            ),
        ]
    elif phase == "approach":
        variants = [
            _txt(
                f"Good approach. Keep moving in direction {clock_direction}. Move {h_horiz} {side_step}, then {h_forward} {fwd_step}.",
                f"Tiep can tot. Giu huong {clock_direction}. Dua tay {h_horiz} {side_step}, roi {h_forward} {fwd_step}.",
            ),
            _txt(
                f"{target} is around {distance_human}. Keep motion smooth and short, avoid wide side sweep.",
                f"{target} cach {distance_human}. Di tay ngan, deu, tranh quat ngang rong.",
            ),
        ]
    else:
        variants = [
            _txt(
                f"{target} is at {clock_direction}, {distance_human}. Start broad alignment first, then move forward slowly.",
                f"{target} o huong {clock_direction}, cach {distance_human}. Canh huong rong truoc, roi tien toi cham.",
            ),
            _txt(
                f"Orientation phase. Keep camera centered, align your hand to {clock_direction}, then advance with short pushes.",
                f"Giai doan dinh huong. Giu camera o giua, dua tay theo {clock_direction}, roi day toi tung doan ngan.",
            ),
        ]
        if hand_visible and hand_xyz is None:
            variants.append(
                _txt(
                    "I can see your hand, but hand depth is still calibrating. Keep your hand centered in camera view.",
                    "Toi thay tay ban nhung do sau cua tay dang hieu chinh. Hay giu tay o giua khung hinh.",
                )
            )

    if hazard_prefix:
        variants = [_sanitize_guide_text(f"{hazard_prefix}{v}") for v in variants]
    if scene_memory_note and phase in {"orientation", "approach"}:
        variants.append(_sanitize_guide_text(f"{variants[0]} {scene_memory_note}"))

    idx = abs(hash(key)) % max(1, len(variants))
    say = _sanitize_guide_text(variants[idx])
    prev_norm = _normalize_guide_text(previous_hint)
    curr_norm = _normalize_guide_text(say)
    if _text_similarity(prev_norm, curr_norm) > 0.92 and len(variants) > 1:
        say = _sanitize_guide_text(variants[(idx + 1) % len(variants)])

    micro_steps = [
        _txt(f"Align hand toward {clock_direction}.", f"Canh tay theo huong {clock_direction}."),
        _txt(f"Move {h_horiz} {side_step} and {h_vert} {vert_step}.", f"Dich tay {h_horiz} {side_step} va {h_vert} {vert_step}."),
        _txt(f"Advance {h_forward} {fwd_step}.", f"Tien tay {h_forward} {fwd_step}."),
    ]
    if hazard_name:
        micro_steps.insert(
            0,
            _txt(
                f"Avoid {hazard_name}; keep a narrow path.",
                f"Ne {hazard_name}; giu duong di hep.",
            ),
        )
    if phase in {"complete-touch", "complete-grasp"}:
        micro_steps = [
            _txt("Hold wrist steady.", "Giu co tay on dinh."),
            _txt("Close fingers gently around target.", "Khep ngon tay nhe quanh vat."),
            _txt("Lift straight up slowly.", "Nang thang len cham."),
        ]

    conf_out = float(np.clip(conf, 0.0, 1.0))
    if depth_available:
        conf_out = float(np.clip(0.78 * conf_out + 0.22 * depth_rel_score, 0.0, 1.0))
    if hand_visible:
        conf_out = float(np.clip(conf_out + (0.05 if hand_xyz is not None else 0.02), 0.0, 1.0))
    if hand_to_target_m is not None and hand_to_target_m <= 0.22:
        conf_out = float(np.clip(conf_out + 0.05, 0.0, 1.0))
    if contact_detected:
        conf_out = float(np.clip(conf_out + 0.08, 0.0, 1.0))
    if grasp_detected:
        conf_out = float(np.clip(conf_out + 0.06, 0.0, 1.0))

    safety_score = float(np.clip(1.0 - (0.55 * hazard_penalty + (0.25 if corridor_blocked else 0.0)), 0.05, 1.0))
    risk_level = "low"
    if corridor_blocked or hazard_level == "high" or safety_score < 0.45:
        risk_level = "high"
    elif hazard_name or safety_score < 0.70:
        risk_level = "medium"

    primary_danger_out = None
    if isinstance(primary_danger, dict):
        primary_danger_out = {
            "name": str(primary_danger.get("name", "object")),
            "level": str(primary_danger.get("level", "")),
            "danger_type": str(primary_danger.get("danger_type", "")),
            "distance_m": round(float(primary_danger.get("distance_m", -1.0)), 3),
            "distance_to_hand_m": (
                round(float(primary_danger.get("distance_to_hand_m", -1.0)), 3)
                if primary_danger.get("distance_to_hand_m", None) is not None
                else None
            ),
            "corridor_blocked": bool(corridor_blocked),
        }

    intent_agent = {
        "task": user_query or f"Reach {target}",
        "target_object": target,
        "confidence": round(float(0.55 + 0.35 * conf_out), 2),
    }
    spatial_agent = {
        "target_visible": True,
        "target_xyz_m": [round(float(target_xyz[0]), 3), round(float(target_xyz[1]), 3), round(float(target_xyz[2]), 3)],
        "target_distance_m": round(float(d), 3),
        "clock_direction": clock_direction,
        "hand_to_target_m": round(float(hand_to_target_m), 3) if hand_to_target_m is not None else None,
        "recommended_approach": f"{phase} via {clock_direction}",
    }
    safety_agent = {
        "risk_level": risk_level,
        "hazards": ([hazard_name] if hazard_name else []),
        "collision_objects": ([hazard_name] if corridor_blocked and hazard_name else []),
        "safety_score": round(float(safety_score), 2),
        "corridor_blocked": bool(corridor_blocked),
    }
    path_agent = {
        "phase": phase,
        "micro_steps": micro_steps[:4],
        "stop_conditions": [
            _txt("Unexpected hard contact.", "Cham manh bat ngo."),
            _txt("Target leaves camera view.", "Mat muc tieu khoi khung hinh."),
        ],
        "adaptive_voice_cadence_sec": round(float(cadence_sec), 2),
    }

    return _base_fields({
        "say": say,
        "target": target,
        "distance_m": round(d, 3),
        "direction": horiz,
        "clock_direction": clock_direction,
        "distance_human": distance_human,
        "confidence": round(conf_out, 2),
        "safety_note": _txt(
            f"Move slowly and stop on unexpected contact.{collision_note}",
            f"Di tay cham va dung lai neu cham bat thuong.{collision_note}",
        ),
        "scene_summary": scene_summary,
        "scene_lines": scene_lines,
        "phase": phase,
        "depth_mode": depth_mode,
        "depth_reliability": round(float(depth_rel_score), 2),
        "depth_note": depth_note,
        "hand_to_target_m": round(float(hand_to_target_m), 3) if hand_to_target_m is not None else None,
        "contact_detected": bool(contact_detected),
        "contact_score": round(float(contact_score), 3),
        "grasp_detected": bool(grasp_detected),
        "grasp_score": round(float(grasp_score), 3),
        "pinch_ratio": round(float(pinch_ratio), 3) if pinch_ratio is not None else None,
        "hand_cover_target": round(float(hand_cover_target), 3),
        "interaction_score": round(float(interaction_score), 3),
        "hand_target_iou": round(float(hand_target_iou), 3),
        "hand_target_overlap": round(float(hand_target_overlap), 3),
        "target_xyz_m": [round(float(target_xyz[0]), 3), round(float(target_xyz[1]), 3), round(float(target_xyz[2]), 3)],
        "primary_danger": primary_danger_out,
        "corridor_blocked": bool(corridor_blocked),
        "intent_agent": intent_agent,
        "spatial_agent": spatial_agent,
        "safety_agent": safety_agent,
        "path_agent": path_agent,
    })


def _render_realtime_guidance_markdown(
    guidance: Dict[str, Any],
    model_used: str,
    latency_ms: float,
    fallback_used: bool,
    error_message: str = "",
) -> str:
    def _f(val: Any, default: float = 0.0) -> float:
        try:
            return float(val)
        except Exception:
            return float(default)

    say = _sanitize_guide_text(str((guidance or {}).get("say", "")))
    target = str((guidance or {}).get("target", "unknown"))
    distance_m = _f((guidance or {}).get("distance_m", -1.0), -1.0)
    direction = str((guidance or {}).get("direction", "center"))
    confidence = _f((guidance or {}).get("confidence", 0.0), 0.0)
    safety_note = str((guidance or {}).get("safety_note", "")).strip()
    scene_summary = str((guidance or {}).get("scene_summary", "")).strip()
    scene_lines = (guidance or {}).get("scene_lines", [])
    if not isinstance(scene_lines, list):
        scene_lines = []
    budget_note = str((guidance or {}).get("budget_note", "")).strip()
    phase = str((guidance or {}).get("phase", "")).strip()
    depth_mode = str((guidance or {}).get("depth_mode", "")).strip()
    depth_note = str((guidance or {}).get("depth_note", "")).strip()
    depth_reliability = _f((guidance or {}).get("depth_reliability", 0.0), 0.0)
    guidance_style = str((guidance or {}).get("guidance_style", "enact")).strip()
    clock_direction = str((guidance or {}).get("clock_direction", "12 o'clock")).strip()
    distance_human = str((guidance or {}).get("distance_human", "")).strip()
    hand_visible = bool((guidance or {}).get("hand_visible", False))
    hand_distance_m = (guidance or {}).get("hand_distance_m", None)
    hand_to_target_m = (guidance or {}).get("hand_to_target_m", None)
    contact_detected = bool((guidance or {}).get("contact_detected", False))
    grasp_detected = bool((guidance or {}).get("grasp_detected", False))
    contact_score = _f((guidance or {}).get("contact_score", 0.0), 0.0)
    grasp_score = _f((guidance or {}).get("grasp_score", 0.0), 0.0)
    pinch_ratio = (guidance or {}).get("pinch_ratio", None)
    interaction_score = _f((guidance or {}).get("interaction_score", 0.0), 0.0)
    touch_streak = int((guidance or {}).get("touch_streak", 0) or 0)
    grasp_streak = int((guidance or {}).get("grasp_streak", 0) or 0)
    hand_xyz_m = (guidance or {}).get("hand_xyz_m", None)
    danger_count = int((guidance or {}).get("danger_count", 0) or 0)
    primary_danger = (guidance or {}).get("primary_danger", None)
    corridor_blocked = bool((guidance or {}).get("corridor_blocked", False))
    scene_memory_note = str((guidance or {}).get("scene_memory_note", "")).strip()
    safety_agent = (guidance or {}).get("safety_agent", {})
    path_agent = (guidance or {}).get("path_agent", {})

    lines = [
        f"Realtime guide: `{model_used}` | latency: `{latency_ms:.1f} ms` | fallback: `{fallback_used}`",
    ]
    if say:
        lines.append(f"**Voice**: {say}")
    lines.append(
        f"Target: `{target}` | distance: `{distance_m:.2f} m` | direction: `{direction}` | conf: `{confidence:.2f}`"
    )
    if phase:
        lines.append(f"Action phase: `{phase}`")
    lines.append(f"Guidance style: `{guidance_style}` | clock: `{clock_direction}`")
    if distance_human:
        lines.append(f"Human distance cue: {distance_human}")
    if depth_mode:
        lines.append(f"Depth mode: `{depth_mode}` | reliability: `{depth_reliability:.2f}`")
    if depth_note:
        lines.append(f"Depth note: {depth_note}")
    if hand_visible:
        hand_line = "Hand: `visible`"
        if hand_distance_m is not None:
            hand_line += f" | distance: `{_f(hand_distance_m):.2f} m`"
        if hand_to_target_m is not None:
            hand_line += f" | hand->target: `{_f(hand_to_target_m):.2f} m`"
        if contact_detected:
            hand_line += " | contact: `yes`"
        if grasp_detected:
            hand_line += " | grasp: `yes`"
        if contact_score > 0:
            hand_line += f" | contact_score: `{contact_score:.2f}`"
        if grasp_score > 0:
            hand_line += f" | grasp_score: `{grasp_score:.2f}`"
        if pinch_ratio is not None:
            hand_line += f" | pinch: `{_f(pinch_ratio):.2f}`"
        if interaction_score > 0:
            hand_line += f" | hand_obj_score: `{interaction_score:.2f}`"
        if isinstance(hand_xyz_m, (list, tuple)) and len(hand_xyz_m) >= 3:
            hand_line += f" | xyz=(`{_f(hand_xyz_m[0]):+.2f}`, `{_f(hand_xyz_m[1]):+.2f}`, `{_f(hand_xyz_m[2]):.2f}`)"
        lines.append(hand_line)
        if contact_detected:
            lines.append("Task status: `target touched`")
        if grasp_detected:
            lines.append("Task status: `target grasped`")
        if touch_streak > 0 or grasp_streak > 0:
            lines.append(f"Stability: touch_streak=`{touch_streak}` | grasp_streak=`{grasp_streak}`")
    else:
        lines.append("Hand: `not visible`")
    if danger_count > 0:
        danger_line = f"Danger objects: `{danger_count}`"
        if isinstance(primary_danger, dict):
            dn = str(primary_danger.get("name", "object"))
            dl = str(primary_danger.get("level", ""))
            dd = primary_danger.get("distance_m", None)
            danger_line += f" | primary: `{dn}`"
            if dl:
                danger_line += f" ({dl})"
            if dd is not None:
                danger_line += f" @ `{_f(dd):.2f} m`"
        if corridor_blocked:
            danger_line += " | corridor: `blocked`"
        lines.append(danger_line)
    if isinstance(safety_agent, dict):
        risk_level = str(safety_agent.get("risk_level", "")).strip()
        safety_score = safety_agent.get("safety_score", None)
        if risk_level:
            line = f"Safety agent: risk=`{risk_level}`"
            if safety_score is not None:
                line += f" | score=`{_f(safety_score):.2f}`"
            lines.append(line)
    if isinstance(path_agent, dict):
        cadence = path_agent.get("adaptive_voice_cadence_sec", None)
        if cadence is not None:
            lines.append(f"Adaptive voice cadence: `{_f(cadence):.2f}s`")
    if scene_memory_note:
        lines.append(f"Scene memory: {scene_memory_note}")
    if scene_summary:
        lines.append(f"Scene: {scene_summary}")
    if scene_lines:
        lines.append("Top objects:")
        for item in scene_lines[:5]:
            lines.append(f"- {item}")
    if safety_note:
        lines.append(f"Safety: {safety_note}")
    if budget_note:
        lines.append(f"Gemini budget: {budget_note}")
    if error_message:
        if "scene stable" in error_message.lower():
            lines.append(f"Engine note: `{error_message}`")
        else:
            lines.append(f"Error: `{error_message}`")
    return "\n\n".join(lines)


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


def _render_plan_markdown(
    plan: Dict[str, object],
    model_used: str,
    latency_ms: float,
    fallback_used: bool,
    error_message: str = "",
) -> str:
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
    if error_message:
        lines.append("")
        lines.append(f"Planner error: `{error_message}`")
    return "\n".join(lines)


def run_smart_planner(target_object: str, user_query: str, profile_name: str, api_key_input: str):
    worker = get_worker()
    scene_state = worker.latest_scene()
    if not scene_state:
        msg = "Scene state is empty. Let the webcam run briefly, then click plan again."
        return msg, "{}"

    query = _build_user_query(target_object, user_query)
    planner = GeminiMultiAgentPlanner(api_key=api_key_input.strip() if api_key_input else None)
    result = planner.plan(
        user_query=query,
        scene_state=scene_state,
        profile_name=profile_name,
        fov_deg=CAM_FOV_DEG,
    )

    fallback_used = False
    planner_error = ""
    if result.ok:
        plan = result.output
        model_used = result.model
        latency_ms = result.latency_ms
    else:
        fallback_used = True
        planner_error = str(result.error or "")
        plan = _build_local_fallback_plan(query, scene_state)
        model_used = "local-fallback"
        latency_ms = 0.0

    md = _render_plan_markdown(plan, model_used, latency_ms, fallback_used, planner_error)
    js = json.dumps(plan, ensure_ascii=False, indent=2)
    return md, js


DESCRIPTION = (
    "YOLOER V2 realtime webcam assistant (CPU/GPU). "
    "Object detection uses YOLOE (THU-MIG) with optimized speed and accuracy. "
    "The engine uses adaptive inference (fast frames + periodic hi-res refresh) and async depth to reduce lag. "
    "You can enable prompt classes to focus on target object groups. "
    "You can also enable MiDaS depth for realtime distance and 3D coordinate analysis."
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


def normalize_frame(frame) -> Optional[np.ndarray]:
    if frame is None:
        return None

    def _extract_payload(obj, depth: int = 0):
        if obj is None or depth > 6:
            return None
        if isinstance(obj, dict):
            for key in ("image", "composite", "background", "path", "url"):
                if key in obj and obj.get(key) is not None:
                    out = _extract_payload(obj.get(key), depth + 1)
                    if out is not None:
                        return out
            return None
        if isinstance(obj, (list, tuple)):
            for item in obj:
                out = _extract_payload(item, depth + 1)
                if out is not None:
                    return out
            return None
        return obj

    def _from_bytes(blob: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(blob, dtype=np.uint8)
        if arr.size == 0:
            return None
        dec = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if dec is None:
            return None
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    def _from_string(text: str) -> Optional[np.ndarray]:
        value = text.strip()
        if not value:
            return None
        if value.startswith("data:image"):
            try:
                _, encoded = value.split(",", 1)
                blob = base64.b64decode(encoded)
                return _from_bytes(blob)
            except Exception:
                return None
        path = Path(value)
        if path.is_file():
            try:
                with Image.open(path) as img:
                    return np.asarray(img.convert("RGB"))
            except Exception:
                return None
        return None

    data = _extract_payload(frame)
    if data is None:
        return None

    if isinstance(data, bytes):
        arr = _from_bytes(data)
        if arr is None:
            return None
    elif isinstance(data, str):
        arr = _from_string(data)
        if arr is None:
            return None
    elif isinstance(data, Image.Image):
        arr = np.asarray(data.convert("RGB"))
    else:
        arr = np.asarray(data)

    if arr.size == 0:
        return None

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pass
    else:
        return None

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def frame_to_imagedata(frame: Optional[np.ndarray], quality: int = OUTPUT_JPEG_QUALITY) -> Dict[str, object]:
    arr = normalize_frame(frame)
    if arr is None:
        arr = np.zeros((max(32, WEBCAM_CAPTURE_H), max(32, WEBCAM_CAPTURE_W), 3), dtype=np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(int(quality), 45, 95))],
    )
    if not ok:
        return {"path": None, "url": None, "meta": {"_type": "gradio.FileData"}}
    b64 = base64.b64encode(enc.tobytes()).decode("ascii")
    return {
        "path": None,
        "url": f"data:image/jpeg;base64,{b64}",
        "orig_name": "frame.jpg",
        "mime_type": "image/jpeg",
        "is_stream": False,
        "meta": {"_type": "gradio.FileData"},
    }


def frame_to_html(frame: Optional[np.ndarray]) -> str:
    data = frame_to_imagedata(frame)
    url = data.get("url")
    if not isinstance(url, str) or not url:
        return "<div style='height:300px;background:#111;color:#bbb;display:flex;align-items:center;justify-content:center;'>No frame</div>"
    return (
        "<div style='height:300px;background:#111;display:flex;align-items:center;justify-content:center;'>"
        f"<img src='{url}' style='max-width:100%;max-height:100%;width:100%;height:100%;object-fit:contain;'/>"
        "</div>"
    )


_local_theme = gr.themes.Base(
    font=["Space Grotesk", "Segoe UI", "sans-serif"],
    font_mono=["JetBrains Mono", "Consolas", "monospace"],
)

APP_CSS = """
:root {
  --panel-bg: #0b1620;
  --panel-bg-soft: #102232;
  --panel-border: #1e435d;
  --text-strong: #edf7ff;
  --text-soft: #a8c7dc;
  --accent: #21d49b;
  --accent-soft: #124f41;
  --accent-2: #2ea0ff;
}
#app-shell {
  background: radial-gradient(1300px 620px at -8% -18%, #12314a 0%, #0a1824 45%, #040a10 100%);
}
.hero-card {
  border: 1px solid var(--panel-border);
  background: linear-gradient(160deg, #0f2333 0%, #0a1623 100%);
  border-radius: 16px;
  padding: 16px 18px;
  margin-bottom: 12px;
  box-shadow: 0 12px 30px rgba(5, 12, 18, 0.45);
}
.hero-title {
  color: var(--text-strong);
  font-size: 28px;
  font-weight: 700;
  letter-spacing: 0.3px;
  margin: 0 0 4px 0;
}
.hero-sub {
  color: var(--text-soft);
  font-size: 14px;
  margin: 0;
}
.panel {
  border: 1px solid var(--panel-border);
  background: linear-gradient(180deg, var(--panel-bg) 0%, var(--panel-bg-soft) 100%);
  border-radius: 14px;
  padding: 12px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}
.panel-title {
  color: var(--text-strong);
  font-size: 12px;
  font-weight: 700;
  margin: 2px 2px 8px 2px;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}
.status-note {
  border: 1px solid #22516e;
  border-radius: 10px;
  background: #0d1f2e;
}
.gradio-container .gr-button {
  border-radius: 10px;
}
.hidden-speech-field {
  display: none !important;
}
.lyrics-shell {
  height: 240px;
  overflow-y: auto;
  border: 1px solid #24506a;
  border-radius: 12px;
  background: linear-gradient(180deg, #0a1a27 0%, #0a1420 100%);
  padding: 10px;
}
.lyrics-empty {
  color: #8fb5ca;
  font-size: 13px;
  opacity: 0.9;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
}
.lyric-row {
  padding: 8px 10px;
  border-radius: 9px;
  margin-bottom: 6px;
  border: 1px solid rgba(45, 111, 145, 0.28);
  background: rgba(10, 27, 39, 0.72);
  opacity: 0.72;
  transition: all 120ms linear;
}
.lyric-row.lyric-active {
  background: linear-gradient(90deg, rgba(33, 212, 155, 0.20) 0%, rgba(46, 160, 255, 0.14) 100%);
  border: 1px solid rgba(33, 212, 155, 0.62);
  box-shadow: 0 0 0 1px rgba(33, 212, 155, 0.18);
  opacity: 1;
}
.lyric-meta {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 4px;
  font-size: 11px;
  color: #95bfd8;
}
.lyric-time {
  color: #8fb2c7;
}
.lyric-phase {
  color: #d4f6ff;
  background: rgba(25, 73, 104, 0.65);
  border: 1px solid rgba(84, 155, 197, 0.55);
  border-radius: 999px;
  padding: 1px 8px;
}
.lyric-model {
  color: #c7ffef;
  background: rgba(27, 102, 80, 0.65);
  border: 1px solid rgba(33, 212, 155, 0.42);
  border-radius: 999px;
  padding: 1px 8px;
}
.lyric-text {
  color: #effbff;
  font-size: 14px;
  line-height: 1.36;
}
.focus-card {
  border: 1px solid #25668a;
  border-radius: 12px;
  background: linear-gradient(160deg, #102537 0%, #0b1928 100%);
  padding: 10px;
}
"""


with gr.Blocks(title="YOLOER V2 - Realtime Distance Estimation", theme=_local_theme, css=APP_CSS) as demo:
    gr.HTML(
        "<div id='app-shell'>"
        "<div class='hero-card'>"
        "<h1 class='hero-title'>HaptiSight Realtime Guide</h1>"
        f"<p class='hero-sub'>One-target focus, depth-aware safety, and realtime voice guidance for object reaching. Build: {APP_BUILD}</p>"
        "</div>"
        "</div>"
    )
    _default_profile_name = "High Accuracy" if "High Accuracy" in PROFILE_PRESETS else ("30fps-stable" if "30fps-stable" in PROFILE_PRESETS else "Realtime")
    _default_profile = PROFILE_PRESETS[_default_profile_name]

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=380):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("<div class='panel-title'>Camera Input</div>")
                webcam = gr.Image(
                    label="Webcam",
                    type="numpy",
                    sources=["webcam"],
                    streaming=True,
                    interactive=True,
                    height=320,
                    elem_id="webcam_input",
                )
                with gr.Row():
                    profile = gr.Dropdown(
                        choices=["Fast", "30fps-stable", "Balanced", "High Accuracy", "Ultra Accuracy", "Realtime", "Precision"],
                        value=_default_profile_name,
                        label="Vision mode",
                    )
                    camera_mode = gr.Dropdown(
                        choices=["auto", "front", "back"],
                        value="auto",
                        label="Camera mode",
                    )
                    cam_perm_btn = gr.Button("Enable Camera Permission", variant="primary")
                    cam_apply_btn = gr.Button("Apply Camera")
                    cam_list_btn = gr.Button("List Cameras")
                    voice_test_btn = gr.Button("Test Voice")
                camera_device_id = gr.Textbox(
                    label="Camera deviceId (optional)",
                    value="",
                    lines=1,
                    placeholder="Paste deviceId here to force specific camera",
                )
                cam_perm_status = gr.Textbox(
                    label="Camera permission status",
                    value="Click once if browser does not show camera prompt.",
                    interactive=False,
                    elem_classes=["status-note"],
                )

            with gr.Accordion("Realtime Guidance & Voice", open=True):
                gr.Markdown("<div class='focus-card'><b>Focus Mission</b>: enter only the object you want to reach. Gemini will auto-extract target and generate realtime steps.</div>")
                with gr.Row():
                    target_object = gr.Textbox(
                        label="Target object to reach",
                        value="cup",
                        lines=1,
                        placeholder="cup, bottle, apple, phone...",
                    )
                    task_query = gr.Textbox(
                        label="Extra context (optional)",
                        value="Guide gently, concise, and safe for visually impaired user.",
                        lines=1,
                        placeholder="optional detail about environment or preference",
                    )
                gemini_api_key = gr.Textbox(
                    label="Gemini API key (session only)",
                    type="password",
                    placeholder="Leave empty to use GEMINI_API_KEY env",
                )
                with gr.Row():
                    rt_guidance_enabled = gr.Checkbox(
                        value=True,
                        label="Enable realtime Gemini guidance",
                    )
                    depth_informed_guidance = gr.Checkbox(
                        value=True,
                        label="Depth-informed guidance (advanced, use with MiDaS)",
                    )
                    voice_enabled = gr.Checkbox(
                        value=True,
                        label="Speak guidance in browser",
                    )
                rt_guidance_interval = gr.Slider(
                    0.8,
                    6.0,
                    value=2.8,
                    step=0.1,
                    label="Guidance interval (sec) - Gemini is auto-throttled",
                )
                with gr.Row():
                    voice_lang = gr.Dropdown(
                        choices=["en-US", "vi-VN"],
                        value="en-US",
                        label="Voice language",
                    )
                    voice_rate = gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="Voice rate")
                    voice_pitch = gr.Slider(0.7, 1.3, value=1.0, step=0.05, label="Voice pitch")

            with gr.Accordion("Detection Settings", open=True):
                conf_slider = gr.Slider(0.10, 0.90, value=_default_profile["conf"], step=0.01, label="Confidence threshold")
                iou_slider = gr.Slider(0.10, 0.90, value=_default_profile["iou"], step=0.01, label="IoU threshold")
                size_slider = gr.Slider(192, 768, value=_default_profile["img_size"], step=32, label="Inference image size")
                max_det_slider = gr.Slider(1, 80, value=_default_profile["max_det"], step=1, label="Max detections")
                smooth_slider = gr.Slider(0.0, 0.95, value=_default_profile["smooth"], step=0.01, label="Distance smoothing")

            with gr.Accordion("Depth + Prompt Classes", open=True):
                depth_enabled = gr.Checkbox(value=bool(_default_profile["depth_enabled"]), label="Enable MiDaS depth")
                depth_alpha = gr.Slider(0.0, 0.7, value=_default_profile["depth_alpha"], step=0.01, label="Depth overlay alpha")
                depth_interval = gr.Slider(1, 12, value=_default_profile["depth_interval"], step=1, label="Depth update every N frames")
                prompt_enabled = gr.Checkbox(
                    value=True,
                    label="Enable YOLOE prompt classes (slower)",
                )
                class_prompt = gr.Textbox(
                    label="YOLOE prompt classes (comma-separated)",
                    value="",
                    lines=2,
                    placeholder="cup, bottle, apple, cell phone",
                )
                auto_target_prompt = gr.Checkbox(
                    value=True,
                    label="Auto add target classes from task query",
                )

        with gr.Column(scale=7, min_width=420):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("<div class='panel-title'>Live Result</div>")
                result = gr.HTML(
                    value="<div style='height:300px;background:#111;color:#bbb;display:flex;align-items:center;justify-content:center;'>Waiting for webcam...</div>",
                    label="Result",
                )
                stats = gr.Textbox(label="Runtime stats")
                gr.Markdown("<div class='panel-title'>Realtime Guidance Timeline</div>")
                guide_timeline = gr.HTML(
                    value=_render_guidance_timeline_html([]),
                    label="Guidance timeline",
                    elem_id="guide_timeline",
                )
                guide_md = gr.Markdown("Realtime guidance idle.")
                speech_payload = gr.Textbox(
                    value="",
                    label="speech_payload",
                    visible=True,
                    elem_id="speech_payload",
                    elem_classes=["hidden-speech-field"],
                )

            with gr.Accordion("Planner (On Demand)", open=False):
                plan_btn = gr.Button("Generate Smart Guidance (Gemini)")
                plan_md = gr.Markdown("Guidance plan will appear here.")
                plan_json = gr.Code(label="Planner JSON", language="json")

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
        "postprocess": False,
        "trigger_mode": "always_last",
        "concurrency_limit": 1,
        "stream_every": STREAM_EVERY_SEC,
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
            prompt_enabled,
            class_prompt,
            auto_target_prompt,
            target_object,
            task_query,
            profile,
            gemini_api_key,
            rt_guidance_enabled,
            rt_guidance_interval,
            depth_informed_guidance,
            voice_enabled,
            voice_lang,
            voice_rate,
            voice_pitch,
        ],
        outputs=[result, stats, guide_md, speech_payload, guide_timeline],
        **stream_kwargs,
    )
    cam_perm_btn.click(
        fn=None,
        inputs=[camera_mode, camera_device_id],
        js=(
            "(mode, deviceId) => {"
            "  const useMode = String(mode || 'auto').toLowerCase();"
            "  const useDevice = String(deviceId || '').trim();"
            "  const picker = window.__yoloerPickCamera;"
            "  if (typeof picker !== 'function') {"
            "    return 'Camera helper is not ready. Refresh page and retry.';"
            "  }"
            "  return picker(useMode, useDevice)"
            "    .then((msg) => {"
            "      window.__yoloerSpeechUnlocked = true;"
            "      try {"
            "        if (window.speechSynthesis) {"
            "          const u = new SpeechSynthesisUtterance('Voice ready');"
            "          u.lang = 'en-US';"
            "          window.speechSynthesis.cancel();"
            "          window.speechSynthesis.speak(u);"
            "        }"
            "      } catch (_e) {}"
            "      return msg + ' Voice unlocked.';"
            "    })"
            "    .catch((e)=>`Camera permission error: ${e?.message || e}`);"
            "}"
        ),
        outputs=[cam_perm_status],
        queue=False,
    )
    cam_apply_btn.click(
        fn=None,
        inputs=[camera_mode, camera_device_id],
        js=(
            "(mode, deviceId) => {"
            "  const picker = window.__yoloerPickCamera;"
            "  if (typeof picker !== 'function') {"
            "    return 'Camera helper is not ready. Refresh page and retry.';"
            "  }"
            "  return picker(String(mode || 'auto').toLowerCase(), String(deviceId || '').trim())"
            "    .catch((e)=>`Apply camera error: ${e?.message || e}`);"
            "}"
        ),
        outputs=[cam_perm_status],
        queue=False,
    )
    camera_mode.change(
        fn=None,
        inputs=[camera_mode, camera_device_id],
        js=(
            "(mode, deviceId) => {"
            "  const picker = window.__yoloerPickCamera;"
            "  if (typeof picker !== 'function') { return 'Camera helper is not ready.'; }"
            "  return picker(String(mode || 'auto').toLowerCase(), String(deviceId || '').trim())"
            "    .then((msg) => `Camera mode applied: ${msg}`)"
            "    .catch((e)=>`Camera mode error: ${e?.message || e}`);"
            "}"
        ),
        outputs=[cam_perm_status],
        queue=False,
    )
    camera_device_id.change(
        fn=None,
        inputs=[camera_mode, camera_device_id],
        js=(
            "(mode, deviceId) => {"
            "  const picker = window.__yoloerPickCamera;"
            "  if (typeof picker !== 'function') { return 'Camera helper is not ready.'; }"
            "  if (!String(deviceId || '').trim()) { return 'DeviceId empty. Using mode only.'; }"
            "  return picker(String(mode || 'auto').toLowerCase(), String(deviceId || '').trim())"
            "    .then((msg) => `Camera device applied: ${msg}`)"
            "    .catch((e)=>`Camera device error: ${e?.message || e}`);"
            "}"
        ),
        outputs=[cam_perm_status],
        queue=False,
    )
    cam_list_btn.click(
        fn=None,
        js=(
            "() => {"
            "  const listFn = window.__yoloerListCameras;"
            "  if (typeof listFn !== 'function') { return 'Camera list helper is not ready.'; }"
            "  return listFn().catch((e)=>`List camera error: ${e?.message || e}`);"
            "}"
        ),
        outputs=[cam_perm_status],
        queue=False,
    )
    voice_test_btn.click(
        fn=None,
        js=(
            "() => {"
            "  window.__yoloerSpeechUnlocked = true;"
            "  if (!window.speechSynthesis) { return 'SpeechSynthesis is not available in this browser.'; }"
            "  const u = new SpeechSynthesisUtterance('Voice test successful. Real-time guidance is ready.');"
            "  u.lang = 'en-US';"
            "  u.rate = 1.0;"
            "  u.pitch = 1.0;"
            "  window.speechSynthesis.cancel();"
            "  window.speechSynthesis.speak(u);"
            "  return 'Voice test played. If you hear sound, guidance voice is active.';"
            "}"
        ),
        outputs=[cam_perm_status],
        queue=False,
    )
    demo.load(
        fn=None,
        js=(
            "() => {"
            "if (window.__yoloerSpeechInit) { return 'Voice hook ready'; }"
            "window.__yoloerSpeechInit = true;"
            "window.__yoloerSpeechUnlocked = false;"
            "window.__yoloerSpeechLastToken = '';"
            "window.__yoloerSpeechLastRaw = '';"
            "window.__yoloerSpeechBusy = false;"
            "window.__yoloerSpeechQueued = null;"
            "window.__yoloerCameraPref = window.__yoloerCameraPref || { mode: 'auto', deviceId: '' };"
            "window.__yoloerGetVideoEl = () => {"
            "  const inScope = Array.from(document.querySelectorAll('#webcam_input video'));"
            "  const all = Array.from(document.querySelectorAll('video'));"
            "  const pool = inScope.length ? inScope : all;"
            "  if (!pool.length) { return null; }"
            "  pool.sort((a, b) => {"
            "    const aa = (a.clientWidth || 0) * (a.clientHeight || 0);"
            "    const bb = (b.clientWidth || 0) * (b.clientHeight || 0);"
            "    return bb - aa;"
            "  });"
            "  return pool[0] || null;"
            "};"
            "if (!window.__yoloerPatchedGUM && navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {"
            "  const origGUM = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);"
            "  window.__yoloerOrigGUM = origGUM;"
            "  navigator.mediaDevices.getUserMedia = (constraints = {}) => {"
            "    const c = (constraints && typeof constraints === 'object') ? { ...constraints } : {};"
            "    if (c.video !== false) {"
            "      const pref = window.__yoloerCameraPref || { mode: 'auto', deviceId: '' };"
            "      const vIn = (c.video && typeof c.video === 'object') ? { ...c.video } : {};"
            "      if (String(pref.deviceId || '').trim()) {"
            "        vIn.deviceId = { exact: String(pref.deviceId).trim() };"
            "      } else if (String(pref.mode || 'auto') === 'front') {"
            "        vIn.facingMode = { ideal: 'user' };"
            "      } else if (String(pref.mode || 'auto') === 'back') {"
            "        vIn.facingMode = { ideal: 'environment' };"
            "      }"
            "      if (!vIn.width) { vIn.width = { ideal: 1280 }; }"
            "      if (!vIn.height) { vIn.height = { ideal: 720 }; }"
            "      c.video = vIn;"
            "    }"
            "    return origGUM(c);"
            "  };"
            "  window.__yoloerPatchedGUM = true;"
            "}"
            "window.__yoloerPickCamera = async (mode, deviceId) => {"
            "  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {"
            "    throw new Error('getUserMedia is not available in this browser');"
            "  }"
            "  const useMode = String(mode || 'auto').toLowerCase();"
            "  const useDevice = String(deviceId || '').trim();"
            "  window.__yoloerCameraPref = { mode: useMode, deviceId: useDevice };"
            "  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });"
            "  const v = window.__yoloerGetVideoEl();"
            "  if (v) {"
            "    const old = v.srcObject;"
            "    v.srcObject = stream;"
            "    try { await v.play(); } catch (_e) {}"
            "    if (old && old.getTracks) { old.getTracks().forEach((t) => t.stop()); }"
            "  }"
            "  window.__yoloerCurrentCamera = { mode: useMode, deviceId: useDevice };"
            "  if (useDevice) { return `Camera active by deviceId: ${useDevice}`; }"
            "  if (useMode === 'front') { return 'Camera active: front/selfie'; }"
            "  if (useMode === 'back') { return 'Camera active: back/environment'; }"
            "  return 'Camera active: auto mode';"
            "};"
            "window.__yoloerListCameras = async () => {"
            "  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {"
            "    throw new Error('enumerateDevices is not available');"
            "  }"
            "  const devices = await navigator.mediaDevices.enumerateDevices();"
            "  const cams = devices.filter((d) => d.kind === 'videoinput');"
            "  if (!cams.length) { return 'No camera device found.'; }"
            "  const rows = cams.map((d, i) => `${i + 1}. ${d.label || 'Camera'} | id=${d.deviceId}`);"
            "  return `Found ${cams.length} camera(s):\\n${rows.join('\\n')}`;"
            "};"
            "window.addEventListener('pointerdown', () => { window.__yoloerSpeechUnlocked = true; }, { once: true });"
            "const pick = () => document.querySelector('#speech_payload textarea, #speech_payload input');"
            "const speak = () => {"
            "  const el = pick();"
            "  if (!el) { return; }"
            "  const raw = (el.value || '').trim();"
            "  if (!raw || raw === window.__yoloerSpeechLastRaw) { return; }"
            "  window.__yoloerSpeechLastRaw = raw;"
            "  let payload = null;"
            "  try { payload = JSON.parse(raw); } catch (err) { return; }"
            "  if (!payload || !payload.enabled) { return; }"
            "  const text = String(payload.text || '').trim();"
            "  if (!text) { return; }"
            "  const token = String(payload.token || '');"
            "  if (token && token === window.__yoloerSpeechLastToken) { return; }"
            "  if (token) { window.__yoloerSpeechLastToken = token; }"
            "  if (!window.speechSynthesis) { return; }"
            "  if (!window.__yoloerSpeechUnlocked) { return; }"
            "  const utter = new SpeechSynthesisUtterance(text);"
            "  utter.lang = String(payload.lang || 'en-US');"
            "  utter.rate = Math.max(0.6, Math.min(1.6, Number(payload.rate || 1.0)));"
            "  utter.pitch = Math.max(0.6, Math.min(1.6, Number(payload.pitch || 1.0)));"
            "  if (window.__yoloerSpeechBusy) {"
            "    window.__yoloerSpeechQueued = utter;"
            "    return;"
            "  }"
            "  window.__yoloerSpeechBusy = true;"
            "  utter.onend = () => {"
            "    window.__yoloerSpeechBusy = false;"
            "    const q = window.__yoloerSpeechQueued || null;"
            "    window.__yoloerSpeechQueued = null;"
            "    if (q && window.speechSynthesis) {"
            "      window.__yoloerSpeechBusy = true;"
            "      q.onend = () => { window.__yoloerSpeechBusy = false; };"
            "      window.speechSynthesis.speak(q);"
            "    }"
            "  };"
            "  utter.onerror = () => { window.__yoloerSpeechBusy = false; };"
            "  window.speechSynthesis.speak(utter);"
            "};"
            "window.__yoloerSpeechTimer = setInterval(speak, 180);"
            "window.__yoloerTimelineTimer = setInterval(() => {"
            "  const root = document.querySelector('#guide_timeline');"
            "  if (!root) { return; }"
            "  const shell = root.querySelector('.lyrics-shell');"
            "  const active = root.querySelector('.lyric-row.lyric-active');"
            "  if (!shell || !active) { return; }"
            "  const sr = shell.getBoundingClientRect();"
            "  const ar = active.getBoundingClientRect();"
            "  const margin = 18;"
            "  const out = (ar.top < (sr.top + margin)) || (ar.bottom > (sr.bottom - margin));"
            "  if (out) {"
            "    active.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });"
            "  }"
            "}, 260);"
            "window.addEventListener('beforeunload', () => {"
            "  if (window.__yoloerSpeechTimer) { clearInterval(window.__yoloerSpeechTimer); }"
            "  if (window.__yoloerTimelineTimer) { clearInterval(window.__yoloerTimelineTimer); }"
            "});"
            "return 'Voice hook ready';"
            "}"
        ),
        outputs=[cam_perm_status],
        queue=False,
    )

    plan_btn.click(
        run_smart_planner,
        inputs=[target_object, task_query, profile, gemini_api_key],
        outputs=[plan_md, plan_json],
        queue=False,
    )


if __name__ == "__main__":
    demo.launch()
