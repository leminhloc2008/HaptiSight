import os
import sys
import importlib.util

import modal

APP_NAME = os.getenv("MODAL_APP_NAME", "yoloer-v2-realtime-stable10-final1").strip() or "yoloer-v2-realtime-stable10-final1"
PROJECT_DIR = "/root/hf_space_deploy"
CACHE_DIR = "/tmp/yoloer_cache"

_gpu_name = os.getenv("MODAL_GPU", "A10G").strip().upper()
_gpu_map = {
    "T4": "T4",
    "L4": "L4",
    "A10G": "A10G",
    "L40S": "L40S",
    "A100": "A100-40GB",
    "H100": "H100",
    "ANY": "ANY",
}
GPU_CONFIG = _gpu_map.get(_gpu_name, "A10G")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libsm6", "libxext6")
    .pip_install_from_requirements("requirements.txt")
    .env(
        {
            "HF_HOME": f"{CACHE_DIR}/hf",
            "TORCH_HOME": f"{CACHE_DIR}/torch",
            "XDG_CACHE_HOME": f"{CACHE_DIR}/xdg",
            "FORCE_CPU": "0",
            "USE_FP16": os.getenv("USE_FP16", "0"),
            "CPU_THREADS": os.getenv("CPU_THREADS", "4"),
            "YOLOE_WEIGHTS": os.getenv("YOLOE_WEIGHTS", "yoloe-11m-seg.pt"),
            "YOLOE_MODEL_ID": os.getenv("YOLOE_MODEL_ID", "yoloe-11m"),
            "ACCURACY_RETRY_ENABLED": os.getenv("ACCURACY_RETRY_ENABLED", "1"),
            "ACCURACY_RETRY_CONF": os.getenv("ACCURACY_RETRY_CONF", "0.16"),
            "ACCURACY_RETRY_IMG": os.getenv("ACCURACY_RETRY_IMG", "768"),
            "PROMPT_CONF_CAP": os.getenv("PROMPT_CONF_CAP", "0.28"),
            "PROMPT_CONF_RETRY": os.getenv("PROMPT_CONF_RETRY", "0.20"),
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
            "GEMINI_TIMEOUT_S": os.getenv("GEMINI_TIMEOUT_S", "12"),
            "GUIDE_GEMINI_MIN_INTERVAL_SEC": os.getenv("GUIDE_GEMINI_MIN_INTERVAL_SEC", "2.8"),
            "GUIDE_GEMINI_RPM_LIMIT": os.getenv("GUIDE_GEMINI_RPM_LIMIT", "5"),
            "GUIDE_GEMINI_HOURLY_LIMIT": os.getenv("GUIDE_GEMINI_HOURLY_LIMIT", "120"),
            "GUIDE_GEMINI_BACKOFF_BASE_SEC": os.getenv("GUIDE_GEMINI_BACKOFF_BASE_SEC", "15"),
            "GUIDE_GEMINI_BACKOFF_MAX_SEC": os.getenv("GUIDE_GEMINI_BACKOFF_MAX_SEC", "600"),
            "GUIDE_GEMINI_SCENE_FORCE_SEC": os.getenv("GUIDE_GEMINI_SCENE_FORCE_SEC", "20"),
            "GUIDE_GEMINI_ALWAYS_ON": os.getenv("GUIDE_GEMINI_ALWAYS_ON", "1"),
            "GUIDE_TARGET_LOCK_SEC": os.getenv("GUIDE_TARGET_LOCK_SEC", "4.0"),
            "TARGET_QUERY_MAX_CLASSES": os.getenv("TARGET_QUERY_MAX_CLASSES", "6"),
            "GUIDE_DEPTH_HAZARD_X_M": os.getenv("GUIDE_DEPTH_HAZARD_X_M", "0.24"),
            "GUIDE_DEPTH_HAZARD_Y_M": os.getenv("GUIDE_DEPTH_HAZARD_Y_M", "0.20"),
            "GUIDE_DEPTH_HAZARD_FRONT_M": os.getenv("GUIDE_DEPTH_HAZARD_FRONT_M", "0.34"),
            "GUIDE_DEPTH_HAZARD_BEHIND_M": os.getenv("GUIDE_DEPTH_HAZARD_BEHIND_M", "0.24"),
            "MAX_FRAME_EDGE": os.getenv("MAX_FRAME_EDGE", "512"),
            "MAX_OUTPUT_EDGE": os.getenv("MAX_OUTPUT_EDGE", "320"),
            "OUTPUT_JPEG_QUALITY": os.getenv("OUTPUT_JPEG_QUALITY", "50"),
            "STREAM_EVERY_SEC": os.getenv("STREAM_EVERY_SEC", "0.05"),
            "MAX_DEPTH_EDGE": os.getenv("MAX_DEPTH_EDGE", "384"),
            "HIRES_REFRESH_EVERY": os.getenv("HIRES_REFRESH_EVERY", "4"),
            "FAST_SIZE_DELTA": os.getenv("FAST_SIZE_DELTA", "0"),
            "RECOVERY_IMG_BOOST": os.getenv("RECOVERY_IMG_BOOST", "0"),
            "RECOVERY_MISS_THRESHOLD": os.getenv("RECOVERY_MISS_THRESHOLD", "5"),
            "DEPTH_MODEL": os.getenv("DEPTH_MODEL", "DPT_Hybrid"),
            "DEPTH_FP16": os.getenv("DEPTH_FP16", "0"),
            "HAND_DETECT_ENABLED": os.getenv("HAND_DETECT_ENABLED", "1"),
            "HAND_DETECT_EVERY_N": os.getenv("HAND_DETECT_EVERY_N", "3"),
            "HAND_MAX_EDGE": os.getenv("HAND_MAX_EDGE", "256"),
            "HAND_MIN_SCORE": os.getenv("HAND_MIN_SCORE", "0.35"),
            "HAND_DETECT_MODE": os.getenv("HAND_DETECT_MODE", "mediapipe"),
            "HAND_YOLOE_ENABLED": os.getenv("HAND_YOLOE_ENABLED", "0"),
            "HAND_YOLOE_EVERY_N": os.getenv("HAND_YOLOE_EVERY_N", "5"),
            "HAND_YOLOE_IMG_SIZE": os.getenv("HAND_YOLOE_IMG_SIZE", "320"),
            "HAND_YOLOE_CONF": os.getenv("HAND_YOLOE_CONF", "0.22"),
            "HAND_YOLOE_MODEL_ID": os.getenv("HAND_YOLOE_MODEL_ID", "yoloe-v8s"),
            "HAND_SMOOTH_ALPHA": os.getenv("HAND_SMOOTH_ALPHA", "0.55"),
            "HAND_TARGET_REACH_M": os.getenv("HAND_TARGET_REACH_M", "0.12"),
            "HAND_CONTACT_DIST_M": os.getenv("HAND_CONTACT_DIST_M", "0.06"),
            "HAND_CONTACT_IOU": os.getenv("HAND_CONTACT_IOU", "0.18"),
            "WEBCAM_CAPTURE_W": os.getenv("WEBCAM_CAPTURE_W", "640"),
            "WEBCAM_CAPTURE_H": os.getenv("WEBCAM_CAPTURE_H", "360"),
            "WEBCAM_CAPTURE_FPS": os.getenv("WEBCAM_CAPTURE_FPS", "24"),
            "APP_BUILD": os.getenv("APP_BUILD", "2026-02-14-enact12-ui-gemini-billing-guard"),
            "HF_FALLBACK_URL": os.getenv("HF_FALLBACK_URL", "https://huggingface.co/spaces/lml2008/haptisight-realtime-gemini"),
            "MODAL_BILLING_LEFT_USD": os.getenv("MODAL_BILLING_LEFT_USD", ""),
            "MODAL_BILLING_GUARD_THRESHOLD_USD": os.getenv("MODAL_BILLING_GUARD_THRESHOLD_USD", "5"),
        }
    )
    .add_local_dir(
        ".",
        remote_path=PROJECT_DIR,
        ignore=[".git", ".torch_hub_cache", "__pycache__", ".venv", ".idea"],
        copy=True,
    )
    .run_commands(
        "python -c \"import os,urllib.request;"
        "p='/root/hf_space_deploy/mobileclip_blt.ts';"
        "os.makedirs(os.path.dirname(p), exist_ok=True);"
        "need=(not os.path.exists(p)) or (os.path.getsize(p) < 500_000_000);"
        "print('mobileclip_cached_before', os.path.exists(p));"
        "urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip_blt.ts', p) if need else None;"
        "print('mobileclip_size', os.path.getsize(p))\""
    )
)

modal_app = modal.App(APP_NAME)
app = modal_app


def _load_app_module():
    app_path = os.path.join(PROJECT_DIR, "app.py")
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    spec = importlib.util.spec_from_file_location("yoloer_runtime_app", app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load app module from {app_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@modal_app.function(
    image=image,
    gpu=GPU_CONFIG,
    cpu=6.0,
    memory=16384,
    timeout=60 * 60 * 24,
    scaledown_window=60 * 60,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import gradio as gr
    from fastapi import FastAPI

    os.chdir(PROJECT_DIR)
    module = _load_app_module()
    demo = module.demo
    get_engine = module.get_engine
    get_worker = module.get_worker

    # Preload model and worker once per container to reduce first-frame lag.
    try:
        get_engine()
        get_worker()
    except Exception as exc:
        print(f"[web preload] {exc}")

    api = FastAPI(title="YOLOER V2 on Modal")
    return gr.mount_gradio_app(api, demo, path="/")


@modal_app.function(
    image=image,
    gpu=GPU_CONFIG,
    cpu=2.0,
    memory=4096,
    timeout=20 * 60,
)
def warmup():
    os.chdir(PROJECT_DIR)
    module = _load_app_module()
    engine = module.get_engine()
    return {
        "status": "warmup_done",
        "device": engine.device,
        "use_cuda": bool(engine.use_cuda),
        "use_fp16": bool(engine.use_half),
        "detector": engine.detector_label,
        "depth_model": engine.depth_model_name,
        "hand_mode": getattr(engine, "hand_detect_mode", "unknown"),
        "hand_backend": getattr(getattr(engine, "hand_tracker", None), "backend", "none"),
        "hand_tracker_error": getattr(getattr(engine, "hand_tracker", None), "error_message", None),
        "hand_last_error": getattr(engine, "hand_last_error", None),
        "hand_yolo_enabled": bool(getattr(engine, "hand_yolo_enabled", False)),
        "hand_yolo_loaded": bool(getattr(engine, "hand_yolo_model", None) is not None),
        "hand_yolo_error": getattr(engine, "hand_yolo_error", None),
        "app_build": getattr(module, "APP_BUILD", "unknown"),
        "app_file": getattr(module, "__file__", "unknown"),
    }


@modal_app.function(
    image=image,
    gpu=GPU_CONFIG,
    cpu=4.0,
    memory=12288,
    timeout=20 * 60,
)
def benchmark(
    runs: int = 80,
    img_size: int = 576,
    depth_enabled: bool = True,
    depth_interval: int = 4,
):
    import time
    import numpy as np

    os.chdir(PROJECT_DIR)
    module = _load_app_module()
    engine = module.get_engine()
    frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        engine.infer(frame, 0.36, 0.52, img_size, 24, 0.45, depth_enabled, 0.09, depth_interval, "")

    lats = []
    for _ in range(max(10, int(runs))):
        t0 = time.perf_counter()
        _, stats, _ = engine.infer(frame, 0.36, 0.52, img_size, 24, 0.45, depth_enabled, 0.09, depth_interval, "")
        lats.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(lats, dtype=np.float32)
    return {
        "device": engine.device,
        "use_cuda": bool(engine.use_cuda),
        "use_fp16": bool(engine.use_half),
        "depth_model": engine.depth_model_name,
        "img_size": int(img_size),
        "depth_enabled": bool(depth_enabled),
        "depth_interval": int(depth_interval),
        "lat_ms_mean": float(arr.mean()),
        "lat_ms_p50": float(np.percentile(arr, 50)),
        "lat_ms_p90": float(np.percentile(arr, 90)),
        "fps_mean": float(1000.0 / max(arr.mean(), 1e-6)),
        "last_stats": stats,
    }
