import os
import sys

import modal

APP_NAME = "yoloer-v2-realtime-fixcam2"
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
            "USE_FP16": os.getenv("USE_FP16", "1"),
            "CPU_THREADS": os.getenv("CPU_THREADS", "4"),
            "YOLOE_MODEL_ID": os.getenv("YOLOE_MODEL_ID", "yoloe-11m"),
            "MAX_FRAME_EDGE": os.getenv("MAX_FRAME_EDGE", "640"),
            "MAX_OUTPUT_EDGE": os.getenv("MAX_OUTPUT_EDGE", "416"),
            "MAX_DEPTH_EDGE": os.getenv("MAX_DEPTH_EDGE", "384"),
            "HIRES_REFRESH_EVERY": os.getenv("HIRES_REFRESH_EVERY", "3"),
            "FAST_SIZE_DELTA": os.getenv("FAST_SIZE_DELTA", "0"),
            "RECOVERY_IMG_BOOST": os.getenv("RECOVERY_IMG_BOOST", "96"),
            "RECOVERY_MISS_THRESHOLD": os.getenv("RECOVERY_MISS_THRESHOLD", "2"),
            "DEPTH_MODEL": os.getenv("DEPTH_MODEL", "DPT_Hybrid"),
            "DEPTH_FP16": os.getenv("DEPTH_FP16", "1"),
            "WEBCAM_CAPTURE_W": os.getenv("WEBCAM_CAPTURE_W", "640"),
            "WEBCAM_CAPTURE_H": os.getenv("WEBCAM_CAPTURE_H", "360"),
            "WEBCAM_CAPTURE_FPS": os.getenv("WEBCAM_CAPTURE_FPS", "24"),
        }
    )
    .add_local_dir(
        ".",
        remote_path=PROJECT_DIR,
        ignore=[".git", ".torch_hub_cache", "__pycache__", "*.pt", ".venv", ".idea"],
    )
)

modal_app = modal.App(APP_NAME)
app = modal_app


@modal_app.function(
    image=image,
    gpu=GPU_CONFIG,
    cpu=6.0,
    memory=16384,
    timeout=60 * 60 * 24,
    scaledown_window=15 * 60,
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def web():
    import gradio as gr
    from fastapi import FastAPI

    os.chdir(PROJECT_DIR)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    from app import demo  # noqa: WPS433

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
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    from app import get_engine  # noqa: WPS433

    engine = get_engine()
    return {
        "status": "warmup_done",
        "device": engine.device,
        "use_cuda": bool(engine.use_cuda),
        "use_fp16": bool(engine.use_half),
        "detector": engine.detector_label,
        "depth_model": engine.depth_model_name,
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
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    from app import get_engine  # noqa: WPS433

    engine = get_engine()
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
