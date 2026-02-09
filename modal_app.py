import os
import sys

import modal

APP_NAME = "yoloer-v2-realtime"
PROJECT_DIR = "/root/hf_space_deploy"
CACHE_DIR = "/cache"

cache_volume = modal.Volume.from_name("yoloer-v2-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libsm6", "libxext6")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path=PROJECT_DIR)
    .env(
        {
            "HF_HOME": f"{CACHE_DIR}/hf",
            "TORCH_HOME": f"{CACHE_DIR}/torch",
            "XDG_CACHE_HOME": f"{CACHE_DIR}/xdg",
            "CPU_THREADS": os.getenv("CPU_THREADS", "7"),
            "MAX_FRAME_EDGE": os.getenv("MAX_FRAME_EDGE", "448"),
            "MAX_DEPTH_EDGE": os.getenv("MAX_DEPTH_EDGE", "256"),
            "HIRES_REFRESH_EVERY": os.getenv("HIRES_REFRESH_EVERY", "6"),
            "FAST_SIZE_DELTA": os.getenv("FAST_SIZE_DELTA", "64"),
        }
    )
)

modal_app = modal.App(APP_NAME)


@modal_app.function(
    image=image,
    cpu=8.0,
    memory=8192,
    timeout=60 * 60 * 24,
    scaledown_window=15 * 60,
    volumes={CACHE_DIR: cache_volume},
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
    cpu=2.0,
    memory=4096,
    timeout=20 * 60,
    volumes={CACHE_DIR: cache_volume},
)
def warmup():
    os.chdir(PROJECT_DIR)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    from app import get_engine  # noqa: WPS433

    get_engine()
    cache_volume.commit()
    return "warmup_done"
