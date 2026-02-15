import html
import os

import modal

APP_NAME = os.getenv("MODAL_APP_NAME", "yoloer-v2-realtime-stable10-final1").strip() or "yoloer-v2-realtime-stable10-final1"
HF_FALLBACK_URL = os.getenv("HF_FALLBACK_URL", "https://huggingface.co/spaces/lml2008/haptisight-realtime-gemini").strip()

modal_app = modal.App(APP_NAME)
app = modal_app

image = modal.Image.debian_slim(python_version="3.10")


def _notice_html() -> str:
    hf_url = html.escape(HF_FALLBACK_URL, quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HaptiSight Notice</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #071019;
      color: #eaf5ff;
      font-family: Segoe UI, Arial, sans-serif;
      padding: 24px;
    }}
    .card {{
      max-width: 860px;
      border: 1px solid #214d66;
      border-radius: 14px;
      background: #0d1f2f;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 12px 0;
      font-size: 30px;
    }}
    p {{
      margin: 0 0 10px 0;
      line-height: 1.55;
      font-size: 16px;
    }}
    a {{
      color: #23d89d;
      font-weight: 700;
      text-decoration: none;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>HaptiSight</h1>
    <p>Due to high usage, the owner has run out Modal quota.</p>
    <p>Please switch to the Hugging Face Space:</p>
    <p><a href="{hf_url}" target="_blank" rel="noopener">{hf_url}</a></p>
    <p>Note: Hugging Face runs on CPU, so realtime speed can be slower. For better speed and accuracy, run the project locally on your computer.</p>
  </div>
</body>
</html>
"""


@modal_app.function(
    image=image,
    cpu=0.5,
    memory=512,
    timeout=10 * 60,
    scaledown_window=60,
)
@modal.asgi_app()
def web():
    page = _notice_html().encode("utf-8")

    async def asgi_app(scope, receive, send):
        if scope.get("type") != "http":
            return
        headers = [
            (b"content-type", b"text/html; charset=utf-8"),
            (b"cache-control", b"no-store"),
        ]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": page})

    return asgi_app
