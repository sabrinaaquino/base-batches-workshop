"""05 - Video generation. Text-to-video and image-to-video, with cost discussion."""

from ._common import Cell, header, install_cell, setup_cell


NOTEBOOK = "05-video-generation.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "Video: from a sentence to a moving clip",
            "Venice exposes text-to-video and image-to-video. Both return short MP4 clips ready to "
            "drop into a marketing post or a UI demo. We will generate one of each and inspect the "
            "result inline.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. **Text-to-video** from a single sentence.\n"
            "2. **Image-to-video** that animates a still you generated in notebook 03.\n"
            "3. A short cost note so you know what you are signing up for.\n\n"
            "Cost: video is the most expensive endpoint Venice offers, plan on roughly $0.10-$0.50 "
            "per generated clip depending on model and length."),
        ("markdown", "## Setup"),
        install_cell("pillow"),
        setup_cell(),
        ("code",
            '''import base64, io, time, requests
from pathlib import Path
from IPython.display import Video, display

OUT = Path("assets_video")
OUT.mkdir(exist_ok=True)

VIDEO_BASE = "https://api.venice.ai/api/v1/video"  # check the Venice docs for current model names

def generate_video(prompt: str, model: str = "wan-2.2", out_name: str = "clip.mp4",
                   image_b64: str | None = None) -> Path:
    payload = {
        "model": model,
        "prompt": prompt,
        "duration_seconds": 5,
    }
    if image_b64:
        payload["init_image"] = image_b64

    start = requests.post(
        f"{VIDEO_BASE}/generate",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    start.raise_for_status()
    job_id = start.json().get("id") or start.json().get("job_id")
    print("Job started:", job_id)

    while True:
        status = requests.get(
            f"{VIDEO_BASE}/jobs/{job_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        ).json()
        state = status.get("status")
        print("...", state)
        if state in {"completed", "succeeded", "done"}:
            break
        if state in {"failed", "error"}:
            raise RuntimeError(status)
        time.sleep(5)

    url = status.get("video_url") or status.get("url")
    if url:
        data = requests.get(url, timeout=120).content
    else:
        data = base64.b64decode(status["video_b64"])

    path = OUT / out_name
    path.write_bytes(data)
    return path'''),
        ("markdown", "## 1. Text-to-video"),
        ("code",
            '''path = generate_video(
    "A small orange robot waving at the camera on a Tokyo rooftop at golden hour, cinematic, slow zoom in",
    out_name="t2v.mp4",
)
display(Video(str(path), embed=True))'''),
        ("markdown",
            "## 2. Image-to-video\n\n"
            "Take any still image, base64 it, and let the video model animate it. Great for product "
            "shots: generate one perfect product image, then animate the camera around it."),
        ("code",
            '''STILL_BASE64 = ""  # paste a base64 PNG here, or load from notebook 03
# Example: load a still you generated previously
candidate = Path("assets_generated/cyber_fox.png")
if candidate.exists():
    STILL_BASE64 = base64.b64encode(candidate.read_bytes()).decode()
    print(f"Loaded {candidate.name} ({len(STILL_BASE64)} base64 chars)")
else:
    print("Skipping image-to-video. Run notebook 03 first to generate a still, then re-run this cell.")'''),
        ("code",
            '''if STILL_BASE64:
    path = generate_video(
        "Slow dolly in, neon glow intensifies, soft mist rolls past the rooftop",
        out_name="i2v.mp4",
        image_b64=STILL_BASE64,
    )
    display(Video(str(path), embed=True))'''),
        ("markdown",
            "## Recap\n\n"
            "Video generation is the most expensive primitive in Venice and also the highest-leverage "
            "for marketing demos. Use it sparingly, cache aggressively, and prefer image-to-video "
            "when you already have a still you love. Next: `06-characters.ipynb`."),
    ]
