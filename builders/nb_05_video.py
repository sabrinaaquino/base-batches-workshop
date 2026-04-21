"""05 - Video generation. Quote, queue, retrieve, complete. Text/image/reference flows."""

from ._common import Cell, header, install_cell, setup_cell


NOTEBOOK = "05-video-generation.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "Video: quote, queue, retrieve, complete",
            "Venice's video API is async by design (generation takes a while). The whole platform "
            "is four endpoints: `quote` for the price tag, `queue` to kick off a job, `retrieve` to "
            "poll until it is ready, and `complete` to clean up the stored media when you are done. "
            "We will exercise every one of them across all three input modes: text-to-video, "
            "image-to-video, and reference-to-video.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. **Discover** the video models available to you via `/models?type=video`.\n"
            "2. **Quote** the price of a job before paying for it.\n"
            "3. **Text-to-video**: queue, poll, save MP4, display inline.\n"
            "4. **Image-to-video**: animate a still you generated in notebook 03.\n"
            "5. **Reference-to-video**: keep a character consistent across shots.\n"
            "6. **Cleanup**: `complete` the job and free the stored media.\n\n"
            "**Cost:** quotes are free. A typical 5-second 720p clip is around $0.05 to $0.20 "
            "depending on model and resolution. We will print the live quote before every job."),
        ("markdown", "## Setup"),
        install_cell("pillow"),
        setup_cell(),
        ("code",
            '''import base64, time, requests
from pathlib import Path
from IPython.display import Video, display, Image as IPyImage

OUT = Path("assets_video")
OUT.mkdir(exist_ok=True)

API = "https://api.venice.ai/api/v1"
H   = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

def jpost(path: str, body: dict, timeout: int = 60):
    r = requests.post(f"{API}{path}", headers=H, json=body, timeout=timeout)
    if not r.ok:
        try:
            print(f"!! {r.status_code} from {path}: {r.json()}")
        except Exception:
            print(f"!! {r.status_code} from {path}: {r.text[:300]}")
        r.raise_for_status()
    return r

print("Helpers ready.")'''),
        ("markdown",
            "## 1. Discover video models\n\n"
            "Append `?type=video` to `/models` to see only video models, what input types they "
            "accept (text vs image vs reference), and what durations and resolutions they support."),
        ("code",
            '''r = requests.get(f"{API}/models?type=video", headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
r.raise_for_status()
models = r.json().get("data", [])

import pandas as pd
rows = []
for m in models:
    spec = m.get("model_spec", {})
    caps = spec.get("capabilities", {}) or {}
    constraints = spec.get("constraints", {}) or {}
    rows.append({
        "id":         m["id"],
        "type":       caps.get("inputType") or caps.get("input_type") or "text",
        "max_dur":    constraints.get("maxDurationSeconds") or constraints.get("durations"),
        "resolutions":constraints.get("resolutions") or constraints.get("supportedResolutions"),
        "audio":      caps.get("supportsAudio", False),
    })

pd.DataFrame(rows)'''),
        ("markdown",
            "## 2. Get a price quote first\n\n"
            "Free, instant, and lets you decide whether the bill is worth it. Same body shape as "
            "`/video/queue`, but it just returns `{ \"quote\": <usd> }`. Use this to surface costs "
            "to your end users, or to gate expensive jobs behind a confirmation."),
        ("code",
            '''MODEL_T2V = "wan-2-7-text-to-video"

quote = jpost("/video/quote", {
    "model":      MODEL_T2V,
    "duration":   "5s",
    "resolution": "720p",
}).json()

print(f"This job will cost ${quote['quote']:.4f} USD ({quote['quote'] * 100:.2f} credits)")'''),
        ("markdown",
            "## 3. Text-to-video\n\n"
            "The async pattern is always the same:\n\n"
            "1. `POST /video/queue` -> get a `queue_id` (sometimes also a pre-signed `download_url`).\n"
            "2. `POST /video/retrieve` in a loop until you either get raw `video/mp4` bytes or a "
            "`COMPLETED` status pointing to the `download_url`.\n"
            "3. Save the file. Optionally `POST /video/complete` to delete the stored media.\n\n"
            "The helper below abstracts that loop. It works for every video model and every input "
            "type, because only the queue body changes."),
        ("code",
            '''def queue_video(body: dict) -> dict:
    """POST /video/queue. Returns the full JSON (model, queue_id, optionally download_url)."""
    return jpost("/video/queue", body, timeout=120).json()

def poll_video(model: str, queue_id: str, *, every: int = 5, max_wait: int = 600) -> bytes | None:
    """POST /video/retrieve in a loop until the job is done. Returns the MP4 bytes."""
    start = time.time()
    download_url = None
    while time.time() - start < max_wait:
        r = requests.post(
            f"{API}/video/retrieve",
            headers=H,
            json={"model": model, "queue_id": queue_id},
            timeout=120,
        )
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")

        if ctype.startswith("video/"):
            return r.content

        body = r.json()
        status = body.get("status", "?")
        avg = body.get("average_execution_time", 0) / 1000
        cur = body.get("execution_duration", 0) / 1000
        print(f"... {status}  ({cur:.0f}s of ~{avg:.0f}s)")

        if status == "COMPLETED":
            download_url = body.get("download_url") or download_url
            if download_url:
                d = requests.get(download_url, timeout=300)
                d.raise_for_status()
                return d.content
            print("(completed but no download_url returned, retrying)")
        elif status in {"FAILED", "ERROR"}:
            raise RuntimeError(body)

        time.sleep(every)
    raise TimeoutError(f"Job {queue_id} did not finish in {max_wait}s")

def complete_video(model: str, queue_id: str) -> dict:
    """POST /video/complete to delete the stored media on Venice's side."""
    return jpost("/video/complete", {"model": model, "queue_id": queue_id}).json()

JOBS = []  # remember (model, queue_id) so we can /video/complete later

def generate_video(body: dict, out_name: str) -> Path:
    """One-call wrapper: queue + retrieve + write to disk."""
    info = queue_video(body)
    print(f"Queued {info['model']} as {info['queue_id']}")
    JOBS.append((info["model"], info["queue_id"]))
    data = poll_video(info["model"], info["queue_id"])
    path = OUT / out_name
    path.write_bytes(data)
    print(f"Saved {path} ({path.stat().st_size / 1024:.0f} KB)")
    return path'''),
        ("code",
            '''path = generate_video({
    "model":        MODEL_T2V,
    "prompt":       "A lone gondola gliding through Venice canals at sunset, cinematic tracking shot, warm reflections on the water",
    "duration":     "5s",
    "resolution":   "720p",
    "aspect_ratio": "16:9",
}, out_name="t2v.mp4")

display(Video(str(path), embed=True))'''),
        ("markdown",
            "## 4. Image-to-video\n\n"
            "Same `/video/queue` endpoint, just include an `image_url`. The image carries the "
            "composition; your prompt should describe **motion, camera, and atmosphere** instead of "
            "describing the subject again. Pass either an HTTPS URL or a `data:image/png;base64,...` "
            "URL when you want to keep things fully local."),
        ("code",
            '''MODEL_I2V = "wan-2-7-image-to-video"

# Use a still you made in notebook 03 if it exists, otherwise pull a free Unsplash photo.
candidate = Path("assets_generated/cyber_fox.png")
if candidate.exists():
    b64 = base64.b64encode(candidate.read_bytes()).decode()
    image_url = f"data:image/png;base64,{b64}"
    print(f"Using local still: {candidate.name}")
else:
    image_url = "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=720&q=80"
    print("Using Unsplash fallback (run notebook 03 first to use your own).")

path = generate_video({
    "model":      MODEL_I2V,
    "prompt":     "Camera slowly pushes in, soft wind ruffles the fur, subtle volumetric light",
    "image_url":  image_url,
    "duration":   "5s",
    "resolution": "720p",
}, out_name="i2v.mp4")

display(Video(str(path), embed=True))'''),
        ("markdown",
            "## 5. Reference-to-video (consistency across shots)\n\n"
            "When you need the same character in shot after shot, `wan-2-7-reference-to-video` "
            "accepts an array of `reference_image_urls`. Refer to them in the prompt as `@Image1`, "
            "`@Image2`, etc. The endpoint is still `/video/queue`. Only the body changes."),
        ("code",
            '''MODEL_R2V = "seedance-2-0-fast-reference-to-video"

# Two references: a still object and a scene background. Swap with your own.
references = [
    "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=720&q=80",  # cat (subject)
    "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=720&q=80",  # field (background)
]

# Quote first so we know the bill before we commit
quote = jpost("/video/quote", {
    "model":        MODEL_R2V,
    "duration":     "5s",
    "resolution":   "720p",
    "aspect_ratio": "16:9",
}).json()
print(f"Reference-to-video quote: ${quote['quote']:.4f}")

# Reference-to-video models accept an array of images. Param shape varies by family;
# wrap in try/except so the rest of the notebook still completes if your account
# lacks access to this specific model.
try:
    path = generate_video({
        "model":                MODEL_R2V,
        "prompt":               "the cat walks slowly through the field, camera tracking from behind, golden hour light",
        "duration":             "5s",
        "resolution":           "720p",
        "aspect_ratio":         "16:9",
        "reference_image_urls": references,
    }, out_name="r2v.mp4")
    display(Video(str(path), embed=True))
except Exception as e:
    print(f"Skipped reference-to-video: {e}")
    print("(Some accounts gate the reference models. Try wan-2-7-reference-to-video or grok-imagine-reference-to-video.)")'''),
        ("markdown",
            "## 6. Cleanup: `/video/complete`\n\n"
            "Generated media stays on Venice servers until you delete it. You have two options:\n\n"
            "- **Inline:** add `\"delete_media_on_completion\": true` to your `/video/retrieve` body. "
            "One fewer round trip, but a network blip can leave the file orphaned.\n"
            "- **Explicit:** call `/video/complete` after you have safely written the file to disk. "
            "More resilient, especially in production pipelines.\n\n"
            "Below is the explicit form for the three jobs we just ran."),
        ("code",
            '''# Use the jobs we already ran instead of paying for a fresh one
for model, queue_id in JOBS:
    try:
        result = complete_video(model, queue_id)
        print(f"Cleaned {queue_id[:8]}... ({model}): {result}")
    except Exception as e:
        print(f"Already completed {queue_id[:8]}...: {e}")'''),
        ("markdown",
            "## Errors worth handling up front\n\n"
            "| Status | Meaning | What to do |\n"
            "|---|---|---|\n"
            "| `400` | Invalid body or unsupported parameter for that model | Validate against the model's `constraints` |\n"
            "| `401` | Bad API key or model needs a higher tier | Check auth and model access |\n"
            "| `402` | Insufficient balance | Top up Venice balance, or fund x402 wallet credits |\n"
            "| `404` | Invalid, expired, or already-deleted `queue_id` | Re-queue |\n"
            "| `413` | Reference image too large | Resize before sending |\n"
            "| `422` | Content policy violation | Adjust prompt or input assets |\n"
            "| `500` / `503` | Provider-side hiccup | Retry with backoff |\n\n"
            "Treat 500/503 as retriable, treat 404 as terminal, and back off when you hit 429."),
        ("markdown",
            "## Recap\n\n"
            "Four endpoints, three input modes, one async pattern.\n\n"
            "- `/video/quote` to know the bill before you pay it.\n"
            "- `/video/queue` for text-to-video, image-to-video, or reference-to-video. The body "
            "is the only thing that changes between modes.\n"
            "- `/video/retrieve` to poll until the MP4 is ready (raw bytes or pre-signed URL).\n"
            "- `/video/complete` to free the stored media when you are done.\n\n"
            "Pair this with notebook 03 (image generation) to build entire content pipelines: one "
            "perfect still, then animate it. Pair it with notebook 07 (x402) if you want each clip "
            "paid for from an agent's own wallet, no API key required."),
    ]
