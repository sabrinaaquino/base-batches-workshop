"""03 - Image generation. Prompt grid, style comparison, image-to-image edit, and a
visual storytelling pipeline."""

from ._common import Cell, header, install_cell, setup_cell


NOTEBOOK = "03-image-generation.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "Image generation: prompts, styles, edits, and visual stories",
            "Venice's image stack covers raw text-to-image, style presets, image-to-image edits, and "
            "an upscaler. We will build a prompt grid, a 3-frame visual story, and a quick edit "
            "pipeline that turns a sketch into a polished concept.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. **First image** in 3 lines of code.\n"
            "2. **Style grid:** the same prompt rendered in 4 styles, displayed side by side.\n"
            "3. **Visual storytelling:** generate a 3-frame story from a single concept.\n"
            "4. **Edit pipeline:** take an existing image and modify it.\n\n"
            "Cost: each image is fractions of a cent on the default tier."),
        ("markdown", "## Setup"),
        install_cell("pillow matplotlib"),
        setup_cell(),
        ("code",
            '''import base64, io, requests
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

OUT = Path("assets_generated")
OUT.mkdir(exist_ok=True)

def venice_image(prompt: str, model="venice-sd35", style_preset=None, width=1024, height=1024,
                 negative_prompt=None, seed=None):
    """Call /image/generate and return a PIL.Image."""
    payload = {
        "model": model,
        "prompt": prompt,
        "width": width,
        "height": height,
        "format": "png",
        "return_binary": False,
    }
    if style_preset:
        payload["style_preset"] = style_preset
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if seed is not None:
        payload["seed"] = seed
    r = requests.post(
        "https://api.venice.ai/api/v1/image/generate",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    b64 = data["images"][0] if isinstance(data["images"][0], str) else data["images"][0].get("b64_json")
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes))

def show(img, title=""):
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()'''),
        ("markdown", "## 1. Your first image"),
        ("code",
            '''img = venice_image(
    "A neon-lit cyberpunk fox standing on a Tokyo rooftop at night, photorealistic, 50mm lens",
    seed=42,
)
img.save(OUT / "cyber_fox.png")
show(img, "First image")'''),
        ("markdown",
            "## 2. Style grid: same prompt, four styles\n\n"
            "Venice ships a catalog of style presets. Here we lock the seed and prompt and only vary "
            "the style, then plot a 2x2 grid so you can see how dramatically the visual identity "
            "changes."),
        ("code",
            '''STYLES = ["Photographic", "Cinematic", "Anime", "Pixel Art"]
PROMPT = "A solo founder coding at 3am, single desk lamp, mug of coffee, ultra detailed"
SEED = 1234

images = []
for s in STYLES:
    try:
        images.append((s, venice_image(PROMPT, style_preset=s, seed=SEED)))
    except Exception as e:
        print(f"Skipped {s}: {e}")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, (style, img) in zip(axes.flatten(), images):
    ax.imshow(img); ax.set_title(style); ax.axis("off")
plt.tight_layout()
plt.show()'''),
        ("markdown",
            "Tip: pin the `seed` whenever you compare prompts or styles. Without a fixed seed every "
            "call is a fresh roll of the dice and you cannot tell if a change came from the prompt or "
            "from randomness."),
        ("markdown",
            "## 3. Visual storytelling\n\n"
            "Three prompts that share characters and palette become a tiny comic strip. The trick is "
            "to keep the subject description identical across prompts and only change the action / "
            "setting line."),
        ("code",
            '''CHARACTER = (
    "A small orange robot with one antenna and a tiny screen for a face, "
    "matte plastic finish, friendly proportions, isometric studio lighting"
)
SCENES = [
    f"{CHARACTER}, sitting alone in front of a glowing terminal, debugging code, sad expression on screen",
    f"{CHARACTER}, jumping up with stars on screen, paper coffee cup tipped over, code is now green",
    f"{CHARACTER}, on a tiny stage holding a trophy, confetti falling, screen shows :)",
]
story = [venice_image(s, style_preset="3D Model", seed=777 + i) for i, s in enumerate(SCENES)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, img, caption in zip(axes, story, ["Stuck", "Breakthrough", "Demo Day"]):
    ax.imshow(img); ax.set_title(caption); ax.axis("off")
plt.tight_layout()
plt.show()'''),
        ("markdown",
            "## 4. Negative prompts\n\n"
            "A negative prompt steers the model *away* from things you do not want. Compare the same "
            "prompt with and without `\"low quality, watermark, blurry\"` as a negative."),
        ("code",
            '''base = "A close-up portrait of a Brazilian developer wearing a Base hoodie, golden hour"
plain = venice_image(base, seed=99)
clean = venice_image(base, seed=99, negative_prompt="low quality, watermark, blurry, deformed hands")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(plain); axes[0].set_title("no negative prompt"); axes[0].axis("off")
axes[1].imshow(clean); axes[1].set_title("with negative prompt"); axes[1].axis("off")
plt.tight_layout(); plt.show()'''),
        ("markdown",
            "## Recap\n\n"
            "You shipped four real image flows: single generation, style comparison, visual story, "
            "and negative-prompted clean-up. Use the `seed` parameter aggressively in production: it "
            "is the only way to ship reproducible images. Next up: `04-audio-tts-stt.ipynb`."),
    ]
