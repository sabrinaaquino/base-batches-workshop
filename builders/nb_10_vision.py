"""10 - Vision. Use Venice's vision-language models to extract text, compare images,
parse documents into structured schemas, and call tools off image evidence.
"""

from ._common import Cell, header, install_cell, setup_cell


NOTEBOOK = "10-vision-understanding.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "Vision: turn pixels into structured answers and tool calls",
            "Most Venice chat models can also see. We will OCR a synthetic receipt, parse it into a "
            "Pydantic schema, diff two images, route a customer-service decision off a damaged-package "
            "photo, and bake-off three vision models on the same task. Same OpenAI-compatible message "
            "format you already know, with one extra `image_url` content part.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. **The Venice vision lineup** as a live DataFrame, with `maxImages` and "
            "`supportsVideoInput` columns so you can pick the right model.\n"
            "2. **OCR + structured extraction**: read a receipt and return a Pydantic schema you can "
            "feed straight into a database.\n"
            "3. **Multi-image diff**: feed two screenshots and ask the model what changed.\n"
            "4. **Vision + tool calling**: a damaged-package image makes the model call "
            "`refund_order` or `escalate_to_human`. Same pattern as OpenAI's GPT-4o vision cookbook.\n"
            "5. **Vision bake-off**: same image, three models, one DataFrame.\n\n"
            "Cost: vision pricing is the same as text on Venice. Most calls are fractions of a cent."),
        ("markdown", "## Setup"),
        install_cell("pillow pandas pydantic"),
        setup_cell(),
        ("code",
            '''import base64, io, json, time, requests
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw
from IPython.display import display
import pandas as pd

API_BASE = "https://api.venice.ai/api/v1"
HEADERS  = {"Authorization": f"Bearer {api_key}"}

OUT = Path("assets_vision"); OUT.mkdir(exist_ok=True)

def to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a data: URL ready for the OpenAI image_url content part."""
    buf = io.BytesIO(); img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()

def see(model: str, prompt: str, *images: Image.Image, **kw) -> str:
    """One-shot vision call. Pass any number of PIL images alongside a text prompt."""
    parts = [{"type": "text", "text": prompt}]
    for img in images:
        parts.append({"type": "image_url", "image_url": {"url": to_data_url(img)}})
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": parts}],
        **kw,
    )
    return r.choices[0].message.content

print("Vision helpers ready.")'''),
        ("markdown",
            "## 1. The Venice vision lineup\n\n"
            "Vision is a per-model capability flag, not a separate endpoint. Pull `/models?type=text` "
            "and filter on `supportsVision`. The `maxImages` column tells you how many images you "
            "can send in one call; `supportsVideoInput` tells you whether the model can also accept "
            "raw video frames as a `video_url` content part."),
        ("code",
            '''r = requests.get(f"{API_BASE}/models", params={"type": "text"}, headers=HEADERS, timeout=30)
r.raise_for_status()

rows = []
for m in r.json().get("data", []):
    caps = (m.get("model_spec") or {}).get("capabilities") or {}
    if not caps.get("supportsVision"):
        continue
    rows.append({
        "model":       m["id"],
        "max_images":  caps.get("maxImages", 1),
        "video_input": bool(caps.get("supportsVideoInput")),
        "ctx":         (m.get("model_spec") or {}).get("availableContextTokens"),
    })

vision_catalog = (
    pd.DataFrame(rows)
      .sort_values(["video_input", "max_images"], ascending=[False, False])
      .reset_index(drop=True)
)
vision_catalog'''),
        ("markdown",
            "Pick a model for the rest of the notebook. We use `qwen3-vl-235b-a22b` as the default "
            "(Qwen 3 VL is Venice's flagship open-source vision model, fast and accurate). For the "
            "bake-off later we will A/B it against Claude and Gemini."),
        ("code",
            '''VISION_MODEL = "qwen3-vl-235b-a22b"
print("Default vision model:", VISION_MODEL)'''),
        ("markdown",
            "## 2. OCR and structured extraction\n\n"
            "Vision-language models can do OCR, but the real win is OCR **plus** structured "
            "extraction in a single call. We synthesize a receipt with PIL (so the demo is "
            "self-contained and reproducible) and then ask the model to fill a Pydantic schema. "
            "The model handles the OCR, the layout interpretation, and the type coercion."),
        ("code",
            '''def make_receipt() -> Image.Image:
    img = Image.new("RGB", (440, 560), "white")
    d   = ImageDraw.Draw(img)
    lines = [
        ("CAFE PRIVADO",                     14, "header"),
        ("123 Main St, San Francisco",       40, "small"),
        ("Order #4781   Table 7",            70, "small"),
        ("2026-04-21  19:42",                90, "small"),
        ("-" * 38,                          120, "small"),
        ("1x  Cappuccino                4.50", 150, "row"),
        ("2x  Espresso                  6.00", 175, "row"),
        ("1x  Almond croissant          5.25", 200, "row"),
        ("1x  Sparkling water           3.00", 225, "row"),
        ("-" * 38,                          255, "small"),
        ("Subtotal                     18.75", 280, "row"),
        ("Tax (8.5%)                    1.59", 305, "row"),
        ("Tip                           3.50", 330, "row"),
        ("TOTAL                       23.84",  365, "total"),
        ("-" * 38,                          400, "small"),
        ("Paid: VISA ****4242",              430, "small"),
        ("Thank you!",                       470, "small"),
    ]
    for text, y, _kind in lines:
        d.text((24, y), text, fill="black")
    return img

receipt = make_receipt()
receipt.save(OUT / "receipt.png")
display(receipt)'''),
        ("code",
            '''from pydantic import BaseModel, Field

class LineItem(BaseModel):
    qty:   int    = Field(ge=1)
    name:  str
    price: float  = Field(ge=0)

class Receipt(BaseModel):
    merchant: str
    date:     str
    items:    list[LineItem]
    subtotal: float
    tax:      float
    tip:      Optional[float] = None
    total:    float
    payment:  Optional[str]   = None

SCHEMA_PROMPT = (
    "Extract this receipt into JSON matching this schema and nothing else:\\n"
    + json.dumps(Receipt.model_json_schema(), indent=2) +
    "\\nReturn JSON only. Numbers must be numbers, not strings."
)

raw = see("kimi-k2-6", SCHEMA_PROMPT, receipt,
          response_format={"type": "json_object"}, temperature=0)
parsed = Receipt.model_validate_json(raw)
parsed'''),
        ("markdown",
            "Two things to note. First, we used **`kimi-k2-6`** for this call instead of the "
            "default Qwen 3 VL because Kimi advertises `response_format=json_object` support, which "
            "lets us guarantee well-formed JSON and skip a try/except around the parse. Second, "
            "the same pattern works for invoices, IDs, prescriptions, business cards, web "
            "screenshots, anything with structured visual content."),
        ("markdown",
            "## 3. Multi-image diff\n\n"
            "Vision models can take multiple images in a single message (up to `max_images` from "
            "the catalog above). Useful for before/after screenshots, A/B design reviews, change "
            "detection, etc. We render two slightly different versions of the same UI and ask the "
            "model to spot every difference."),
        ("code",
            '''def render_dashboard(*, revenue: str, status: str, alert: bool) -> Image.Image:
    img = Image.new("RGB", (520, 320), (245, 247, 252))
    d   = ImageDraw.Draw(img)
    d.rectangle([(0, 0), (520, 48)], fill=(30, 41, 89))
    d.text((20, 18), "Venice Cookbook  -  Live dashboard", fill="white")

    d.rectangle([(20, 80),  (240, 200)], outline=(180, 188, 200), width=2)
    d.text((36, 96),  "Revenue (24h)",   fill=(70, 80, 100))
    d.text((36, 130), revenue,           fill=(20, 30, 60))

    d.rectangle([(280, 80), (500, 200)], outline=(180, 188, 200), width=2)
    d.text((296, 96),  "Inference status", fill=(70, 80, 100))
    d.text((296, 130), status,             fill=(20, 30, 60))

    d.rectangle([(20, 220), (500, 290)], outline=(180, 188, 200), width=2)
    if alert:
        d.text((36, 240), "[!] TEE attestation FAILED on enclave eu-3", fill=(180, 30, 30))
        d.text((36, 262), "Action required.",                            fill=(180, 30, 30))
    else:
        d.text((36, 240), "All systems nominal.", fill=(40, 130, 60))
    return img

before = render_dashboard(revenue="$12,480", status="Healthy", alert=False)
after  = render_dashboard(revenue="$11,920", status="Degraded", alert=True)

before.save(OUT / "dash_before.png"); after.save(OUT / "dash_after.png")
display(before); display(after)'''),
        ("code",
            '''diff = see(VISION_MODEL,
    "Image 1 is BEFORE, image 2 is AFTER. List every visual change as bullet points. "
    "Be specific about exact text changes (old -> new), color changes, and any new elements.",
    before, after,
    temperature=0,
)
print(diff)'''),
        ("markdown",
            "## 4. Vision + tool calling\n\n"
            "OpenAI's vision cookbook walks through a customer-service agent that looks at package "
            "photos and decides whether to refund or escalate. We do the same on Venice. The model "
            "gets two tools, looks at the image, and returns a `tool_calls` array we can dispatch."),
        ("code",
            '''def make_package(state: str) -> Image.Image:
    """Render a stylized 'package on a doorstep' image for one of three scenarios."""
    img = Image.new("RGB", (480, 360), (210, 220, 230))
    d   = ImageDraw.Draw(img)
    d.rectangle([(0, 280), (480, 360)], fill=(120, 100, 80))
    if state == "intact":
        d.rectangle([(160, 160), (320, 290)], fill=(184, 134, 90), outline="black", width=3)
        d.line([(160, 200), (320, 200)], fill="black", width=2)
        d.text((180, 170), "FRAGILE",   fill="black")
        d.text((180, 250), "Order #781", fill="black")
    elif state == "crushed":
        d.polygon([(150, 230), (200, 170), (250, 250), (220, 290), (150, 290)],
                  fill=(150, 100, 60), outline="black", width=3)
        d.polygon([(260, 200), (340, 180), (330, 290), (250, 290)],
                  fill=(160, 110, 70), outline="black", width=3)
        d.polygon([(360, 230), (410, 220), (410, 290), (350, 290)],
                  fill=(150, 100, 60), outline="black", width=3)
        for (x1, y1, x2, y2) in [(170, 200, 250, 180), (220, 220, 320, 240),
                                 (260, 260, 340, 250), (180, 270, 280, 290)]:
            d.line([(x1, y1), (x2, y2)], fill="black", width=4)
        d.text((180, 130), "FRAGILE",  fill="black")
        d.text((180, 100), "DAMAGED",  fill=(200, 30, 30))
        d.text((300, 130), "OPEN BOX", fill=(200, 30, 30))
        d.text((180, 305), "Order #781", fill="black")
    elif state == "missing":
        d.text((140, 200), "(empty doorstep, no package)", fill=(70, 70, 70))
    return img

intact   = make_package("intact")
crushed  = make_package("crushed")
missing  = make_package("missing")
display(intact); display(crushed); display(missing)'''),
        ("code",
            '''TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "refund_order",
            "description": "Issue a full refund to the customer. Use only when the package is clearly damaged.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id":   {"type": "string"},
                    "reason":     {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["order_id", "reason", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Hand off to a human agent. Use when the situation is ambiguous or the package is missing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "summary":  {"type": "string"},
                },
                "required": ["order_id", "summary"],
            },
        },
    },
]

SYSTEM = (
    "You are a delivery support agent. Look at the photo the customer sent. "
    "If the package is visibly damaged or destroyed, call refund_order. "
    "If the package looks intact, do nothing and reply that the delivery looks fine. "
    "If the photo shows no package or is ambiguous, call escalate_to_human."
)

def triage(image: Image.Image, label: str) -> dict:
    r = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text",      "text": "Customer says: 'My order #781 just arrived. Photo attached.'"},
                {"type": "image_url", "image_url": {"url": to_data_url(image)}},
            ]},
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
    )
    msg = r.choices[0].message
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        return {"label": label, "action": tc.function.name, "args": json.loads(tc.function.arguments)}
    return {"label": label, "action": "no_op", "args": {"reply": (msg.content or "").strip()}}

triage_rows = [triage(intact, "intact"),
               triage(crushed, "crushed"),
               triage(missing, "missing")]
pd.DataFrame(triage_rows)'''),
        ("markdown",
            "Three images, three different routes, zero hand-tuning. The model decided each one "
            "from the visual evidence alone. In production you would dispatch each `tool_calls` "
            "entry to your real refund / escalation systems."),
        ("markdown",
            "## 5. Vision bake-off\n\n"
            "Same prompt, same image, three vision models. Useful when you are choosing which "
            "model to standardize on, or when you want a consensus answer for high-stakes calls. "
            "We re-use the receipt from section 2 and time each model end-to-end."),
        ("code",
            '''CANDIDATES = ["qwen3-vl-235b-a22b", "claude-opus-4-6", "gemini-3-flash-preview"]
QUESTION   = ("What was the TOTAL on this receipt, what was the tax amount, and what "
              "payment method was used? Answer in one short sentence.")

rows = []
for model in CANDIDATES:
    t0 = time.perf_counter()
    try:
        ans = see(model, QUESTION, receipt, temperature=0, max_tokens=300)
    except Exception as e:
        ans = f"FAILED: {e}"
    rows.append({
        "model":  model,
        "latency_s": round(time.perf_counter() - t0, 2),
        "answer": (ans or "").strip(),
    })

pd.set_option("display.max_colwidth", 220)
pd.DataFrame(rows)'''),
        ("markdown",
            "## Recap\n\n"
            "- **Vision is just an extra `image_url` content part** on the standard chat completions "
            "endpoint. No new SDK, no new auth.\n"
            "- **Send PNG/JPEG as a `data:` URL** if the file is local. Public URLs work too, but "
            "data URLs are reproducible and survive offline runs.\n"
            "- **Pair vision with `response_format=json_object`** when you want guaranteed structured "
            "output. Use a model that advertises the capability (Kimi K2.6 in this notebook).\n"
            "- **Pair vision with `tools`** when you want the model to act on what it sees. The "
            "tool-calling behavior is identical to text-only chat.\n"
            "- **Bake off models** before you commit. Quality on visual tasks varies more by model "
            "than text tasks do, especially on dense documents and low-contrast images.\n\n"
            "Next, peek at `08-e2ee-encryption.ipynb` if you want to run vision tasks inside a "
            "TEE. The `e2ee-qwen3-5-122b-a10b` model accepts vision input and runs in an attested "
            "enclave, which is the combo for medical imaging, document review of contracts, and "
            "anything else where the picture itself is the secret."),
    ]
