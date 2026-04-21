"""00 - Quickstart. The absolute fastest path from zero to first call."""

from ._common import Cell, header, install_cell, setup_cell


NOTEBOOK = "00-quickstart.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "Quickstart: Your first Venice call in 60 seconds",
            "If you only run one notebook in this workshop, run this one. We will install the SDK, "
            "wire up your key, send a chat completion, and benchmark three models so you can pick the "
            "right one for the rest of the day.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. Install the OpenAI Python SDK (Venice is OpenAI-compatible, so the same client works).\n"
            "2. Authenticate with your `VENICE_API_KEY`.\n"
            "3. Send a chat completion and stream the response.\n"
            "4. Benchmark three models on the same prompt and rank them by latency.\n"
            "5. Decide which notebook to open next.\n\n"
            "**Cost:** every prompt in this notebook costs less than $0.001 on Venice's default tier. "
            "If you have a Pro key the same calls are free."),
        ("markdown", "## 1. Install"),
        install_cell(),
        ("markdown",
            "## 2. Set your API key\n\n"
            "Get one at [venice.ai/settings/api](https://venice.ai/settings/api). In Colab, add it under "
            "the key icon in the left sidebar with the name `VENICE_API_KEY`. Locally, copy `.env.example` "
            "to `.env` and fill it in."),
        setup_cell(),
        ("markdown",
            "## 3. Your first chat completion\n\n"
            "Venice exposes a `/chat/completions` endpoint that mirrors the OpenAI spec. The only thing "
            "that changes is the base URL. Everything else (messages, roles, tool calling, JSON mode, "
            "streaming) is identical."),
        ("code",
            '''resp = client.chat.completions.create(
    model="venice-uncensored",
    messages=[
        {"role": "system", "content": "You are a sharp, irreverent assistant. Reply in 2 sentences."},
        {"role": "user", "content": "Pitch Venice to a Base Batches builder in one tweet."},
    ],
)
print(resp.choices[0].message.content)'''),
        ("markdown", "## 4. Stream tokens as they arrive"),
        ("code",
            '''stream = client.chat.completions.create(
    model="venice-uncensored",
    messages=[{"role": "user", "content": "Write a 4-line haiku about end-to-end encryption."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()'''),
        ("markdown",
            "## 5. Benchmark three models\n\n"
            "Same prompt, three models, side by side. We report wall-clock latency, tokens used, and the "
            "first 120 characters of each response so you can eyeball quality. Pandas makes the table "
            "render nicely both in Colab and on GitHub."),
        ("code",
            '''import pandas as pd

PROMPT = "Explain why a Trusted Execution Environment is different from a regular cloud VM, in 3 bullets."
MODELS = ["llama-3.3-70b", "venice-uncensored", "qwen3-235b"]

rows = []
for model in MODELS:
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
    )
    elapsed = time.perf_counter() - t0
    rows.append({
        "model": model,
        "latency_s": round(elapsed, 2),
        "prompt_tokens": r.usage.prompt_tokens,
        "completion_tokens": r.usage.completion_tokens,
        "preview": shorten(r.choices[0].message.content.replace("\\n", " "), 120),
    })

df = pd.DataFrame(rows).sort_values("latency_s").reset_index(drop=True)
df'''),
        ("markdown",
            "## 6. Inspect what Venice actually sees\n\n"
            "Venice's default `Private` mode keeps zero logs of prompts or completions. To prove the API "
            "is plain HTTPS underneath, here is the raw `requests` version of the same call. Useful for "
            "debugging and for porting to languages without an OpenAI SDK."),
        ("code",
            '''import requests

r = requests.post(
    "https://api.venice.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json={
        "model": "llama-3.3-70b",
        "messages": [{"role": "user", "content": "Say hi in one word."}],
    },
    timeout=30,
)
print("status:", r.status_code)
print("model:", r.json()["model"])
print("text:", r.json()["choices"][0]["message"]["content"])'''),
        ("markdown",
            "## 7. Where to go next\n\n"
            "| If you want to... | Open |\n"
            "|---|---|\n"
            "| Build a chatbot or agent | `01-chat-completions.ipynb` |\n"
            "| Build search, RAG, or recommendations | `02-embeddings-and-rag.ipynb` |\n"
            "| Generate images | `03-image-generation.ipynb` |\n"
            "| Talk to or transcribe users | `04-audio-tts-stt.ipynb` |\n"
            "| Generate video | `05-video-generation.ipynb` |\n"
            "| Use named personas | `06-characters.ipynb` |\n"
            "| Pay with a wallet (no API key) | `07-x402-wallet-payments.ipynb` |\n"
            "| Make Venice unable to read your prompt | `08-e2ee-encryption.ipynb` |\n\n"
            "Have fun. Break things. Ask in #base-batches if you get stuck."),
    ]
