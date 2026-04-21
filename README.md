# Venice Cookbook

> Hands-on Python notebooks for building with private, uncensored, multimodal AI on Venice.

Ten self-contained notebooks that walk through every Venice endpoint, including the things that make Venice different: **x402 wallet payments** (no API key, pay per call in USDC on Base), **E2EE inference** with **provable privacy** (encrypt client-side, only a hardware enclave can decrypt), and a **unified Crypto RPC** that talks to every major chain through a single key.

Every notebook runs in **Google Colab** with one click. No local setup needed.

---

## Notebooks

| # | Notebook | What you will learn | Open in Colab |
|---|---|---|---|
| 00 | [Quickstart](notebooks/00-quickstart.ipynb) | First call, streaming, model bake-off across 3 models with a latency table. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sabrinaaquino/venice-cookbook/blob/main/notebooks/00-quickstart.ipynb) |
| 01 | [Chat Completions](notebooks/01-chat-completions.ipynb) | Model bake-off with an LLM-as-judge leaderboard, Pydantic structured outputs, and a tool-calling agent loop. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sabrinaaquino/venice-cookbook/blob/main/notebooks/01-chat-completions.ipynb) |
| 02 | [Embeddings + RAG](notebooks/02-embeddings-and-rag.ipynb) | t-SNE visualization, Random Forest classification, k-means clustering with auto-naming, and a full mini-RAG pipeline. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sabrinaaquino/venice-cookbook/blob/main/notebooks/02-embeddings-and-rag.ipynb) |
| 03 | [Image Generation](notebooks/03-image-generation.ipynb) | First image, 4-style comparison grid, 3-frame visual story, and negative-prompt cleanup. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sabrinaaquino/venice-cookbook/blob/main/notebooks/03-image-generation.ipynb) |
| 04 | [Audio (TTS + STT)](notebooks/04-audio-tts-stt.ipynb) | Two-host podcast intro, silence trimming and segmentation, baseline transcription vs post-processed transcription. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sabrinaaquino/venice-cookbook/blob/main/notebooks/04-audio-tts-stt.ipynb) |
| 05 | [Video Generation](notebooks/05-video-generation.ipynb) | The full async flow (`quote` -> `queue` -> `retrieve` -> `complete`) across text-to-video, image-to-video, and reference-to-video, with live price quotes and a model discovery table. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sabrinaaquino/venice-cookbook/blob/main/notebooks/05-video-generation.ipynb) |
| 06 | [Characters](notebooks/06-characters.ipynb) | Single-character chat, three-character panel discussion, and character + RAG. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sabrinaaquino/venice-cookbook/blob/main/notebooks/06-characters.ipynb) |
| 07 | [x402 Wallet Payments](notebooks/07-x402-wallet-payments.ipynb) | One paid call, an autonomous agent with a transaction ledger, and pay-per-call vs flat-rate cost analysis. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sabrinaaquino/venice-cookbook/blob/main/notebooks/07-x402-wallet-payments.ipynb) |
| 08 | [End-to-End Encryption](notebooks/08-e2ee-encryption.ipynb) | Privacy mode comparison table, full ECDH (secp256k1) + AES-GCM handshake in pure Python, before/after diff of what Venice sees, and attestation inspection. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sabrinaaquino/venice-cookbook/blob/main/notebooks/08-e2ee-encryption.ipynb) |
| 09 | [Tools and Crypto RPC](notebooks/09-tools-and-rpc.ipynb) | Web Search (Brave ZDR), Web Scrape, Text Parser, multichain Crypto RPC with live cost-per-call from response headers, batched fan-out across Base/Optimism/Arbitrum/Polygon, and an LLM-powered onchain assistant. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sabrinaaquino/venice-cookbook/blob/main/notebooks/09-tools-and-rpc.ipynb) |

---

## Quickstart

### Option A: Run in Colab (recommended)

1. Click any **Open in Colab** badge above.
2. In Colab, open the **Secrets** panel (key icon, left sidebar) and add:
   - `VENICE_API_KEY`: get one at [venice.ai/settings/api](https://venice.ai/settings/api). Used by every notebook **except 07**.
   - `WALLET_PRIVATE_KEY` (optional, used by notebook 07): any Base wallet private key. Notebook 07 generates a throwaway wallet automatically if this is not set.
3. Run each cell.

### Option B: Run locally

```bash
git clone https://github.com/sabrinaaquino/venice-cookbook.git
cd venice-cookbook

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env       # then fill in your Venice API key

jupyter notebook
```

---

## What is inside

```
venice-cookbook/
├── README.md                          (you are here)
├── requirements.txt                   (all Python deps)
├── .env.example                       (template for local secrets)
├── helpers/
│   └── venice.py                      (shared client setup: env / Colab / .env)
├── builders/
│   ├── _common.py                     (shared cell helpers)
│   └── nb_*.py                        (one builder module per notebook)
├── notebooks/
│   ├── 00-quickstart.ipynb
│   ├── 01-chat-completions.ipynb
│   ├── 02-embeddings-and-rag.ipynb
│   ├── 03-image-generation.ipynb
│   ├── 04-audio-tts-stt.ipynb
│   ├── 05-video-generation.ipynb
│   ├── 06-characters.ipynb
│   ├── 07-x402-wallet-payments.ipynb
│   ├── 08-e2ee-encryption.ipynb
│   └── 09-tools-and-rpc.ipynb
├── build_notebooks.py                 (orchestrates the builders)
└── validate_notebooks.py              (syntax-checks every code cell)
```

The notebooks are generated from `builders/nb_*.py`. To edit a notebook, edit the matching builder and re-run:

```bash
python build_notebooks.py
python validate_notebooks.py
```

---

## Why Venice?

- **Same SDK as OpenAI.** Change one URL and every existing OpenAI client works.
- **Privacy you can verify.** TEE and E2EE inference signed by hardware enclaves. Math, not promises.
- **Pay with USDC.** No API key, no signup, no email. x402 turns your Base wallet into your account.
- **DIEM.** Lock VVV, mint DIEM, get a perpetual $1/day API credit. Tradeable. Forever.
- **Uncensored.** Open-source models with no content filters. Build the apps other providers will not ship.
- **One key, every chain.** Crypto RPC proxies JSON-RPC for Ethereum, Base, Arbitrum, Optimism, Polygon, Linea, Avalanche, BSC, Blast, zkSync Era, and Starknet. No more juggling Alchemy + Infura + dRPC accounts.

---

## Resources

- [Venice API docs](https://docs.venice.ai/)
- [Privacy architecture](https://venice.ai/privacy)
- [x402 guide](https://docs.venice.ai/overview/guides/x402-venice-api)
- [TEE and E2EE guide](https://docs.venice.ai/overview/guides/tee-e2ee-models)
- [DIEM tokenomics](https://venice.ai/lp/diem)
- [Venice Discord](https://discord.gg/venice-ai)

---

## License

MIT. See [LICENSE](LICENSE).

Built by the Venice team. Contributions welcome.
