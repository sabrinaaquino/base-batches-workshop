"""09 - Venice tools: web search, web scrape, text parser, crypto networks, crypto RPC.

Crypto RPC is the headliner. It is the single endpoint that turns Venice into a
one-stop backend for any onchain agent: same auth, same billing, every major
EVM chain plus Starknet.
"""

from ._common import Cell, header, install_cell, setup_cell


NOTEBOOK = "09-tools-and-rpc.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "Tools: web search, scrape, parse, and the unified crypto RPC",
            "Venice ships five first-class tools as plain HTTP endpoints. Web Search, Web Scrape, "
            "and Text Parser pull external context into your prompts. Crypto Networks and Crypto RPC "
            "give your agents a single endpoint that speaks to Ethereum, Base, Arbitrum, Optimism, "
            "Polygon, Linea, Avalanche, BSC, Blast, zkSync Era, and Starknet. One key, one bill, "
            "every chain.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. **Web Search** with Brave (zero data retention) and a results DataFrame.\n"
            "2. **Web Scrape** chained into a chat completion to summarize any URL.\n"
            "3. **Text Parser** that lifts text out of a PDF in one HTTP call.\n"
            "4. **Crypto Networks** to discover supported chains.\n"
            "5. **Crypto RPC**: read your wallet balance on Base, batch a chain-ID + block-number + "
            "USDC balance call across four chains in a single HTTP round trip, and inspect the live "
            "credit cost from the response headers.\n"
            "6. **Onchain assistant**: a chat agent that uses Crypto RPC as a tool to answer "
            "questions like 'what is my ETH balance on Base?' in plain English.\n\n"
            "**Cost:** $0.01 per call for web search, scrape, and text parser. Crypto RPC is billed "
            "in credits at roughly $0.0000125 per standard EVM call. Networks list is free."),
        ("markdown", "## Setup"),
        install_cell("pandas requests reportlab"),
        setup_cell(),
        ("code",
            '''import requests, json, pandas as pd

API = "https://api.venice.ai/api/v1"
H   = {"Authorization": f"Bearer {api_key}"}

def post(path: str, *, json_body=None, files=None, data=None, extra_headers=None):
    headers = {**H, **(extra_headers or {})}
    if json_body is not None and files is None and data is None:
        headers["Content-Type"] = "application/json"
        return requests.post(f"{API}{path}", headers=headers, json=json_body, timeout=120)
    return requests.post(f"{API}{path}", headers=headers, files=files, data=data, timeout=120)

def get(path: str, *, headers=None):
    headers = {**H, **(headers or {})} if headers else H
    return requests.get(f"{API}{path}", headers=headers, timeout=30)

print("Helpers ready.")'''),
        ("markdown",
            "## 1. Web Search (Brave, zero data retention)\n\n"
            "One call, structured results. Brave is the default and never logs your query. Pass "
            "`search_provider=\"google\"` if you want Google results proxied through Venice (also "
            "anonymized). Returns title, url, snippet, and date for each hit."),
        ("code",
            '''r = post("/augment/search", json_body={
    "query": "venice ai api documentation",
    "limit": 8,
    "search_provider": "brave",
})
r.raise_for_status()
results = r.json()["results"]
df = pd.DataFrame(results)
df["snippet"] = df["content"].fillna("").str.replace(r"\\s+", " ", regex=True).str.slice(0, 140) + "..."
df[["title", "snippet", "url"]]'''),
        ("markdown",
            "## 2. Web Scrape\n\n"
            "Hand it any URL and get back clean markdown. Combine with chat for a one-shot "
            "summarizer of any page on the internet. The two-step flow (scrape, then summarize) is "
            "deterministic and cheap."),
        ("code",
            '''def scrape(url: str) -> str:
    r = post("/augment/scrape", json_body={"url": url})
    r.raise_for_status()
    body = r.json()
    return body.get("content") or body.get("markdown") or body.get("text") or ""

def summarize_url(url: str, model: str = "kimi-k2-6") -> str:
    md = scrape(url)
    if not md:
        return "(scraper returned empty content)"
    snippet = md[:6000]  # keep prompt small
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Summarize the article in 5 crisp bullets, then list any concrete numbers, dates, or names worth remembering."},
            {"role": "user", "content": snippet},
        ],
        temperature=0.2,
    )
    return r.choices[0].message.content

print(summarize_url("https://venice.ai/blog/web-scraping-live-on-venice-turn-any-url-into-ai-context"))'''),
        ("markdown",
            "## 3. Text Parser\n\n"
            "Drop in a PDF, DOCX, XLSX, or plain text file (up to 25MB) and get back the extracted "
            "text plus token count. Runs in-memory on Venice infrastructure with zero data "
            "retention. The fallback below builds a tiny PDF on the fly so the cell always has "
            "something to chew on, but you can swap in any local document."),
        ("code",
            '''import io
from pathlib import Path

# Try to use a PDF that already exists in this Colab session, otherwise build a tiny one.
sample = Path("sample.pdf")
if not sample.exists():
    try:
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(str(sample))
        c.drawString(72, 750, "Venice Cookbook Sample")
        c.drawString(72, 730, "This is a sample PDF generated inside the notebook.")
        c.drawString(72, 710, "Venice's Text Parser will extract these lines back as plain text.")
        c.save()
        print("Built sample.pdf")
    except Exception:
        sample = Path("sample.txt")
        sample.write_text("Sample document for Venice Text Parser.\\nLine two for token-count fun.")
        print("Built sample.txt (reportlab not available)")

with open(sample, "rb") as f:
    r = post(
        "/augment/text-parser",
        files={"file": (sample.name, f, "application/pdf" if sample.suffix == ".pdf" else "text/plain")},
        data={"response_format": "json"},
    )
r.raise_for_status()
parsed = r.json()
print("Tokens:", parsed["tokens"])
print()
print(parsed["text"])'''),
        ("markdown",
            "## 4. Crypto Networks (the supported-chains directory)\n\n"
            "Public endpoint, no auth, instant. Use this any time you need to validate a network "
            "slug before sending an RPC call."),
        ("code",
            '''r = get("/crypto/rpc/networks")
networks = r.json()["networks"]

# Show them as a table grouped by chain family
def family(slug: str) -> str:
    return slug.split("-")[0]

net_df = pd.DataFrame({"slug": networks})
net_df["family"]  = net_df["slug"].map(family)
net_df["mainnet"] = net_df["slug"].str.endswith("mainnet")
net_df.sort_values(["family", "slug"]).reset_index(drop=True)'''),
        ("markdown",
            "## 5. Crypto RPC: your one endpoint for every chain\n\n"
            "`POST /api/v1/crypto/rpc/{network}` accepts a standard JSON-RPC 2.0 body (single or "
            "batch). Your Venice key works on every chain in the table above. Two response headers "
            "tell you exactly what each call cost:\n\n"
            "- `X-Venice-RPC-Credits`: credits charged.\n"
            "- `X-Venice-RPC-Cost-USD`: dollar cost to 8 decimal places.\n\n"
            "Standard methods like `eth_call` and `eth_getBalance` cost ~20 credits (~$0.0000125 per "
            "call on most EVM chains). That is ~80,000 calls per dollar."),
        ("code",
            '''def rpc(network: str, method: str, params=None, request_id: int = 1):
    """Single JSON-RPC call. Returns (result_or_error, response_headers).

    We return the raw `requests` headers (CaseInsensitiveDict) so the cost
    headers can be read with any casing the server happens to use.
    """
    body = {"jsonrpc": "2.0", "method": method, "params": params or [], "id": request_id}
    r = post(f"/crypto/rpc/{network}", json_body=body)
    r.raise_for_status()
    return r.json(), r.headers

def rpc_batch(network: str, calls: list[dict]):
    """Batch JSON-RPC. `calls` is a list of {method, params}. Returns (list_of_responses, headers)."""
    body = [
        {"jsonrpc": "2.0", "method": c["method"], "params": c.get("params", []), "id": i + 1}
        for i, c in enumerate(calls)
    ]
    r = post(f"/crypto/rpc/{network}", json_body=body)
    r.raise_for_status()
    return r.json(), r.headers

# Sanity: chain ID and latest block on Base mainnet
chain_id, hdrs = rpc("base-mainnet", "eth_chainId")
block_no, _    = rpc("base-mainnet", "eth_blockNumber")
print(f"Base chain id : {int(chain_id['result'], 16)}  (expected 8453)")
print(f"Latest block  : {int(block_no['result'], 16)}")
print(f"Cost of last call: {hdrs.get('X-Venice-RPC-Cost-USD')} USD ({hdrs.get('X-Venice-RPC-Credits')} credits)")'''),
        ("markdown",
            "### Read your own wallet balance\n\n"
            "Plug in any Ethereum-style address. We will read the native ETH balance on Base, then "
            "use `eth_call` against the canonical USDC contract to read the ERC-20 balance, all in "
            "one batched HTTP call."),
        ("code",
            '''ADDRESS = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"  # vitalik.eth, swap with your own

USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # USDC on Base mainnet

def balance_of_calldata(address: str) -> str:
    """ERC-20 balanceOf(address) calldata: 0x70a08231 + 32-byte padded address."""
    addr = address.lower().replace("0x", "").rjust(64, "0")
    return "0x70a08231" + addr

calls = [
    {"method": "eth_getBalance", "params": [ADDRESS, "latest"]},
    {"method": "eth_call",       "params": [{"to": USDC_BASE, "data": balance_of_calldata(ADDRESS)}, "latest"]},
    {"method": "eth_blockNumber","params": []},
]
batch, headers = rpc_batch("base-mainnet", calls)

eth_wei  = int(batch[0]["result"], 16)
usdc_raw = int(batch[1]["result"], 16) if batch[1].get("result") not in (None, "0x") else 0
block    = int(batch[2]["result"], 16)

print(f"Block            : {block}")
print(f"Native ETH (Base): {eth_wei / 1e18:.6f} ETH")
print(f"USDC (Base)      : {usdc_raw / 1e6:.2f} USDC")
print(f"Batch cost       : {headers.get('X-Venice-RPC-Cost-USD')} USD ({headers.get('X-Venice-RPC-Credits')} credits)")'''),
        ("markdown",
            "### Multichain fan-out: same address, four chains, one Venice key\n\n"
            "Loop the same balance query over Base, Optimism, Arbitrum, and Polygon. Notice that "
            "each chain bills its own credit count via the response headers, but they all share the "
            "same auth and the same billing meter."),
        ("code",
            '''CHAINS = [
    ("base-mainnet",     "ETH"),
    ("optimism-mainnet", "ETH"),
    ("arbitrum-mainnet", "ETH"),
    ("polygon-mainnet",  "MATIC"),
]

rows = []
for slug, native in CHAINS:
    res, hdrs = rpc(slug, "eth_getBalance", [ADDRESS, "latest"])
    wei = int(res["result"], 16)
    rows.append({
        "chain":     slug,
        "balance":   wei / 1e18,
        "symbol":    native,
        "credits":   int(hdrs.get("X-Venice-RPC-Credits") or 0),
        "cost_usd":  float(hdrs.get("X-Venice-RPC-Cost-USD") or 0),
    })

multichain = pd.DataFrame(rows)
print(f"Total spent on these 4 calls: ${multichain['cost_usd'].sum():.8f}")
multichain'''),
        ("markdown",
            "### Idempotent retries\n\n"
            "Set the `Idempotency-Key` header to safely retry a call without paying twice or "
            "double-broadcasting a transaction. Replays return the cached response with "
            "`Idempotent-Replayed: true`."),
        ("code",
            '''import uuid

key = uuid.uuid4().hex
body = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}

first  = post("/crypto/rpc/base-mainnet", json_body=body, extra_headers={"Idempotency-Key": key})
second = post("/crypto/rpc/base-mainnet", json_body=body, extra_headers={"Idempotency-Key": key})

print("First  call: replayed?", first.headers.get("Idempotent-Replayed", "false"),
      "cost", first.headers.get("X-Venice-RPC-Cost-USD"))
print("Second call: replayed?", second.headers.get("Idempotent-Replayed", "false"),
      "cost", second.headers.get("X-Venice-RPC-Cost-USD"))'''),
        ("markdown",
            "## 6. The killer demo: an onchain assistant\n\n"
            "Now compose. We hand the chat model a `crypto_rpc` tool and let it answer plain-English "
            "questions about any wallet on any supported chain. The model picks the network, picks "
            "the method, fills in the params, and we execute the call. This is the foundation of an "
            "agent that can do anything from reading prices off a DEX to estimating gas before "
            "sending a transaction."),
        ("code",
            '''ONCHAIN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "crypto_rpc",
            "description": (
                "Make a JSON-RPC call to any supported blockchain via Venice. "
                "Use this to read on-chain data: balances, block numbers, contract storage, etc. "
                "Always pick the smallest possible method (eth_getBalance, eth_blockNumber, eth_chainId, eth_call) "
                "and never invent unsupported methods."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "network": {
                        "type": "string",
                        "description": "Network slug, e.g. base-mainnet, ethereum-mainnet, arbitrum-mainnet.",
                    },
                    "method": {
                        "type": "string",
                        "description": "JSON-RPC method, e.g. eth_getBalance, eth_blockNumber, eth_call.",
                    },
                    "params": {
                        "type": "array",
                        "description": "JSON-RPC params array. Empty list when not needed.",
                        "items": {},
                    },
                },
                "required": ["network", "method", "params"],
            },
        },
    }
]

def run_onchain_agent(user_msg: str, model: str = "kimi-k2-6") -> str:
    messages = [
        {"role": "system", "content": (
            "You are an on-chain assistant. When the user asks about balances, blocks, contracts, or transactions, "
            "call the crypto_rpc tool. Convert wei to ETH (divide by 1e18) and USDC raw to USDC (divide by 1e6) "
            "before reporting. Network slugs you can use: base-mainnet, ethereum-mainnet, arbitrum-mainnet, "
            "optimism-mainnet, polygon-mainnet, base-sepolia."
        )},
        {"role": "user", "content": user_msg},
    ]

    for _ in range(4):
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=ONCHAIN_TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        msg = r.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            try:
                result, hdrs = rpc(args["network"], args["method"], args.get("params", []))
                content = json.dumps({
                    "result":  result.get("result"),
                    "error":   result.get("error"),
                    "credits": hdrs.get("X-Venice-RPC-Credits"),
                    "cost":    hdrs.get("X-Venice-RPC-Cost-USD"),
                })
            except Exception as e:
                content = json.dumps({"error": str(e)})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    return "(agent gave up after 4 turns)"

print(run_onchain_agent(f"What is the ETH balance of {ADDRESS} on Base mainnet, and what block was it read at?"))'''),
        ("code",
            '''print(run_onchain_agent("What is the latest block number on Arbitrum and Optimism right now? Compare them."))'''),
        ("markdown",
            "## Recap\n\n"
            "Five tools, one base URL, one bill:\n\n"
            "| Tool | Endpoint | What it replaces |\n"
            "|---|---|---|\n"
            "| Web Search | `/augment/search` | Brave / Google API accounts |\n"
            "| Web Scrape | `/augment/scrape` | Firecrawl / ScrapingBee |\n"
            "| Text Parser | `/augment/text-parser` | Unstructured / Tika / a custom PDF stack |\n"
            "| Crypto Networks | `/crypto/rpc/networks` | Maintaining your own chain registry |\n"
            "| Crypto RPC | `/crypto/rpc/{network}` | Alchemy + Infura + dRPC + QuickNode + 7 others |\n\n"
            "If you only remember one thing from this notebook: Crypto RPC means your agent never "
            "needs a separate RPC provider account again. One Venice key, every chain, billed in "
            "Venice credits or paid via x402 from the agent's own wallet."),
    ]
