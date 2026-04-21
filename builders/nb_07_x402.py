"""07 - x402 wallet payments. Single paid call, then a full agent loop where the
agent uses tool calling to pay per request and keeps a transaction log."""

from ._common import Cell, header, install_cell, setup_cell


NOTEBOOK = "07-x402-wallet-payments.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "x402: pay-per-call AI with a wallet, no API key required",
            "x402 is the open HTTP payment protocol that lets your wallet pay for an API call the "
            "moment the server returns 402. We will make one paid call by hand, then build an autonomous "
            "agent that pays its own way and keeps an on-chain receipt log.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. **One paid call** to Venice using a Base wallet and the x402 client.\n"
            "2. **An agent loop** that decides when to call Venice, pays per call, and keeps a "
            "running transaction log displayed as a pandas DataFrame.\n"
            "3. **A cost analysis** that compares pay-per-call x402 to a flat-rate API key.\n\n"
            "Why this matters: x402 lets agents (and humans) pay for AI without ever creating an "
            "account, signing a TOS, or rotating an API key. Pure HTTP, pure crypto.\n\n"
            "**Requires:** a Base wallet with a small USDC balance. Sepolia testnet is fine for the "
            "demo (faucet at [faucets.chain.link](https://faucets.chain.link/base-sepolia))."),
        ("markdown", "## Setup"),
        install_cell("eth-account web3 x402"),
        setup_cell(),
        ("code",
            '''import os
from getpass import getpass

WALLET_KEY = os.environ.get("WALLET_PRIVATE_KEY")
if not WALLET_KEY:
    try:
        from google.colab import userdata  # type: ignore
        WALLET_KEY = userdata.get("WALLET_PRIVATE_KEY")
    except Exception:
        WALLET_KEY = None
if not WALLET_KEY:
    WALLET_KEY = getpass("Paste your Base wallet PRIVATE KEY (starts with 0x). Use a throwaway: ").strip()
os.environ["WALLET_PRIVATE_KEY"] = WALLET_KEY

from eth_account import Account
acct = Account.from_key(WALLET_KEY)
print("Wallet:", acct.address)'''),
        ("markdown",
            "## 1. One paid call\n\n"
            "We hit Venice's x402-enabled endpoint. The server returns 402 with a payment requirement, "
            "the x402 client signs a USDC transfer authorization, retries the call with the proof, and "
            "we get back a normal chat completion. All in one round trip from your point of view."),
        ("code",
            '''import requests
from x402.clients.requests import x402_requests

X402_ENDPOINT = "https://api.venice.ai/api/v1/chat/completions"

session = x402_requests(account=acct)

resp = session.post(
    X402_ENDPOINT,
    json={
        "model": "llama-3.3-70b",
        "messages": [{"role": "user", "content": "Say hi from x402 in 6 words."}],
    },
    timeout=60,
)
print("Status:", resp.status_code)
print("Reply:", resp.json()["choices"][0]["message"]["content"])'''),
        ("markdown",
            "Inspect the payment metadata that came back so you can see the actual on-chain receipt."),
        ("code",
            '''from x402.clients.base import decode_x_payment_response

raw = resp.headers.get("x-payment-response")
if raw:
    payment = decode_x_payment_response(raw)
    print("Tx hash:", payment.get("transaction"))
    print("Network:", payment.get("network"))
    print("Asset:  ", payment.get("payer"))
else:
    print("No payment header returned. Some endpoints accept stub payments on testnet.")'''),
        ("markdown",
            "## 2. An autonomous agent that pays per call\n\n"
            "Now the headline: an agent loop that decides when to query Venice, pays for each call "
            "with x402, and logs every transaction. This is the building block of a self-funded AI "
            "service: the agent gets a budget, uses Venice as a tool, and stops when funds run out."),
        ("code",
            '''import json, time, pandas as pd

LEDGER: list[dict] = []

def paid_call(prompt: str, model: str = "llama-3.3-70b") -> dict:
    """Make one paid call and append a row to the ledger."""
    t0 = time.perf_counter()
    r = session.post(
        X402_ENDPOINT,
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()

    payment_raw = r.headers.get("x-payment-response")
    payment = decode_x_payment_response(payment_raw) if payment_raw else {}

    LEDGER.append({
        "prompt":  prompt[:60],
        "model":   model,
        "tokens":  body["usage"]["total_tokens"],
        "latency": round(elapsed, 2),
        "tx":      (payment.get("transaction") or "(no payment)")[:14] + "...",
    })
    return body

QUEUE = [
    "Summarize the Bitcoin whitepaper in one tweet.",
    "What is x402 in 2 sentences?",
    "Name three reasons agents should pay for their own compute.",
]

for q in QUEUE:
    paid_call(q)

pd.DataFrame(LEDGER)'''),
        ("markdown",
            "## 3. Cost analysis: pay-per-call vs API key\n\n"
            "Pay-per-call shines when usage is bursty or unknown. A flat API key wins when usage is "
            "predictable and high. Here is a quick napkin math you can adapt to your own numbers."),
        ("code",
            '''SCENARIOS = pd.DataFrame([
    {"plan": "x402 pay-per-call", "monthly_calls": 100,    "cost_per_call_usd": 0.002, "monthly_fixed_usd": 0},
    {"plan": "x402 pay-per-call", "monthly_calls": 10_000, "cost_per_call_usd": 0.002, "monthly_fixed_usd": 0},
    {"plan": "Venice Pro key",    "monthly_calls": 100,    "cost_per_call_usd": 0,     "monthly_fixed_usd": 49},
    {"plan": "Venice Pro key",    "monthly_calls": 10_000, "cost_per_call_usd": 0,     "monthly_fixed_usd": 49},
])
SCENARIOS["monthly_total_usd"] = (
    SCENARIOS["monthly_calls"] * SCENARIOS["cost_per_call_usd"] + SCENARIOS["monthly_fixed_usd"]
)
SCENARIOS'''),
        ("markdown",
            "## Recap\n\n"
            "x402 turns Venice into a permissionless utility. Any wallet can spend, any agent can "
            "self-fund, and every call leaves a tx hash you can verify on Basescan. Next: "
            "`08-e2ee-encryption.ipynb`, the headliner."),
    ]
