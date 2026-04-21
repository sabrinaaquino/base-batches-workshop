"""07 - x402 wallet payments. Sign-In-With-X (SIWE), balance check, paid call, top-up flow."""

from ._common import Cell, header, install_cell, setup_cell


NOTEBOOK = "07-x402-wallet-payments.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "x402: pay for AI with a wallet, no API key required",
            "Venice supports the x402 wallet auth standard end to end. Sign a SIWE message with any "
            "Ethereum wallet, top up with USDC on Base, then call any inference endpoint with no "
            "API key. We will build the full flow in pure Python and inspect every header.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. **Sign a SIWE message** and pack it into the `X-Sign-In-With-X` header.\n"
            "2. **Check the balance** of any wallet address via `/x402/balance`.\n"
            "3. **Send a paid request** with just the SIWE header (no API key).\n"
            "4. **Inspect the 402 response** when the wallet is empty, to see Venice's payment "
            "requirements (USDC token address, recipient, network, amount).\n"
            "5. **Compare costs** between pay-per-call x402 and a flat API key.\n\n"
            "**Wallet:** if `WALLET_PRIVATE_KEY` is in your env we will use it. Otherwise we "
            "generate a throwaway wallet so you can see the full protocol without funding anything. "
            "Top-ups require USDC on Base (chain id 8453, min $5)."),
        ("markdown", "## Setup"),
        install_cell("eth-account siwe"),
        setup_cell(),
        ("code",
            '''import os, json, base64, secrets, time
from datetime import datetime, timedelta, timezone
import requests
from eth_account import Account
from eth_account.messages import encode_defunct
from siwe import SiweMessage

API = "https://api.venice.ai/api/v1"

# Use your real wallet if you have one set as WALLET_PRIVATE_KEY, otherwise generate a throwaway.
key = os.environ.get("WALLET_PRIVATE_KEY")
if not key:
    acct = Account.create()
    key  = acct.key.hex()
    print(f"Generated throwaway wallet: {acct.address}")
    print("(Set WALLET_PRIVATE_KEY in your env to use a funded one.)")
else:
    acct = Account.from_key(key)
    print(f"Using your wallet: {acct.address}")'''),
        ("markdown",
            "## 1. Build the X-Sign-In-With-X header\n\n"
            "Venice expects a base64-encoded JSON object with five fields: `address`, `message` "
            "(an EIP-4361 SIWE string), `signature`, `timestamp`, and `chainId: 8453` for Base. "
            "The SIWE message itself includes a fresh nonce and a 5-minute expiry so each header "
            "is a single-use credential."),
        ("code",
            '''def build_siwx_header(account, *, resource: str = f"{API}/chat/completions") -> str:
    now = datetime.now(timezone.utc)
    siwe = SiweMessage(
        domain="api.venice.ai",
        address=account.address,
        statement="Sign in to Venice AI",
        uri=resource,
        version="1",
        chain_id=8453,
        nonce=secrets.token_hex(8),
        issued_at=now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        expiration_time=(now + timedelta(minutes=5)).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )
    message   = siwe.prepare_message()
    signed    = account.sign_message(encode_defunct(text=message))
    sig_hex   = signed.signature.hex()
    if not sig_hex.startswith("0x"):
        sig_hex = "0x" + sig_hex
    payload   = {
        "address":   account.address,
        "message":   message,
        "signature": sig_hex,
        "timestamp": int(now.timestamp() * 1000),
        "chainId":   8453,
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()

token = build_siwx_header(acct)
print(f"X-Sign-In-With-X: {token[:60]}... ({len(token)} chars)")'''),
        ("markdown",
            "## 2. Check the wallet's spendable balance\n\n"
            "`/x402/balance/{address}` is free, just needs the SIWE header. Useful before every "
            "paid call so you can surface a top-up prompt to your user instead of hitting a 402."),
        ("code",
            '''r = requests.get(
    f"{API}/x402/balance/{acct.address}",
    headers={"X-Sign-In-With-X": token},
    timeout=30,
)
print("Status:", r.status_code)
body = r.json()
print(json.dumps(body, indent=2))
print()
data = body.get("data", body)
print(f"Can consume:    {data.get('canConsume')}")
print(f"USDC balance:   ${data.get('balanceUsd', 0):.4f}")
print(f"Min top-up:     ${data.get('minimumTopUpUsd', 0)}")
print(f"Suggested:      ${data.get('suggestedTopUpUsd', 0)}")'''),
        ("markdown",
            "## 3. Try a paid request (with no balance, expect 402)\n\n"
            "Same `/chat/completions` endpoint as everywhere else. Just swap `Authorization: Bearer` "
            "for `X-Sign-In-With-X`. If the wallet has DIEM or topped-up USDC the call returns "
            "200 like any other inference call. If the wallet is empty Venice returns **402 Payment "
            "Required** with the exact USDC payment requirements you can sign and submit to "
            "`/x402/top-up`."),
        ("code",
            '''paid_token = build_siwx_header(acct)  # fresh nonce per request
r = requests.post(
    f"{API}/chat/completions",
    headers={
        "X-Sign-In-With-X": paid_token,
        "Content-Type":     "application/json",
    },
    json={
        "model":    "kimi-k2-5",
        "messages": [{"role": "user", "content": "Hi from an x402 wallet, in 8 words."}],
    },
    timeout=60,
)
print("Status:", r.status_code)
remaining = r.headers.get("X-Balance-Remaining")
if remaining is not None:
    print(f"Balance remaining after this call: ${float(remaining):.4f}")
print()

if r.status_code == 200:
    print("Response:", r.json()["choices"][0]["message"]["content"])
else:
    body = r.json()
    print(json.dumps(body, indent=2)[:1000])'''),
        ("markdown",
            "## 4. Inspect the top-up flow\n\n"
            "`POST /x402/top-up` without a payment header returns 402 with the canonical x402 "
            "payment requirements. The `accepts` array tells you the network, the USDC contract, "
            "the recipient, and the minimum amount. You feed those to the official `x402` SDK to "
            "build a signed `X-402-Payment` header and re-submit."),
        ("code",
            '''r = requests.post(f"{API}/x402/top-up", timeout=30)
print("Status:", r.status_code)
print(json.dumps(r.json(), indent=2)[:1200])'''),
        ("markdown",
            "Doing the actual top-up requires signing a USDC `transferWithAuthorization` on Base. "
            "The Coinbase `x402` Python SDK handles this for you. Skipped here so we do not move "
            "real money during the workshop. The four lines you would run are:\n\n"
            "```python\n"
            "# pip install x402\n"
            "from x402.clients.requests import x402_requests\n"
            "session = x402_requests(account=acct)\n"
            "session.post(f'{API}/x402/top-up', json={'amount_usd': 5})\n"
            "```"),
        ("markdown",
            "## 5. Cost analysis: pay-per-call vs API key vs DIEM\n\n"
            "Pay-per-call x402 wins when usage is bursty, agent-driven, or you do not want an "
            "account. API keys win at predictable high volume. DIEM wins for stable monthly spend."),
        ("code",
            '''import pandas as pd

scenarios = pd.DataFrame([
    {"plan": "x402 pay-per-call",  "monthly_calls": 100,    "cost_per_call_usd": 0.002, "monthly_fixed_usd": 0},
    {"plan": "x402 pay-per-call",  "monthly_calls": 10_000, "cost_per_call_usd": 0.002, "monthly_fixed_usd": 0},
    {"plan": "Venice Pro API key", "monthly_calls": 100,    "cost_per_call_usd": 0,     "monthly_fixed_usd": 49},
    {"plan": "Venice Pro API key", "monthly_calls": 10_000, "cost_per_call_usd": 0,     "monthly_fixed_usd": 49},
    {"plan": "DIEM (1 token = $1/day)", "monthly_calls": 10_000, "cost_per_call_usd": 0, "monthly_fixed_usd": 30},
])
scenarios["monthly_total_usd"] = (
    scenarios["monthly_calls"] * scenarios["cost_per_call_usd"] + scenarios["monthly_fixed_usd"]
)
scenarios.sort_values(["monthly_calls", "monthly_total_usd"]).reset_index(drop=True)'''),
        ("markdown",
            "## Recap\n\n"
            "x402 turns Venice into a permissionless utility:\n\n"
            "- **One header**, `X-Sign-In-With-X`, replaces the API key.\n"
            "- **One endpoint**, `/x402/top-up`, hands you the payment requirements; the official "
            "`x402` SDK signs USDC on Base and submits them back.\n"
            "- Every paid response returns `X-Balance-Remaining` so agents can self-throttle.\n"
            "- DIEM staking gives you daily credits (1 DIEM = $1/day) instead of pay-per-call.\n\n"
            "Combine this with notebook 09's Crypto RPC and you have an agent that can read any "
            "chain and pay for its own AI without ever asking a human."),
    ]
