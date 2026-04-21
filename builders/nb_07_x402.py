"""07 - x402 wallet payments. Sign-In-With-X (SIWE), balance check, paid call, top-up flow.

Self-contained: this notebook does NOT require a Venice API key. The whole point of
x402 is wallet auth.
"""

from ._common import Cell, header, install_cell


NOTEBOOK = "07-x402-wallet-payments.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "x402: pay for AI with a wallet, no API key required",
            "Venice implements the [x402](https://www.x402.org/) wallet auth standard end to end. "
            "Sign a SIWE message with any Ethereum wallet, top up with USDC on Base (or stake DIEM), "
            "then call any inference endpoint. We will build the full flow in pure Python and "
            "inspect every header.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. **Sign a SIWE message** and pack it into the `X-Sign-In-With-X` header.\n"
            "2. **Check the balance** of any wallet address via `/x402/balance/{address}`.\n"
            "3. **Send a paid request** with just the SIWE header (no API key).\n"
            "4. **Inspect the 402 response** when the wallet is empty, to see Venice's payment "
            "requirements (USDC token address, recipient, network, amount).\n"
            "5. **Read the transaction ledger** via `/x402/transactions/{address}`.\n"
            "6. **Compare access models**: x402 vs Pro plan vs DIEM staking.\n\n"
            "**No Venice API key needed.** This whole notebook authenticates with a wallet.\n\n"
            "**Wallet:** if `WALLET_PRIVATE_KEY` is in your env we will use it. Otherwise we "
            "generate a throwaway wallet so you can see the full protocol without funding "
            "anything. Top-ups require USDC on Base (chain id 8453, min $5)."),
        ("markdown", "## Setup"),
        install_cell("eth-account siwe"),
        ("code",
            '''import os, json, base64, secrets
from datetime import datetime, timedelta, timezone
import requests
from eth_account import Account
from eth_account.messages import encode_defunct
from siwe import SiweMessage

API = "https://api.venice.ai/api/v1"

# Use your real wallet if WALLET_PRIVATE_KEY is set, otherwise generate a throwaway.
# Throwaway wallets see the full protocol but cannot top up (no USDC).
key = os.environ.get("WALLET_PRIVATE_KEY")
if not key:
    try:
        from google.colab import userdata  # type: ignore
        key = userdata.get("WALLET_PRIVATE_KEY")
    except Exception:
        pass

if key:
    acct = Account.from_key(key)
    print(f"Using your wallet: {acct.address}")
else:
    acct = Account.create()
    print(f"Generated throwaway wallet: {acct.address}")
    print("(Set WALLET_PRIVATE_KEY in your env or Colab secrets to use a funded one.)")'''),
        ("markdown",
            "## 1. Build the X-Sign-In-With-X header\n\n"
            "Venice expects a base64-encoded JSON object with five fields: `address`, `message` "
            "(an EIP-4361 SIWE string), `signature`, `timestamp`, and `chainId: 8453` for Base. "
            "The SIWE message itself includes a fresh nonce and a 5-minute expiry so each header "
            "is a single-use credential. Source of truth: [Venice x402 guide](https://docs.venice.ai/overview/guides/x402-venice-api)."),
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
    message = siwe.prepare_message()
    signed  = account.sign_message(encode_defunct(text=message))
    sig_hex = signed.signature.hex()
    if not sig_hex.startswith("0x"):
        sig_hex = "0x" + sig_hex
    payload = {
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
            "Inspect the underlying SIWE message so you can see exactly what was signed:"),
        ("code",
            '''import textwrap
decoded = json.loads(base64.b64decode(token))
print("--- SIWE message (signed) ---")
print(decoded["message"])
print()
print("--- Signature (first 20 bytes) ---")
print(decoded["signature"][:42] + "...")'''),
        ("markdown",
            "## 2. Check the wallet's spendable balance\n\n"
            "`GET /x402/balance/{address}` is free, just needs the SIWE header. Useful before "
            "every paid call so you can surface a top-up prompt to your user instead of hitting a "
            "402. The response shows `canConsume` (is the wallet ready to pay?), `balanceUsd` "
            "(USDC top-up balance), and `diemBalanceUsd` (DIEM-backed credits if the wallet is "
            "linked to a Venice account)."),
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
print(f"DIEM balance:   ${data.get('diemBalanceUsd', 0):.4f}")
print(f"Min top-up:     ${data.get('minimumTopUpUsd', 0)}")
print(f"Suggested:      ${data.get('suggestedTopUpUsd', 0)}")'''),
        ("markdown",
            "## 3. Try a paid request\n\n"
            "Same `/chat/completions` endpoint as everywhere else. Just swap `Authorization: "
            "Bearer` for `X-Sign-In-With-X`. If the wallet has DIEM or topped-up USDC the call "
            "returns 200 like any other inference call. If the wallet is empty Venice returns "
            "**402 Payment Required**, which is your cue to run the top-up flow in step 4."),
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
    print(json.dumps(body, indent=2)[:1200])'''),
        ("markdown",
            "## 4. The top-up flow\n\n"
            "`POST /x402/top-up` without a payment header returns **402 Payment Required** with "
            "the canonical x402 payment requirements. The `accepts` array tells you the network, "
            "the USDC contract, the recipient, and the minimum amount."),
        ("code",
            '''r = requests.post(f"{API}/x402/top-up", timeout=30)
print("Status:", r.status_code)
print(json.dumps(r.json(), indent=2)[:1500])'''),
        ("markdown",
            "To actually fund the wallet you sign a USDC `transferWithAuthorization` on Base "
            "using the [`x402` library](https://github.com/coinbase/x402) and resubmit the same "
            "request with an `X-402-Payment` header. As of today the official Venice helper SDK "
            "is [`venice-x402-client`](https://github.com/veniceai/x402-client) (TypeScript / "
            "Node.js):\n\n"
            "```typescript\n"
            "// npm install venice-x402-client\n"
            "import { VeniceClient } from 'venice-x402-client'\n\n"
            "const venice = new VeniceClient(process.env.WALLET_KEY)\n"
            "await venice.topUp(10) // $10 USDC on Base, signs and submits in one call\n"
            "const r = await venice.chat({\n"
            "  model: 'kimi-k2-5',\n"
            "  messages: [{ role: 'user', content: 'Hello!' }],\n"
            "})\n"
            "```\n\n"
            "From Python you can do the equivalent today by signing the EIP-3009 "
            "`transferWithAuthorization` with `eth-account` and assembling the `X-402-Payment` "
            "header by hand. The 402 response above gives you every value you need (`payTo`, "
            "`asset`, `network`, `amount`). We skip the actual on-chain submission here so we do "
            "not move real money during the workshop."),
        ("markdown",
            "## 5. Read the transaction ledger\n\n"
            "Once the wallet has activity, `GET /x402/transactions/{address}` returns a paged "
            "ledger of `TOP_UP`, `CHARGE`, and `REFUND` entries. Same SIWE auth as everything "
            "else. Useful for showing per-call billing in your own UI."),
        ("code",
            '''ledger_token = build_siwx_header(
    acct,
    resource=f"{API}/x402/transactions/{acct.address}",
)
r = requests.get(
    f"{API}/x402/transactions/{acct.address}",
    headers={"X-Sign-In-With-X": ledger_token},
    params={"limit": 10, "offset": 0},
    timeout=30,
)
print("Status:", r.status_code)
print(json.dumps(r.json(), indent=2)[:1200])'''),
        ("markdown",
            "## 6. Three ways to pay Venice\n\n"
            "Pick the one that matches your usage shape. None of these are mutually exclusive: "
            "Pro subscribers can also stake DIEM, and any wallet can use x402 even without a "
            "Venice account."),
        ("code",
            '''import pandas as pd

pd.DataFrame([
    {
        "option":   "x402 pay-per-call",
        "auth":     "Wallet signature (SIWE)",
        "billing":  "Per request, against USDC top-up",
        "minimum":  "$5 top-up",
        "best for": "Agents, one-off calls, no-account use, on-chain apps",
    },
    {
        "option":   "Venice Pro subscription",
        "auth":     "API key (Bearer token)",
        "billing":  "$18/mo + pay-as-you-go API credits ($10 one-time API credit on upgrade)",
        "minimum":  "$18 + $10 USD top-up minimum",
        "best for": "Humans using both the web app and the API",
    },
    {
        "option":   "Stake DIEM",
        "auth":     "API key OR wallet (DIEM is spent first if linked)",
        "billing":  "Each DIEM = $1/day of API capacity, refreshes daily",
        "minimum":  "1 DIEM (mint by locking sVVV)",
        "best for": "Predictable daily usage, capacity you want to own and trade",
    },
])'''),
        ("markdown",
            "Pricing of the underlying inference is the same across all three: see "
            "[docs.venice.ai/overview/pricing](https://docs.venice.ai/overview/pricing) for the "
            "per-million-token rates. x402 just changes how you pay, not what you pay."),
        ("markdown",
            "## Recap\n\n"
            "x402 turns Venice into a permissionless utility:\n\n"
            "- **One header**, `X-Sign-In-With-X`, replaces the API key.\n"
            "- **Three free, wallet-only routes** (`balance`, `transactions`, the initial "
            "`top-up` 402 response) let you check state and discover payment requirements without "
            "spending anything.\n"
            "- **Top up with USDC on Base** (min $5) using the official `x402` library, or stake "
            "DIEM for daily credits that refresh automatically.\n"
            "- **`X-Balance-Remaining`** comes back with every paid response so agents can "
            "self-throttle.\n\n"
            "Combine this with notebook 09's Crypto RPC and you have an agent that can read any "
            "chain, pay for its own AI, and never touch an API key.\n\n"
            "Reference: [Venice x402 guide](https://docs.venice.ai/overview/guides/x402-venice-api), "
            "[x402.org](https://www.x402.org/)."),
    ]
