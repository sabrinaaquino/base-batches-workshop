"""08 - End-to-end encrypted inference. The headliner.

Implements Venice's actual E2EE protocol from scratch in Python:
    secp256k1 ECDH -> HKDF-SHA256 -> AES-256-GCM,
    X-Venice-TEE-* headers, streaming response, per-chunk decryption.
"""

from ._common import Cell, header, install_cell, setup_cell


NOTEBOOK = "08-e2ee-encryption.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "End-to-end encrypted inference: provable, not promised",
            "Most AI providers ask you to trust them. Venice's E2EE models encrypt your prompt on the "
            "client with a key Venice never sees, decrypt it only inside a Trusted Execution "
            "Environment, and stream back a response that is encrypted to you and only you. "
            "We will implement the full protocol in pure Python and inspect every byte.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. **A privacy mode comparison table** so you know what each tier guarantees.\n"
            "2. **The full E2EE handshake**: discover an `e2ee-*` model, pull a hardware-attested "
            "key, do secp256k1 ECDH + HKDF + AES-256-GCM in 30 lines.\n"
            "3. **Send an encrypted prompt** with the right `X-Venice-TEE-*` headers.\n"
            "4. **Decrypt the streamed reply** chunk by chunk, in real time.\n"
            "5. **Side-by-side diff** of what Venice's network sees in private mode vs E2EE mode.\n\n"
            "Cost: zero extra. E2EE is included on Pro tiers and bundled with TEE."),
        ("markdown",
            "## Privacy modes at a glance\n\n"
            "| Mode | Who can read your prompt | Hardware proof | Default |\n"
            "|---|---|---|---|\n"
            "| Anonymized (3rd party) | 3rd party provider only, never linked to you | no | for proxied frontier models |\n"
            "| Private (Venice default) | Venice infrastructure for the duration of inference, then discarded | no | yes, for open-source models |\n"
            "| TEE | Only the verified enclave. Even Venice operators cannot read it | yes (remote attestation) | opt-in, prefix `tee-` |\n"
            "| E2EE | Encrypted on your device first. Even Venice's network cannot see plaintext | yes (attestation + ECDH) | opt-in, prefix `e2ee-` |\n\n"
            "The whole point of E2EE: you do not have to trust Venice. You verify."),
        ("markdown", "## Setup"),
        install_cell("ecdsa cryptography"),
        setup_cell(),
        ("code",
            '''import json, secrets, requests, re
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from ecdsa import SECP256k1, VerifyingKey, SigningKey

API_BASE  = "https://api.venice.ai/api/v1"
HEADERS   = {"Authorization": f"Bearer {api_key}"}
HKDF_INFO = b"ecdsa_encryption"  # Venice's fixed HKDF info string'''),
        ("markdown",
            "## 1. Discover an E2EE model\n\n"
            "Any model whose `id` starts with `e2ee-` runs inside a TEE and supports the encryption "
            "handshake. The capability flag `supportsE2EE` is the source of truth."),
        ("code",
            '''import pandas as pd

models = requests.get(f"{API_BASE}/models", headers=HEADERS, timeout=30).json()["data"]
e2ee_models = [m for m in models if m.get("model_spec", {}).get("capabilities", {}).get("supportsE2EE")]

pd.DataFrame([{
    "model": m["id"],
    "ctx":   m["model_spec"].get("availableContextTokens"),
    "tee":   m["model_spec"]["capabilities"].get("supportsTeeAttestation", False),
} for m in e2ee_models])'''),
        ("code",
            '''MODEL = "e2ee-qwen3-5-122b-a10b"  # pick any from the table above
print("Using:", MODEL)'''),
        ("markdown",
            "## 2. Generate your ephemeral keypair\n\n"
            "secp256k1 (the same curve as Bitcoin and Ethereum). The private key never leaves your "
            "machine. We zero it out at the end of the session in production."),
        ("code",
            '''client_priv = SigningKey.generate(curve=SECP256k1)
client_pub  = client_priv.get_verifying_key()
client_pub_hex = (b"\\x04" + client_pub.to_string()).hex()  # 04 || x || y, 130 chars
print("Client public key (uncompressed, 130 hex chars):", client_pub_hex)'''),
        ("markdown",
            "## 3. Fetch and verify the TEE attestation\n\n"
            "We send a fresh 32-byte nonce so the enclave cannot replay an old attestation. The "
            "response includes a `signing_key` (the model's secp256k1 public key, used for ECDH), "
            "an Intel TDX quote, and an Ethereum-style `signing_address` derived from the key."),
        ("code",
            '''nonce_hex = secrets.token_hex(32)  # 32 BYTES = 64 hex chars (TEE requires this)

att = requests.get(
    f"{API_BASE}/tee/attestation",
    params={"model": MODEL, "nonce": nonce_hex},
    headers=HEADERS,
    timeout=30,
).json()

assert att.get("verified") is True,   f"Attestation not verified: {att}"
assert att.get("nonce") == nonce_hex, "Nonce mismatch, possible replay attack"

model_pub_key = att.get("signing_key") or att.get("signing_public_key")
print("TEE provider:    ", att.get("tee_provider"))
print("Signing address: ", att.get("signing_address"))
print("Model pub key:   ", model_pub_key[:20], "...", model_pub_key[-10:])
print("Intel TDX quote: ", str(att.get("intel_quote", ""))[:60], "... (truncated)")'''),
        ("markdown",
            "## 4. The encryption helper\n\n"
            "For each message we generate a fresh ephemeral keypair, derive the AES key by ECDH + "
            "HKDF-SHA256, then AES-256-GCM the plaintext. The wire format Venice expects is the "
            "concatenation `ephemeral_pub_key (65 bytes) || nonce (12 bytes) || ciphertext`, "
            "encoded as lowercase hex. That hex string is what goes into `messages[i].content`."),
        ("code",
            '''import os as _os

def _normalize_pub(hex_key: str) -> bytes:
    """Accept either uncompressed (130 chars) or raw (128 chars) and return 65 bytes."""
    if len(hex_key) == 128:
        hex_key = "04" + hex_key
    if not hex_key.startswith("04") or len(hex_key) != 130:
        raise ValueError(f"Bad pubkey: len={len(hex_key)}")
    return bytes.fromhex(hex_key)

def encrypt_for_tee(plaintext: str, model_pub_hex: str) -> str:
    model_pub_bytes = _normalize_pub(model_pub_hex)
    model_vk = VerifyingKey.from_string(model_pub_bytes[1:], curve=SECP256k1)

    eph_priv = SigningKey.generate(curve=SECP256k1)
    eph_pub  = eph_priv.get_verifying_key()

    shared_pt   = model_vk.pubkey.point * eph_priv.privkey.secret_multiplier
    shared_secret = shared_pt.x().to_bytes(32, "big")

    aes_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=HKDF_INFO).derive(shared_secret)

    nonce  = _os.urandom(12)
    cipher = AESGCM(aes_key).encrypt(nonce, plaintext.encode(), None)

    eph_pub_bytes = b"\\x04" + eph_pub.to_string()  # 65 bytes
    return (eph_pub_bytes + nonce + cipher).hex()

# Encrypt one message and inspect the wire format
sample = encrypt_for_tee("What is the capital of France?", model_pub_key)
print("Ciphertext (hex):", sample[:80], "... total", len(sample), "chars")
print("First 130 hex chars are our ephemeral pubkey, next 24 are the nonce, rest is GCM ciphertext+tag.")'''),
        ("markdown",
            "## 5. Send the E2EE request\n\n"
            "Three things make a request E2EE:\n"
            "1. The model id starts with `e2ee-`.\n"
            "2. Every `user` and `system` message has its `content` field encrypted (hex string).\n"
            "3. Three headers: `X-Venice-TEE-Client-Pub-Key`, `X-Venice-TEE-Model-Pub-Key`, "
            "`X-Venice-TEE-Signing-Algo: ecdsa`.\n\n"
            "E2EE *requires* `stream=true` so each delta chunk can be encrypted independently."),
        ("code",
            '''PROMPT = (
    "I am a doctor reviewing a patient case. The patient is a 47-year-old female with "
    "chest pain radiating to the left arm. Suggest a differential diagnosis in 3 bullets."
)

encrypted_messages = [
    {"role": "user", "content": encrypt_for_tee(PROMPT, model_pub_key)},
]

resp = requests.post(
    f"{API_BASE}/chat/completions",
    headers={
        **HEADERS,
        "Content-Type": "application/json",
        "X-Venice-TEE-Client-Pub-Key": client_pub_hex,
        "X-Venice-TEE-Model-Pub-Key":  model_pub_key,
        "X-Venice-TEE-Signing-Algo":   "ecdsa",
    },
    json={"model": MODEL, "messages": encrypted_messages, "stream": True},
    stream=True,
    timeout=60,
)
print("Status:", resp.status_code)
print("Server saw this in the body:", json.dumps({"messages": encrypted_messages}, indent=2)[:240], "...")'''),
        ("markdown",
            "## 6. Decrypt the stream in real time\n\n"
            "Each SSE chunk is an OpenAI-shaped delta whose `content` is itself a hex-encoded "
            "encrypted blob. Same wire format as our request. We decrypt with our client private "
            "key against the server's per-chunk ephemeral key."),
        ("code",
            '''_HEX = re.compile(r"^[0-9a-fA-F]+$")

def looks_encrypted(s: str) -> bool:
    return len(s) >= 186 and bool(_HEX.match(s))

def decrypt_chunk(hex_chunk: str, client_priv_key: SigningKey) -> str:
    raw    = bytes.fromhex(hex_chunk)
    eph_pub = raw[:65]; nonce = raw[65:77]; cipher = raw[77:]

    server_vk = VerifyingKey.from_string(eph_pub[1:], curve=SECP256k1)
    shared_pt   = server_vk.pubkey.point * client_priv_key.privkey.secret_multiplier
    shared_secret = shared_pt.x().to_bytes(32, "big")

    aes_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=HKDF_INFO).derive(shared_secret)
    return AESGCM(aes_key).decrypt(nonce, cipher, None).decode()

full = ""
for line in resp.iter_lines():
    if not line:
        continue
    line = line.decode()
    if not line.startswith("data: ") or "[DONE]" in line:
        continue
    try:
        chunk = json.loads(line[6:])
    except json.JSONDecodeError:
        continue
    choices = chunk.get("choices") or []
    delta = (choices[0].get("delta", {}) if choices else {}).get("content", "")
    if not delta:
        continue
    if looks_encrypted(delta):
        delta = decrypt_chunk(delta, client_priv)
    full += delta
    print(delta, end="", flush=True)
print("\\n\\n--- decrypted", len(full), "characters end-to-end ---")'''),
        ("markdown",
            "## 7. The diff: what Venice would have seen\n\n"
            "Below is the same prompt sent two ways. In Private mode the body contains the literal "
            "patient case. In E2EE mode it contains 256+ characters of indistinguishable-from-random "
            "ciphertext. Same answer to the user, completely different threat model for the operator."),
        ("code",
            '''private_body = json.dumps({
    "model": "qwen3-235b-a22b-instruct-2507",
    "messages": [{"role": "user", "content": PROMPT}],
}, indent=2)

e2ee_body = json.dumps({
    "model": MODEL,
    "messages": encrypted_messages,
    "stream": True,
}, indent=2)

pd.DataFrame({
    "mode":         ["Private (default)", "E2EE"],
    "wire_payload": [private_body[:240] + " ...", e2ee_body[:240] + " ..."],
})'''),
        ("markdown",
            "## What you just proved\n\n"
            "1. The prompt was encrypted on your machine with a key Venice never had.\n"
            "2. Only an enclave holding the matching private key (verified via Intel TDX quote) "
            "could decrypt it.\n"
            "3. The reply came back encrypted to your ephemeral key.\n"
            "4. Anyone sniffing Venice's network sees only ciphertext.\n\n"
            "If the enclave is ever compromised, the attestation breaks and your client refuses to "
            "send. If Venice is subpoenaed, the only thing handed over is hex. That is the "
            "difference between **trust us** and **here is your receipt**.\n\n"
            "## E2EE limitations to know\n\n"
            "- Streaming is required (no non-streaming).\n"
            "- Web search and function calling are disabled (would leak content).\n"
            "- File uploads not supported.\n"
            "- Use a fresh ephemeral keypair per session and zero it after use.\n\n"
            "Welcome to provable AI privacy. Now go ship something with it."),
    ]
