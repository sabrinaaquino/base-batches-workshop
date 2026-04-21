"""08 - End-to-end encrypted inference. The headliner. We compare what each privacy
mode looks like over the wire, then implement E2EE step by step in pure Python."""

from ._common import Cell, header, install_cell, setup_cell


NOTEBOOK = "08-e2ee-encryption.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "End-to-end encrypted inference: provable, not promised",
            "Most AI providers ask you to trust them. Venice's E2EE mode encrypts your prompt on the "
            "client with a key Venice never sees, decrypts it only inside a Trusted Execution "
            "Environment, and gives you a cryptographic attestation that proves the enclave was "
            "honest. We will implement it from scratch and inspect every byte.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. **A privacy mode comparison table** so you know what each tier actually guarantees.\n"
            "2. **The full E2EE handshake** end to end: fetch the enclave's signed key, derive a "
            "shared secret with ECDH (secp256k1), encrypt the prompt with AES-256-GCM, and verify "
            "the attestation.\n"
            "3. **A side-by-side diff** of what Venice would see in private mode vs E2EE mode for "
            "the same prompt.\n"
            "4. **Decrypt the response** the model produced inside the enclave.\n\n"
            "Cost: zero extra. E2EE is included on Pro tiers and bundled with TEE."),
        ("markdown",
            "## Privacy modes at a glance\n\n"
            "| Mode | Who can read your prompt | Hardware proof | Default |\n"
            "|---|---|---|---|\n"
            "| Anonymized (3rd party) | 3rd party provider only, never linked to you | no | for proxied frontier models |\n"
            "| Private (Venice default) | Venice infrastructure for the duration of inference, then discarded | no | yes, for open-source models |\n"
            "| TEE | Only the verified enclave. Even Venice operators cannot read it | yes (remote attestation) | opt-in |\n"
            "| E2EE | Only the verified enclave. Encrypted on your device first. Even Venice's network cannot see plaintext | yes (attestation + ECDH) | opt-in |\n\n"
            "The whole point of E2EE: you do not have to trust Venice. You verify."),
        ("markdown", "## Setup"),
        install_cell("coincurve cryptography"),
        setup_cell(),
        ("code",
            '''import json, base64, hashlib, requests
import coincurve
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

API_BASE = "https://api.venice.ai/api/v1"
HEADERS = {"Authorization": f"Bearer {api_key}"}

def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode()

def b64d(s: str) -> bytes:
    return base64.b64decode(s)'''),
        ("markdown",
            "## 1. Fetch the enclave's signed public key\n\n"
            "Venice rotates a per-enclave keypair. The public half is published with a remote "
            "attestation that proves it came from a real TEE (Intel TDX / NVIDIA Confidential "
            "Computing). You should verify the attestation against your expected enclave measurement, "
            "but for the demo we just inspect it."),
        ("code",
            '''r = requests.get(f"{API_BASE}/keys/e2ee", headers=HEADERS, timeout=30)
key_info = r.json()
print(json.dumps(key_info, indent=2)[:600], "...")
enclave_pub_hex = key_info.get("public_key") or key_info.get("publicKey")
attestation    = key_info.get("attestation")
print("\\nEnclave public key (compressed secp256k1):", enclave_pub_hex)'''),
        ("markdown",
            "## 2. Generate your ephemeral keypair and derive the shared secret\n\n"
            "ECDH over secp256k1 (the same curve Bitcoin and Ethereum use). Your private key never "
            "leaves your machine. The enclave private key never leaves the enclave. Only the resulting "
            "shared secret can decrypt the prompt."),
        ("code",
            '''my_priv = coincurve.PrivateKey()
my_pub  = my_priv.public_key.format(compressed=True)
print("My ephemeral public key:", my_pub.hex())

enclave_pub = coincurve.PublicKey(bytes.fromhex(enclave_pub_hex))
shared = my_priv.ecdh(enclave_pub.format(compressed=True))
session_key = hashlib.sha256(shared).digest()
print("Derived 256-bit AES key:", session_key.hex()[:32], "...")'''),
        ("markdown",
            "## 3. Encrypt the prompt with AES-256-GCM\n\n"
            "GCM gives us confidentiality and authentication in one shot. Without the right key the "
            "ciphertext is indistinguishable from random bytes."),
        ("code",
            '''import os as _os

PROMPT = (
    "I am a doctor reviewing a patient case. The patient is a 47-year-old female with "
    "chest pain radiating to the left arm. Suggest a differential diagnosis."
)

aesgcm = AESGCM(session_key)
nonce  = _os.urandom(12)
ciphertext = aesgcm.encrypt(nonce, PROMPT.encode(), None)

envelope = {
    "client_pub": my_pub.hex(),
    "nonce":      b64e(nonce),
    "ciphertext": b64e(ciphertext),
}
print("Ciphertext size:", len(ciphertext), "bytes")
print("Ciphertext (first 60 bytes):", b64e(ciphertext)[:60], "...")
print()
print("Envelope keys:", list(envelope.keys()))'''),
        ("markdown",
            "## 4. The diff: what Venice sees in Private mode vs E2EE mode\n\n"
            "This is the punchline. In Private mode the request body contains the literal prompt. "
            "In E2EE mode it contains an encrypted blob. Same outcome for the user, completely "
            "different threat model for the operator."),
        ("code",
            '''import pandas as pd

private_request = {
    "model": "llama-3.3-70b",
    "messages": [{"role": "user", "content": PROMPT}],
}

e2ee_request = {
    "model": "llama-3.3-70b",
    "encrypted": envelope,
}

pd.DataFrame({
    "mode":         ["Private (default)", "E2EE"],
    "request_body": [json.dumps(private_request)[:200], json.dumps(e2ee_request)[:200]],
})'''),
        ("markdown",
            "## 5. Send the encrypted request and decrypt the reply\n\n"
            "The enclave decrypts your prompt, runs the model, encrypts the response with the same "
            "shared key, and returns it. Plaintext never crosses the network."),
        ("code",
            '''r = requests.post(
    f"{API_BASE}/chat/completions",
    headers={**HEADERS, "Content-Type": "application/json"},
    json={
        "model": "llama-3.3-70b",
        "encrypted": envelope,
    },
    timeout=120,
)
print("Status:", r.status_code)
encrypted_resp = r.json()
print("Encrypted response keys:", list(encrypted_resp.keys())[:6])'''),
        ("code",
            '''nonce_resp  = b64d(encrypted_resp["nonce"])
cipher_resp = b64d(encrypted_resp["ciphertext"])

plaintext = aesgcm.decrypt(nonce_resp, cipher_resp, None).decode()
parsed = json.loads(plaintext)
print(parsed["choices"][0]["message"]["content"])'''),
        ("markdown",
            "## 6. Verify the attestation\n\n"
            "The most important step. The attestation is a signed quote from the TEE hardware that "
            "says \"the code running here has measurement X, and this public key belongs to me.\" "
            "You should compare the measurement against the published Venice build hash. If they "
            "match, the enclave is honest. If they do not, abort.\n\n"
            "For the workshop we just print the attestation. In production you would parse the quote "
            "with a verifier like [Phala's attestation explorer](https://proof.t16z.com/) or the "
            "[Intel TDX SDK](https://github.com/intel/SGXDataCenterAttestationPrimitives)."),
        ("code",
            '''import textwrap
print("Attestation type:", type(attestation).__name__)
preview = json.dumps(attestation, indent=2) if isinstance(attestation, dict) else str(attestation)
print(textwrap.shorten(preview, width=600))'''),
        ("markdown",
            "## What we just proved\n\n"
            "1. Your prompt was encrypted on your laptop using a key Venice never had.\n"
            "2. Only an enclave holding the matching private key could decrypt it.\n"
            "3. The enclave's identity is provable via a hardware-signed attestation.\n"
            "4. The response came back encrypted to the same session key.\n\n"
            "If the enclave is compromised, the attestation breaks and you abort. If Venice's "
            "operators are subpoenaed, the only thing they can hand over is ciphertext. That is the "
            "difference between *trust us* and *here is your receipt*.\n\n"
            "Welcome to provable AI privacy. Now go ship something with it."),
    ]
