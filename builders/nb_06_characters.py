"""06 - Characters. Use a Venice character, build a panel discussion, and combine
characters with the RAG pipeline from notebook 02."""

from ._common import Cell, header, install_cell, setup_cell


NOTEBOOK = "06-characters.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "Characters: named personas with persistent voice and expertise",
            "A character bundles a system prompt, a persona, and sometimes a reference voice into a "
            "first-class object you can call by name. We will use one, then put three on a panel and "
            "make them debate, then give one of them a knowledge base.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. List available characters.\n"
            "2. Single-character chat.\n"
            "3. **Panel discussion**: three characters answer the same question in turn so you can "
            "compare their takes side by side.\n"
            "4. **Character + RAG**: combine a character with the BB003 program brief from notebook "
            "02 so the character only answers using grounded facts."),
        ("markdown", "## Setup"),
        install_cell("requests"),
        setup_cell(),
        ("code",
            '''import requests, pandas as pd

def list_characters(limit: int = 20):
    r = requests.get(
        "https://api.venice.ai/api/v1/characters",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    r.raise_for_status()
    items = r.json().get("data", [])[:limit]
    return pd.DataFrame([
        {"slug": c.get("slug"), "name": c.get("name"), "summary": (c.get("description") or "")[:80]}
        for c in items
    ])

list_characters(limit=12)'''),
        ("markdown",
            "## 1. Single-character chat\n\n"
            "Pick a character slug from the table above and pass it as `character_slug` in "
            "`extra_body`. Everything else is a normal chat completion."),
        ("code",
            '''def chat_as(character_slug: str, user_msg: str, model: str = "venice-uncensored") -> str:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_msg}],
        extra_body={"venice_parameters": {"character_slug": character_slug}},
    )
    return r.choices[0].message.content

# Replace with a slug that exists on your account
SLUG = "venice"
print(chat_as(SLUG, "In one paragraph, why should I care about end-to-end encrypted AI?"))'''),
        ("markdown",
            "## 2. Panel discussion\n\n"
            "Same question, three characters, side by side. Useful for product naming, brainstorming, "
            "or deciding which voice your brand should sound like."),
        ("code",
            '''PANEL = ["venice", "alan-watts", "satoshi"]  # adjust slugs to ones available on your account
QUESTION = "Should builders trust closed AI labs with their users' data?"

panel_rows = []
for slug in PANEL:
    try:
        ans = chat_as(slug, QUESTION)
    except Exception as e:
        ans = f"(skipped: {e})"
    panel_rows.append({"character": slug, "answer": ans})

pd.DataFrame(panel_rows)'''),
        ("markdown",
            "## 3. Character + RAG\n\n"
            "Characters have voice. RAG has facts. Combine them and you get an on-brand expert that "
            "only speaks from your knowledge base. Here we re-use the BB003 program brief from "
            "notebook 02 and have a Venice character answer questions about the program."),
        ("code",
            '''import numpy as np

EMBEDDING_MODEL = "text-embedding-bge-m3"

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    r = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([d.embedding for d in r.data])

DOC = """Base Batches 003 (BB003) is a 12-week builder program. Each accepted team
gets a $25k grant in USDC on Base, weekly office hours with Base engineering, and
credits with Venice AI. Eligibility: at least one full-time technical founder.
The program structure is 2 weeks of research, 4 weeks of build, 3 weeks of beta,
and 3 weeks of polish before Demo Day. Mentorship is weekly 45 minute sessions.
The grant is paid as $15k at week 1 and $10k at week 6. Base may lead a follow-on
pre-seed round of $250k to $1M for top performers. Demo Day is in San Francisco
and live-streamed. Each team gets 4 minutes to demo and 2 minutes of Q&A. Teams
keep all IP and Base takes no equity for the grant."""

chunks = [c.strip() for c in DOC.split(". ") if c.strip()]
chunk_matrix = embed(chunks)

def grounded_chat(character_slug: str, question: str, k: int = 3) -> str:
    qv = embed(question)[0]
    sims = (chunk_matrix @ qv) / (np.linalg.norm(chunk_matrix, axis=1) * np.linalg.norm(qv))
    top = np.argsort(sims)[::-1][:k]
    context = "\\n".join(chunks[i] for i in top)

    r = client.chat.completions.create(
        model="venice-uncensored",
        messages=[
            {"role": "system", "content": (
                "Answer the user using ONLY the context. If you cannot answer from it, say so.\\n"
                "CONTEXT:\\n" + context
            )},
            {"role": "user", "content": question},
        ],
        extra_body={"venice_parameters": {"character_slug": character_slug}},
        temperature=0.2,
    )
    return r.choices[0].message.content

print(grounded_chat(SLUG, "What is the funding structure for BB003 grants?"))'''),
        ("markdown",
            "## Recap\n\n"
            "Characters give you persistent voice. Add RAG and you have a grounded, on-brand expert "
            "you can hand to users. Next: `07-x402-wallet-payments.ipynb` for paying with a wallet "
            "instead of an API key."),
    ]
