"""02 - Embeddings + RAG. The richest notebook: visualization, classification,
clustering with auto-naming, and a full mini-RAG pipeline.

Heavily inspired by the structure of the OpenAI cookbook embeddings notebooks
(Clustering.ipynb, Visualizing_embeddings_in_3D.ipynb, Classification_using_embeddings.ipynb,
Question_answering_using_embeddings.ipynb), all reimplemented against Venice.
"""

from ._common import Cell, header, install_cell, setup_cell


NOTEBOOK = "02-embeddings-and-rag.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "Embeddings: visualize, classify, cluster, and retrieve",
            "Embeddings turn any text into a 1024-dimensional vector. With that vector you can do "
            "search, recommendations, classification, clustering, and retrieval-augmented generation. "
            "We will do all five against a fictional AI builder cohort.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. **Embed** all 14 cohort teams with `text-embedding-bge-m3`.\n"
            "2. **Visualize** them in 2D with t-SNE so you can literally see which teams are similar.\n"
            "3. **Classify** unseen teams into one of 4 tracks (DeFi / AI / Social / Infra) using a "
            "scikit-learn random forest on top of the embeddings.\n"
            "4. **Cluster** the cohort with k-means and let an LLM auto-name each cluster.\n"
            "5. **Build a mini-RAG**: chunk a program brief, embed it, retrieve the top-3 "
            "snippets for any question, and generate an answer.\n\n"
            "Why this matters: this is the same recipe used by every serious search and assistant "
            "product. Embeddings are the cheapest way to add memory to an LLM."),
        ("markdown", "## Setup"),
        install_cell("numpy scikit-learn pandas matplotlib"),
        setup_cell(),
        ("code",
            '''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EMBEDDING_MODEL = "text-embedding-bge-m3"
CHAT_MODEL = "llama-3.3-70b"

def embed(texts):
    """Batch-embed a list of strings. Returns an (n, d) numpy array."""
    if isinstance(texts, str):
        texts = [texts]
    r = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([d.embedding for d in r.data])

def cosine(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T

print("Helper functions ready.")'''),
        ("markdown",
            "## The dataset: an AI builder cohort\n\n"
            "Fourteen fictional but realistic onchain teams, each with a one-paragraph description "
            "and a ground-truth track label (DeFi / AI / Social / Infra). Same shape as the data "
            "you would get from any accelerator application form."),
        ("code",
            '''COHORT = [
    ("PrivPay",      "DeFi",   "A privacy-preserving payments app on Base. Uses E2EE chat between counterparties and stealth addresses to hide transaction graphs from chain analytics firms."),
    ("YieldKit",     "DeFi",   "An automated yield router for Base liquidity. Rebalances between Aerodrome and Uniswap based on real-time fee data and gas costs."),
    ("StableSwap",   "DeFi",   "A stablecoin DEX optimized for low-slippage swaps between USDC, DAI, and USDbC on Base."),
    ("LendIQ",       "DeFi",   "Risk-scored undercollateralized loans. Borrowers attest to off-chain income inside a TEE, lenders see only the score."),
    ("MediNote",     "AI",     "Doctors dictate visit notes that are encrypted on-device, transcribed in a TEE, and summarized into SOAP notes. HIPAA-compliant by construction."),
    ("PrivateRAG",   "AI",     "An end-to-end encrypted retrieval-augmented generation API. Customers upload their corpus, ciphertext goes in, plaintext only ever lives inside an enclave."),
    ("ColdStart",    "AI",     "An AI cofounder for solo builders. Generates landing pages, ad copy, and growth experiments. Powered by uncensored open-source models."),
    ("VoiceAgent",   "AI",     "A multilingual phone agent for restaurants. Books, cancels, and upsells reservations using TTS + STT in the same enclave."),
    ("FrenZone",     "Social", "A friend-graph social network where every post is end-to-end encrypted to a chosen group. No ads, no platform reads your messages."),
    ("AnonReply",    "Social", "An anonymous reply layer for X / Farcaster. Uses zero-knowledge group membership proofs so replies are unlinkable but provably from a verified community."),
    ("FarcasterFM",  "Social", "A radio station for Farcaster. AI DJ comments on the trending casts of the hour and pipes audio to a Frame."),
    ("PartyDAO",     "Social", "Coordinates real-life events between friends with on-chain RSVPs and shared expense tracking."),
    ("BaseRPC",      "Infra",  "A privacy-preserving public RPC for Base. Mixes user requests so no single node can deanonymize a wallet."),
    ("AttestKit",    "Infra",  "A developer toolkit for verifying TEE attestations. Drop-in SDK for any Node or Python backend."),
]
df = pd.DataFrame(COHORT, columns=["team", "track", "description"])
df'''),
        ("markdown",
            "## 1. Embed the whole cohort\n\n"
            "One call, 14 vectors of length 1024. The total cost is fractions of a cent on Venice."),
        ("code",
            '''matrix = embed(df["description"].tolist())
print("Shape:", matrix.shape, "  (one row per team, 1024 dims each)")
print("First 8 dims of PrivPay:", np.round(matrix[0][:8], 3))'''),
        ("markdown",
            "## 2. Visualize the cohort with t-SNE\n\n"
            "1024 dimensions is too many for the human eye. t-SNE projects them down to 2 while "
            "preserving local neighborhoods. Teams that work on similar problems will literally cluster "
            "together on the plot, even though we never told t-SNE about the track labels."),
        ("code",
            '''from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=4, random_state=42, init="random", learning_rate=200)
xy = tsne.fit_transform(matrix)

palette = {"DeFi": "#2563eb", "AI": "#dc2626", "Social": "#16a34a", "Infra": "#9333ea"}

plt.figure(figsize=(9, 6))
for track, color in palette.items():
    mask = df["track"] == track
    plt.scatter(xy[mask, 0], xy[mask, 1], c=color, s=120, alpha=0.7, label=track, edgecolor="black")
    for i in df[mask].index:
        plt.annotate(df.loc[i, "team"], (xy[i, 0] + 1, xy[i, 1] + 0.5), fontsize=8)

plt.legend(loc="best")
plt.title("Cohort projected to 2D with t-SNE")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.show()'''),
        ("markdown",
            "Notice anything? Teams within the same track cluster together because the language they "
            "use is similar, even when we never gave the model the track label. This is the entire "
            "intuition behind embedding-based search."),
        ("markdown",
            "## 3. Classification: predict the track of an unseen team\n\n"
            "Since teams in the same track land in similar regions of embedding space, we can train a "
            "tiny classifier on top of the embeddings to predict the track of a brand new team. We use "
            "a Random Forest from scikit-learn (this is exactly the OpenAI cookbook pattern, just "
            "with Venice embeddings)."),
        ("code",
            '''from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    matrix, df["track"].values, test_size=0.3, random_state=7, stratify=df["track"].values
)

clf = RandomForestClassifier(n_estimators=200, random_state=7)
clf.fit(X_train, y_train)

print("Held-out accuracy:", round(clf.score(X_test, y_test), 2))
print()
print(classification_report(y_test, clf.predict(X_test)))'''),
        ("markdown",
            "Now predict the track of three teams that were never in the cohort. The classifier has "
            "never seen these descriptions before, only their embedding."),
        ("code",
            '''new_teams = [
    ("ZkRoll",   "A zero-knowledge rollup that batches Base transactions into a single proof to cut fees by 95%."),
    ("EncMail",  "An end-to-end encrypted email client with a TEE-hosted spam filter."),
    ("VibeCast", "A Farcaster-native podcast app where every episode is one infinite scroll of casts."),
]
new_matrix = embed([t[1] for t in new_teams])
preds = clf.predict(new_matrix)

for (name, desc), p in zip(new_teams, preds):
    print(f"{name:9s}  ->  predicted track: {p}")'''),
        ("markdown",
            "## 4. Clustering with k-means and auto-naming\n\n"
            "What if we did not have the track labels? We can let k-means discover the natural groups, "
            "then ask an LLM to name each cluster from a few sample descriptions. This is how product "
            "teams build customer-segment dashboards from raw support tickets."),
        ("code",
            '''from sklearn.cluster import KMeans

n_clusters = 4
km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
df["cluster"] = km.fit_predict(matrix)

# Plot clusters with the same t-SNE projection
cluster_palette = ["#f97316", "#06b6d4", "#a855f7", "#22c55e"]
plt.figure(figsize=(9, 6))
for c in range(n_clusters):
    mask = df["cluster"] == c
    plt.scatter(xy[mask, 0], xy[mask, 1], c=cluster_palette[c], s=120, alpha=0.7,
                label=f"cluster {c}", edgecolor="black")
    for i in df[mask].index:
        plt.annotate(df.loc[i, "team"], (xy[i, 0] + 1, xy[i, 1] + 0.5), fontsize=8)
plt.legend(loc="best")
plt.title("k-means clusters over the same embedding space")
plt.tight_layout()
plt.show()'''),
        ("markdown",
            "Now ask the LLM to name each cluster:"),
        ("code",
            '''def name_cluster(samples: list[str]) -> str:
    joined = "\\n".join(f"- {s}" for s in samples)
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "Below are descriptions of startups that landed in the same cluster. "
                "What is the single best 3-word category name for them?\\n\\n"
                f"{joined}\\n\\nReply with only the 3-word category."
            ),
        }],
        temperature=0,
    )
    return r.choices[0].message.content.strip().strip('"')

names = {}
for c in range(n_clusters):
    samples = df[df["cluster"] == c]["description"].tolist()
    names[c] = name_cluster(samples)

named = df.copy()
named["cluster_name"] = named["cluster"].map(names)
named[["team", "track", "cluster", "cluster_name"]].sort_values("cluster")'''),
        ("markdown",
            "How well do the unsupervised clusters line up with the ground-truth tracks?"),
        ("code",
            '''pd.crosstab(named["track"], named["cluster_name"])'''),
        ("markdown",
            "## 5. A full mini-RAG pipeline\n\n"
            "Now the headline act. We will build a tiny retrieval-augmented question-answering system "
            "over a fictional accelerator program brief. The recipe:\n\n"
            "1. **Chunk** the source into ~150-word passages.\n"
            "2. **Embed** every chunk.\n"
            "3. For a question, embed it and retrieve the top-k most similar chunks.\n"
            "4. **Stuff** them into a system prompt and ask the model to answer using only that "
            "context.\n\n"
            "This is the exact pattern used by every doc-Q&A bot, customer support assistant, and "
            "internal knowledge tool."),
        ("code",
            '''DOC = """
The Builder Program is a 12-week accelerator for teams shipping onchain consumer apps.
The cohort starts in Q3 and ends with a public Demo Day in Q4. Each accepted team
receives a $25k grant in USDC, weekly office hours with the program's engineering
partners, and credits with platforms including Venice AI, Privy, and Coinbase
Developer Platform.

Eligibility: teams must have at least one full-time technical founder, be incorporated
or willing to incorporate during the program, and commit to shipping a working app
on mainnet by Demo Day. Pre-product teams are welcome.

Program structure: weeks 1-2 focus on user research and product spec. Weeks 3-6 are
heads-down build. Weeks 7-9 are private beta with a target of 100 weekly active
users. Weeks 10-12 are growth, polish, and Demo Day prep.

Mentorship: every team is paired with two mentors, one from the program's engineering
side and one from the partner network. Mentor sessions are weekly, 45 minutes, and
recorded with consent. Office hours are open: any team can drop in to any session.

Funding: the $25k grant is paid in two tranches, $15k at week 1 and $10k at week 6
contingent on shipping milestones. The program may also lead a follow-on pre-seed
round of $250k-$1M for top performing teams at Demo Day. No equity is taken in
exchange for the grant or the program.

Demo Day: held in San Francisco and live-streamed globally. Each team gets a 4-minute
live demo and a 2-minute Q&A in front of an audience of investors, partners, and the
broader builder community. Teams are judged on traction, product quality, and onchain
volume in the 4 weeks leading up to Demo Day.

Privacy and credentials: teams keep all IP. Code can be open or closed source.
Partner credit balances are sandbox until you ship to mainnet, then upgraded.
"""

def chunk(text: str, target_words: int = 80, overlap: int = 15) -> list[str]:
    """Naive but workable: word-window chunking with overlap."""
    words = text.split()
    chunks = []
    step = target_words - overlap
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i + target_words]).strip()
        if piece:
            chunks.append(piece)
    return chunks

chunks = chunk(DOC)
print(f"Document split into {len(chunks)} chunks.")
chunk_matrix = embed(chunks)
print("Embeddings shape:", chunk_matrix.shape)'''),
        ("code",
            '''def retrieve(question: str, k: int = 3):
    qv = embed(question)[0]
    sims = cosine(qv, chunk_matrix).flatten()
    top = np.argsort(sims)[::-1][:k]
    return [(float(sims[i]), chunks[i]) for i in top]

def answer(question: str, k: int = 3) -> str:
    hits = retrieve(question, k=k)
    context = "\\n\\n---\\n\\n".join(h[1] for h in hits)
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": (
                "Answer the user using ONLY the context below. If the answer is not in the context, "
                "say 'I do not know based on the program brief.'\\n\\nCONTEXT:\\n" + context
            )},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return r.choices[0].message.content

QUESTIONS = [
    "How much grant funding do teams get and when?",
    "Does the program take equity in exchange for the grant?",
    "What happens during weeks 7 to 9?",
    "Can pre-product teams apply?",
    "Who is the head judge at Demo Day?",
]

for q in QUESTIONS:
    print("Q:", q)
    print("A:", answer(q))
    print()'''),
        ("markdown",
            "Notice how the last question (\"Who is the head judge at Demo Day?\") gets a clean "
            "\"I do not know\" because the answer is genuinely not in the program brief. That refusal "
            "is the difference between a helpful RAG bot and a hallucinating one."),
        ("markdown", "## Inspect what was retrieved"),
        ("code",
            '''for score, snippet in retrieve("how is the cohort funded"):
    print(f"score={score:.3f}")
    print(snippet[:200] + "...")
    print()'''),
        ("markdown",
            "## Recap\n\n"
            "You now have all four superpowers of embeddings:\n"
            "- **Search and similarity** (cosine over the matrix)\n"
            "- **Classification** (random forest on top of the matrix)\n"
            "- **Clustering** (k-means + LLM auto-naming)\n"
            "- **RAG** (retrieve top-k, stuff into prompt, answer)\n\n"
            "All four cost less than $0.001 to demo. Next: `03-image-generation.ipynb` for the visual "
            "side of Venice."),
    ]
