"""01 - Chat completions. Three sub-examples in the OpenAI cookbook style:
    1. Side by side model comparison with a pandas DataFrame
    2. Pydantic structured outputs
    3. Tool calling for a Base Batches schedule assistant
"""

from ._common import Cell, header, install_cell, setup_cell


NOTEBOOK = "01-chat-completions.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "Chat completions, the way you will actually ship them",
            "Three sub-examples that map to the three patterns you will use 90% of the time: "
            "model selection by benchmark, structured outputs you can trust, and tool calling for "
            "agents.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. **Model bake-off:** run the same prompt across 3 Venice models, score them on "
            "latency / tokens / a quality rubric, and pick a winner.\n"
            "2. **Structured outputs:** force the model to return a strict Pydantic schema so your "
            "downstream code never crashes on a stray `\"sure!\"`.\n"
            "3. **Tool calling:** wire up two Python functions (search a schedule, RSVP to an event) "
            "and let the model orchestrate them.\n\n"
            "Every Venice model is OpenAI-compatible, so what you learn here transfers straight to "
            "any project that already uses the OpenAI SDK."),
        ("markdown", "## Setup"),
        install_cell("pydantic pandas"),
        setup_cell(),
        ("markdown",
            "## Sub-example 1: model bake-off\n\n"
            "Picking a model is half the work. We run the same business-relevant prompt across three "
            "Venice models and grade each response with a fourth model acting as judge. This is the "
            "same pattern OpenAI uses internally to evaluate releases."),
        ("code",
            '''import pandas as pd

CANDIDATES = ["llama-3.3-70b", "venice-uncensored", "qwen3-235b-a22b-instruct-2507"]
JUDGE     = "qwen3-235b-a22b-instruct-2507"  # supports response_format=json_object

PROMPT = (
    "You are pitching a privacy-preserving AI to a non-technical CEO of a hospital. "
    "In <120 words, explain why end-to-end encrypted inference matters and give one "
    "concrete example they will remember at dinner."
)

results = []
for model in CANDIDATES:
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.4,
    )
    results.append({
        "model": model,
        "latency_s": round(time.perf_counter() - t0, 2),
        "tokens": r.usage.completion_tokens,
        "answer": r.choices[0].message.content.strip(),
    })

df = pd.DataFrame(results)
df[["model", "latency_s", "tokens"]]'''),
        ("markdown",
            "Now we use the judge model to score each answer on a 1-5 scale for clarity, persuasion, "
            "and accuracy. Returning a strict schema means we can sort the table programmatically."),
        ("code",
            '''import json

RUBRIC = """You are an expert evaluator. Score the answer 1-5 on each axis:
- clarity: is it easy for a non-technical CEO to follow?
- persuasion: would they remember this at dinner?
- accuracy: is the technical claim about TEEs / E2EE correct?

Reply with ONLY a JSON object: {"clarity": int, "persuasion": int, "accuracy": int, "comment": str}
"""

def judge(answer: str) -> dict:
    r = client.chat.completions.create(
        model=JUDGE,
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"Answer to score:\\n\\n{answer}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(r.choices[0].message.content)

scores = [judge(row["answer"]) for row in results]
for row, s in zip(results, scores):
    row.update(s)
    row["total"] = s["clarity"] + s["persuasion"] + s["accuracy"]

leaderboard = pd.DataFrame(results)[
    ["model", "latency_s", "tokens", "clarity", "persuasion", "accuracy", "total"]
].sort_values("total", ascending=False).reset_index(drop=True)
leaderboard'''),
        ("markdown",
            "Read the actual winning answer:"),
        ("code",
            '''winner = leaderboard.iloc[0]["model"]
print("WINNER:", winner)
print()
print(next(r["answer"] for r in results if r["model"] == winner))'''),
        ("markdown",
            "## Sub-example 2: structured outputs with Pydantic\n\n"
            "Free-form text is great for humans, terrible for downstream code. With a Pydantic schema "
            "we can guarantee the response is valid JSON that matches our types, then pass it straight "
            "into a database or a UI."),
        ("code",
            '''from pydantic import BaseModel, Field
from typing import Literal

class TeamPitch(BaseModel):
    team_name: str
    one_liner: str = Field(description="Punchy elevator pitch, max 18 words.")
    primary_track: Literal["DeFi", "AI", "Social", "Infra", "Consumer"]
    risk_level: int = Field(ge=1, le=5)
    venice_endpoints_to_use: list[str]

raw = """We are 'NoteTaker'. We let doctors dictate visit notes that are encrypted before
they ever leave the device, transcribed inside a TEE, and summarized into a SOAP note.
Doctors love it because they keep HIPAA cover, patients love it because nobody at OpenAI
or Google can read it."""

schema_prompt = (
    "Extract a TeamPitch from the description. Return only valid JSON matching this schema:\\n"
    + json.dumps(TeamPitch.model_json_schema(), indent=2)
)

r = client.chat.completions.create(
    model="qwen3-235b-a22b-instruct-2507",  # supports response_format
    messages=[
        {"role": "system", "content": schema_prompt},
        {"role": "user", "content": raw},
    ],
    response_format={"type": "json_object"},
    temperature=0,
)
pitch = TeamPitch.model_validate_json(r.choices[0].message.content)
pitch'''),
        ("markdown",
            "Because `pitch` is a real Python object, we can use it like one:"),
        ("code",
            '''print(f"{pitch.team_name} ({pitch.primary_track}, risk {pitch.risk_level}/5)")
print(f">> {pitch.one_liner}")
print(f">> Endpoints: {', '.join(pitch.venice_endpoints_to_use)}")'''),
        ("markdown",
            "## Sub-example 3: tool calling\n\n"
            "Venice supports the OpenAI tool calling spec. We will give the model two tools (look up "
            "the schedule, RSVP to an event) and let it route a vague user question to the right one. "
            "This is the foundation of every agent you will build later in the workshop."),
        ("code",
            '''SCHEDULE = [
    {"id": "TALK-1", "title": "Welcome from Base", "time": "Mon 09:00", "track": "main"},
    {"id": "TALK-2", "title": "Building with x402",  "time": "Mon 11:00", "track": "DeFi"},
    {"id": "TALK-3", "title": "E2EE inference live demo", "time": "Mon 14:00", "track": "AI"},
    {"id": "TALK-4", "title": "Demo day office hours", "time": "Tue 10:00", "track": "main"},
]
RSVPS: list[dict] = []

def search_schedule(query: str) -> list[dict]:
    q = query.lower()
    return [t for t in SCHEDULE if q in t["title"].lower() or q in t["track"].lower()]

def rsvp(talk_id: str, attendee: str) -> dict:
    talk = next((t for t in SCHEDULE if t["id"] == talk_id), None)
    if not talk:
        return {"ok": False, "error": "talk_id not found"}
    RSVPS.append({"talk_id": talk_id, "attendee": attendee})
    return {"ok": True, "confirmed_for": talk["title"]}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_schedule",
            "description": "Search the workshop schedule by topic, track, or keyword.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rsvp",
            "description": "RSVP an attendee to a talk by talk_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "talk_id": {"type": "string"},
                    "attendee": {"type": "string"},
                },
                "required": ["talk_id", "attendee"],
            },
        },
    },
]'''),
        ("markdown",
            "Now the agent loop. The model decides which tool to call, we execute it locally, return "
            "the result, and let the model produce the final answer. This is exactly the OpenAI agent "
            "loop, ported to Venice."),
        ("code",
            '''def run_agent(user_msg: str, attendee: str = "Sabrina"):
    messages = [
        {"role": "system", "content": (
            f"You help workshop attendees find talks and RSVP. The user's name is {attendee}. "
            "Always call a tool when relevant. Never make up talk_ids."
        )},
        {"role": "user", "content": user_msg},
    ]

    for _ in range(4):
        r = client.chat.completions.create(
            model="qwen3-235b-a22b-instruct-2507",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = r.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            if call.function.name == "search_schedule":
                result = search_schedule(**args)
            elif call.function.name == "rsvp":
                result = rsvp(**args)
            else:
                result = {"error": "unknown tool"}
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return "(agent gave up after 4 turns)"

print(run_agent("Anything happening Monday afternoon about privacy stuff? sign me up if so."))
print()
print("Stored RSVPs:", RSVPS)'''),
        ("markdown",
            "## Recap\n\n"
            "You now have the three primitives that ship every real chat product:\n"
            "- a benchmarking harness to pick the right model,\n"
            "- a Pydantic gate that turns LLM output into safe Python objects,\n"
            "- a tool calling loop that lets the model take action.\n\n"
            "Next: `02-embeddings-and-rag.ipynb` adds memory and retrieval."),
    ]
