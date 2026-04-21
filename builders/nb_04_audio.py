"""04 - Audio. Multi-voice podcast, silence trimming and segmentation, and a baseline
vs post-processed transcription comparison (the OpenAI Whisper cookbook pattern).
"""

from ._common import Cell, header, install_cell, setup_cell


NOTEBOOK = "04-audio-tts-stt.ipynb"


def cells() -> list[Cell]:
    return [
        ("markdown", header(
            NOTEBOOK,
            "Audio: voices, transcription, and the post-processing trick that fixes everything",
            "Venice ships text-to-speech with multiple voices and Whisper-class speech-to-text. We "
            "will build a two-host podcast intro, segment a long file the right way, and use the "
            "OpenAI cookbook's classic baseline-vs-post-processed pattern to fix domain-specific "
            "transcription errors.",
        )),
        ("markdown",
            "## What you will build\n\n"
            "1. **One-line TTS** with a voice catalog displayed as a pandas DataFrame.\n"
            "2. **Two-host podcast intro** that swaps voices on every turn.\n"
            "3. **Silence trimming + segmentation** with pydub for files longer than 30 seconds.\n"
            "4. **Baseline transcription** with the raw STT endpoint.\n"
            "5. **Post-processed transcription** that runs the raw output through chat to fix "
            "domain-specific spellings (Venice, x402, BB003, TEE, E2EE).\n\n"
            "Cost: TTS is ~$0.015 per minute, STT is ~$0.006 per minute on Venice."),
        ("markdown", "## Setup"),
        install_cell("pydub"),
        setup_cell(),
        ("code",
            '''from pathlib import Path
from IPython.display import Audio, display
import pandas as pd

OUT = Path("assets_audio")
OUT.mkdir(exist_ok=True)

VOICES = pd.DataFrame([
    {"voice": "af_alloy",   "gender": "F", "accent": "American", "vibe": "neutral, professional"},
    {"voice": "af_bella",   "gender": "F", "accent": "American", "vibe": "warm, friendly"},
    {"voice": "af_nicole",  "gender": "F", "accent": "American", "vibe": "calm, confident"},
    {"voice": "am_adam",    "gender": "M", "accent": "American", "vibe": "deep, narrator"},
    {"voice": "am_michael", "gender": "M", "accent": "American", "vibe": "clear, podcast host"},
    {"voice": "bf_emma",    "gender": "F", "accent": "British",  "vibe": "crisp, broadcast"},
    {"voice": "bm_george",  "gender": "M", "accent": "British",  "vibe": "deep, BBC documentary"},
])
VOICES'''),
        ("markdown",
            "## 1. One-line TTS\n\n"
            "Speech in three lines: pick a voice, send text, save the file. Colab will play it inline."),
        ("code",
            '''def speak(text: str, voice: str = "af_bella", out_name: str = "out.mp3") -> Path:
    r = client.audio.speech.create(
        model="tts-kokoro",
        voice=voice,
        input=text,
        response_format="mp3",
    )
    path = OUT / out_name
    r.write_to_file(str(path))
    return path

p = speak("Hello Base Batches. This audio came out of Venice with no logs and no API key. Welcome.",
          voice="af_bella", out_name="hello.mp3")
display(Audio(str(p)))'''),
        ("markdown",
            "## 2. Two-host podcast intro\n\n"
            "Stitch multiple TTS calls together with pydub. We use one female and one male voice that "
            "trade lines, then export a single mp3 you can drop straight onto X."),
        ("code",
            '''from pydub import AudioSegment

SCRIPT = [
    ("af_bella",   "Welcome to Private by Default, the show where we put privacy back into AI."),
    ("am_michael", "Today on the show: end-to-end encrypted inference. What it actually means and why your hospital should care."),
    ("af_bella",   "Plus: a live demo where we encrypt a prompt on the client and watch Venice generate a reply they cannot read."),
    ("am_michael", "Stick around. This is going to be fun."),
]

clips = []
for i, (voice, line) in enumerate(SCRIPT):
    p = speak(line, voice=voice, out_name=f"podcast_{i}.mp3")
    clips.append(AudioSegment.from_file(p))

intro = AudioSegment.silent(duration=200)
for c in clips:
    intro += c + AudioSegment.silent(duration=120)

podcast_path = OUT / "podcast_intro.mp3"
intro.export(podcast_path, format="mp3")
display(Audio(str(podcast_path)))'''),
        ("markdown",
            "## 3. Silence trimming and segmentation\n\n"
            "Long files trip up STT in two ways: leading silence confuses the start, and >30s of "
            "audio in one shot hurts accuracy. The OpenAI cookbook fix is to trim leading silence and "
            "chunk into 60-second segments. Here is the same pattern, in 20 lines."),
        ("code",
            '''def first_sound_ms(sound: AudioSegment, threshold_db: float = -35.0, step_ms: int = 10) -> int:
    """Return the offset in ms where audio first crosses `threshold_db`."""
    t = 0
    while t < len(sound) and sound[t:t + step_ms].dBFS < threshold_db:
        t += step_ms
    return t

def trim_and_segment(audio: AudioSegment, segment_seconds: int = 60) -> list[AudioSegment]:
    start = first_sound_ms(audio)
    trimmed = audio[start:]
    seg_ms = segment_seconds * 1000
    return [trimmed[i:i + seg_ms] for i in range(0, len(trimmed), seg_ms)]

# Demo on the podcast we just built (silence at start, ~10 seconds long)
segments = trim_and_segment(intro, segment_seconds=15)
print(f"Trimmed leading silence and split into {len(segments)} segment(s).")'''),
        ("markdown",
            "## 4. Baseline transcription\n\n"
            "Run STT on the podcast we just generated. We expect domain words like \"Base Batches\" "
            "and \"Venice\" to occasionally come out as \"face badges\" or \"Vinnies\". That is what "
            "the next step fixes."),
        ("code",
            '''def transcribe(path: Path) -> str:
    with open(path, "rb") as f:
        r = client.audio.transcriptions.create(
            model="whisper-large",
            file=f,
            response_format="text",
        )
    return r.strip() if isinstance(r, str) else r.text.strip()

baseline = transcribe(podcast_path)
print(baseline)'''),
        ("markdown",
            "## 5. Post-processed transcription\n\n"
            "Pipe the raw transcript through a chat call with a glossary of domain terms. The model "
            "fixes brand names, acronyms, and capitalization in one pass. Same trick the OpenAI "
            "cookbook uses for ZyntriQix product SKUs, applied to our world."),
        ("code",
            '''GLOSSARY = """
Domain terms that must appear exactly as written:
- Venice (the AI platform; never "Vinnies", "Venezia")
- Base Batches (the program; never "face badges")
- BB003 (the cohort; never "B.B. zero zero three")
- TEE (Trusted Execution Environment, always uppercase)
- E2EE (End-to-End Encryption, always uppercase)
- x402 (the protocol; lowercase x, no space)
- Whisper, Llama, Kokoro (model names, capitalize)
"""

def postprocess(transcript: str) -> str:
    r = client.chat.completions.create(
        model="llama-3.3-70b",
        messages=[
            {"role": "system", "content": (
                "You correct speech-to-text transcripts. Fix capitalization, punctuation, and "
                "domain term spellings using the glossary. Do NOT add or remove content.\\n"
                + GLOSSARY
            )},
            {"role": "user", "content": transcript},
        ],
        temperature=0,
    )
    return r.choices[0].message.content.strip()

cleaned = postprocess(baseline)

print("BEFORE:\\n", baseline)
print()
print("AFTER:\\n", cleaned)'''),
        ("markdown",
            "Side by side as a DataFrame so you can copy it into a deck:"),
        ("code",
            '''pd.DataFrame({
    "version":   ["raw STT", "STT + post-process"],
    "transcript": [baseline, cleaned],
})'''),
        ("markdown",
            "## Recap\n\n"
            "Audio in production needs three things: **the right voice**, **smart chunking**, and "
            "**a post-process pass**. The post-process step is the cheapest 90% accuracy boost you "
            "will ever get. Next: `05-video-generation.ipynb`."),
    ]
