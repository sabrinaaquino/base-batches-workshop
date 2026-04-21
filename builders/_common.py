"""Shared cell helpers used by every notebook builder.

Conventions:
    - One function = one notebook.
    - Each function returns a list of (cell_type, source) tuples.
    - Markdown is plain text. Code is plain Python (with %pip and %time magics allowed).
    - No em dashes anywhere. Use a regular hyphen, parens, or rephrase.
"""

from __future__ import annotations

from typing import List, Tuple

GH_USER = "sabrinaaquino"
GH_REPO = "base-batches-workshop"
BRANCH = "main"

Cell = Tuple[str, str]


def colab_badge(notebook: str) -> str:
    return (
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/github/{GH_USER}/{GH_REPO}/blob/{BRANCH}/notebooks/{notebook})"
    )


def header(notebook_filename: str, title: str, subtitle: str = "") -> str:
    """Standard header block: badge, title, byline, subtitle."""
    sub = f"\n\n{subtitle}" if subtitle else ""
    return (
        f"# {title}\n\n"
        f"{colab_badge(notebook_filename)}\n\n"
        f"_Sabrina Aquino, Venice AI / Base Batches 003_{sub}"
    )


def install_cell(extra: str = "") -> Cell:
    """Quiet pip install. Always includes the basics."""
    pkgs = "openai requests python-dotenv rich"
    if extra:
        pkgs = f"{pkgs} {extra}"
    return ("code", f"%pip install --quiet {pkgs}")


def setup_cell() -> Cell:
    """Reusable Venice client setup that works in env / .env / Colab."""
    return (
        "code",
        '''import os, time, json
from textwrap import shorten

# Pick up the API key from Colab secrets, environment, or interactive prompt
try:
    from google.colab import userdata  # type: ignore
    api_key = userdata.get("VENICE_API_KEY")
    os.environ["VENICE_API_KEY"] = api_key
except Exception:
    api_key = os.environ.get("VENICE_API_KEY")

if not api_key:
    from getpass import getpass
    api_key = getpass("Paste your Venice API key: ").strip()
    os.environ["VENICE_API_KEY"] = api_key

from openai import OpenAI
client = OpenAI(api_key=api_key, base_url="https://api.venice.ai/api/v1")
print("Connected to Venice")''',
    )
