"""Build all workshop notebooks from the per-notebook builder modules.

Run from the repo root:
    python build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

from builders import (
    nb_00_quickstart,
    nb_01_chat,
    nb_02_embeddings,
    nb_03_image,
    nb_04_audio,
    nb_05_video,
    nb_06_characters,
    nb_07_x402,
    nb_08_e2ee,
)


BUILDERS = [
    ("00-quickstart.ipynb",            nb_00_quickstart),
    ("01-chat-completions.ipynb",      nb_01_chat),
    ("02-embeddings-and-rag.ipynb",    nb_02_embeddings),
    ("03-image-generation.ipynb",      nb_03_image),
    ("04-audio-tts-stt.ipynb",         nb_04_audio),
    ("05-video-generation.ipynb",      nb_05_video),
    ("06-characters.ipynb",            nb_06_characters),
    ("07-x402-wallet-payments.ipynb",  nb_07_x402),
    ("08-e2ee-encryption.ipynb",       nb_08_e2ee),
]


def to_cell(cell_type: str, source: str) -> dict:
    """Build one Jupyter cell from a (type, source) tuple."""
    lines = source.splitlines(keepends=True)
    base = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def to_notebook(cells: list[tuple[str, str]]) -> dict:
    return {
        "cells": [to_cell(t, s) for t, s in cells],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebooks_dir = Path(__file__).parent / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)

    for filename, module in BUILDERS:
        cells = module.cells()
        nb = to_notebook(cells)
        out = notebooks_dir / filename
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  wrote {out.relative_to(Path(__file__).parent)}  ({len(cells)} cells)")

    print(f"\nDone. {len(BUILDERS)} notebooks generated.")


if __name__ == "__main__":
    main()
