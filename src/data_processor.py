from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict
import re
import requests

SourceType = Literal["text", "pdf", "url"]
ChunkKey = Literal["intro", "opening", "meditation", "closure"]



@dataclass
class RawInput:
    source_type: SourceType
    text: Optional[str] = None
    pdf_path: Optional[str] = None
    url: Optional[str] = None



@dataclass
class ExtractedContent:
    source_type: SourceType
    content: str                 # full cleaned text
    chunks: Dict[ChunkKey, str]  # structured sections
    meta: dict



def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[TRUNCATED]"



def _chunk_meditation(text: str) -> Dict[ChunkKey, str]:
    """
    Simple heuristic chunker for your specific meditation format.
    If it can’t confidently find boundaries, it falls back safely.
    """
    t = text.strip()

    # Heuristic anchors (tweak these to match your script wording)
    intro_end = t.find("Let's begin.")
    opening_start = intro_end + len("Let's begin.") if intro_end != -1 else 0

    # Opening typically begins with “Find a comfortable position…”
    opening_anchor = t.find("Find a comfortable position")
    if opening_anchor != -1:
        opening_start = opening_anchor

    # Meditation typically starts at the first “I am …” affirmation
    meditation_start = t.find("I am ")
    if meditation_start == -1:
        meditation_start = t.find("I’m ")  # fallback (rare)

    # Closure anchor
    closure_start = t.find("Take a deep breath")
    if closure_start == -1:
        closure_start = t.find("Thank you for being here")

    # Build chunks with safe fallbacks
    intro = t[:opening_start].strip() if opening_start > 0 else ""
    opening = ""
    meditation = ""
    closure = ""

    if meditation_start != -1:
        opening = t[opening_start:meditation_start].strip()
    else:
        opening = t[opening_start:].strip()

    if meditation_start != -1 and closure_start != -1 and closure_start > meditation_start:
        meditation = t[meditation_start:closure_start].strip()
        closure = t[closure_start:].strip()
    elif meditation_start != -1:
        meditation = t[meditation_start:].strip()
    else:
        # If we can’t find affirmations, treat everything after intro/opening as meditation
        meditation = t[opening_start:].strip()

    return {
        "intro": intro,
        "opening": opening,
        "meditation": meditation,
        "closure": closure,
    }



def extract_text(raw: RawInput, max_chars: int = 50_000, timeout_s: int = 20) -> ExtractedContent:
    if raw.source_type == "text":
        if not raw.text or not raw.text.strip():
            raise ValueError("No text provided. Please paste a script or notes.")
        cleaned = _truncate(_clean_text(raw.text), max_chars)
        chunks = _chunk_meditation(cleaned)
        return ExtractedContent("text", cleaned, chunks, {"length": len(cleaned)})

    if raw.source_type == "url":
        if not raw.url or not raw.url.strip():
            raise ValueError("No URL provided. Please paste a valid article URL.")
        url = raw.url.strip()
        headers = {"User-Agent": "Ironhack-Podcast-Studio/1.0"}
        resp = requests.get(url, headers=headers, timeout=timeout_s)
        resp.raise_for_status()

        cleaned = _truncate(_clean_text(resp.text), max_chars)
        chunks = _chunk_meditation(cleaned)
        return ExtractedContent("url", cleaned, chunks, {"url": url, "length": len(cleaned)})

    if raw.source_type == "pdf":
        if not raw.pdf_path or not raw.pdf_path.strip():
            raise ValueError("No PDF file provided. Please upload a PDF.")
        pdf_path = raw.pdf_path.strip()

        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        joined = "\n".join((p.extract_text() or "") for p in reader.pages)

        cleaned = _clean_text(joined)
        if not cleaned:
            raise ValueError("This PDF has no extractable text (may be scanned image).")

        cleaned = _truncate(cleaned, max_chars)
        chunks = _chunk_meditation(cleaned)
        return ExtractedContent("pdf", cleaned, chunks, {"pdf_path": pdf_path, "length": len(cleaned)})

    raise ValueError(f"Unsupported source_type: {raw.source_type}")
from src.data_processor import RawInput, extract_text

if __name__ == "__main__":
    sample = """Welcome to the Ironhack Mindfulness Podcast.
Let's begin.

Find a comfortable position and gently close your eyes.
Bring your attention to your breath.

I am strong in body, mind, and spirit.
My intelligence grows with every experience I encounter.

Take a deep breath in... and out.
Thank you for being here today.
"""

    raw = RawInput(source_type="text", text=sample)
    out = extract_text(raw)

    print("✅ Test ran")
    print("SOURCE:", out.source_type)
    print("META:", out.meta)

    print("\n--- CHUNKS ---")
    for k, v in out.chunks.items():
        print(f"\n[{k.upper()}]\n{v}\n")
    

from pathlib import Path

def load_meditation_chunks_from_folder(input_folder: str = "1. Input") -> Dict[ChunkKey, str]:
    """
    Loads your existing script files from the folder:
    - Script Intro
    - Script Opening
    - Script Affirmations
    - Script Closing
    and returns them as chunks.
    """
    folder = Path(input_folder)

    intro_path = folder / "Script Intro"
    opening_path = folder / "Script Opening"
    meditation_path = folder / "Script Affirmations"
    closure_path = folder / "Script Closing"

    intro = intro_path.read_text(encoding="utf-8") if intro_path.exists() else ""
    opening = opening_path.read_text(encoding="utf-8") if opening_path.exists() else ""
    meditation = meditation_path.read_text(encoding="utf-8") if meditation_path.exists() else ""
    closure = closure_path.read_text(encoding="utf-8") if closure_path.exists() else ""

    return {
        "intro": _clean_text(intro),
        "opening": _clean_text(opening),
        "meditation": _clean_text(meditation),
        "closure": _clean_text(closure),
    }
