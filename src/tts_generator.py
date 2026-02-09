from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()



@dataclass
class AudioArtifact:
    path: str
    format: str
    voice: str
    model: str


def generate_audio(
    text: str,
    out_dir: str = "outputs",
    filename: str = "meditation.mp3",
    voice: str = "alloy",
    model: str = "gpt-4o-mini-tts",
    fmt: str = "mp3",
) -> AudioArtifact:
    """
    Convert text into an audio file using OpenAI TTS.
    Returns an AudioArtifact with the saved file path.
    """
    if not text or not text.strip():
        raise ValueError("No text provided for TTS generation.")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / filename

    audio = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text.strip(),
    )
    audio.write_to_file(str(out_path))

    return AudioArtifact(
        path=str(out_path),
        format=fmt,
        voice=voice,
        model=model,
    )
