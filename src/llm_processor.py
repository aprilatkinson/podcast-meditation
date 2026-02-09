from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from typing import Dict

client = OpenAI()


def build_meditation_script(chunks: Dict[str, str]) -> str:
    """
    Takes meditation chunks (intro, opening, meditation, closure)
    and lets the LLM refine them into one smooth guided meditation script.
    """

    prompt = f"""
You are a meditation writer.

Your task is to combine and gently refine the following meditation sections
into one smooth guided meditation script.

Rules:
- Keep the meaning and affirmations intact.
- Improve flow and transitions.
- Keep calm, supportive tone.
- Do NOT make it longer than necessary.
- Output only the final meditation script.

INTRO:
{chunks.get("intro", "")}

OPENING:
{chunks.get("opening", "")}

MEDITATION:
{chunks.get("meditation", "")}

CLOSURE:
{chunks.get("closure", "")}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text
