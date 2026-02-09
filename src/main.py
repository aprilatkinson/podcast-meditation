import gradio as gr

from src.data_processor import load_meditation_chunks_from_folder
from src.llm_processor import build_meditation_script
from src.tts_generator import generate_audio


def run_pipeline():
    # 1) Load chunks from 1. Input/
    chunks = load_meditation_chunks_from_folder("1. Input")

    # 2) LLM transforms chunks into one meditation script
    script = build_meditation_script(chunks)

    # 3) TTS generates an mp3 file
    audio_artifact = generate_audio(script)

    # Return both text and audio path for Gradio
    return script, audio_artifact.path


with gr.Blocks(title="Ironhack Mindfulness Podcast") as demo:
    gr.Markdown("# 🧘 Ironhack Mindfulness Podcast")
    gr.Markdown(
        "Click **Generate** to create a guided meditation audio from the scripts in `1. Input/`."
    )

    generate_btn = gr.Button("Generate Meditation Audio")

    script_out = gr.Textbox(label="Generated Meditation Script", lines=18)
    audio_out = gr.Audio(label="Meditation Audio (MP3)", type="filepath")

    generate_btn.click(fn=run_pipeline, inputs=[], outputs=[script_out, audio_out])

if __name__ == "__main__":
    demo.launch()
