from src.data_processor import RawInput, extract_text

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
