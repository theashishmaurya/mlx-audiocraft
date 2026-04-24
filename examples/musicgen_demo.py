"""MusicGen demo — generate background music from a text prompt.

Run:
    python examples/musicgen_demo.py
"""

import time
import numpy as np
import soundfile as sf
from mlx_audiocraft import MusicGen

# ── Load model ────────────────────────────────────────────────────────────────
# First run downloads ~1.2 GB (small) or ~3.2 GB (medium) to ~/.cache/huggingface/
# Subsequent runs are instant.
print("Loading MusicGen small...")
t0 = time.time()
model = MusicGen.get_pretrained("facebook/musicgen-small")
print(f"  Ready in {time.time()-t0:.1f}s")

# ── Set generation params ─────────────────────────────────────────────────────
model.set_generation_params(
    duration=10,          # seconds of audio to generate
    temperature=1.0,      # 1.0 = model default. Try 0.8 for more conservative output.
    cfg_coef=3.0,         # classifier-free guidance strength
)

# ── Generate ──────────────────────────────────────────────────────────────────
prompts = [
    "upbeat cinematic tech promo, clean piano with electronic pads, 120 BPM, no vocals",
    "calm educational background, soft piano and ambient pads, 75 BPM, no vocals",
]

print(f"\nGenerating {len(prompts)} clips × {model.duration}s...")
t0 = time.time()
wavs = model.generate(prompts, progress=True)
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s ({model.duration / (elapsed/len(prompts)):.2f}x realtime)")

# ── Save ──────────────────────────────────────────────────────────────────────
for i, (wav, prompt) in enumerate(zip(wavs, prompts)):
    arr = np.array(wav[0]).T          # [channels, samples] → [samples, channels]
    if arr.shape[1] == 1:
        arr = arr[:, 0]
    path = f"musicgen_output_{i}.wav"
    sf.write(path, arr, model.sample_rate)
    print(f"Saved: {path}  — \"{prompt[:50]}...\"")
