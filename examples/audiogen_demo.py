"""AudioGen demo — generate sound effects from text prompts.

This is the part that makes mlx-audiocraft unique:
AudioGen on MLX hasn't been done before this package.

Run:
    python examples/audiogen_demo.py
"""

import time
import numpy as np
import soundfile as sf
from mlx_audiocraft import AudioGen

# ── Load model ────────────────────────────────────────────────────────────────
# Downloads ~3.6 GB total on first run (compression + LM checkpoints).
# Cached in ~/.cache/huggingface/ — subsequent runs are instant.
print("Loading AudioGen medium...")
t0 = time.time()
model = AudioGen.get_pretrained("facebook/audiogen-medium")
print(f"  Ready in {time.time()-t0:.1f}s")
print(f"  Sample rate: {model.sample_rate} Hz")
print(f"  Max duration: {model.max_duration}s")

# ── Generate sound effects ────────────────────────────────────────────────────
model.set_generation_params(
    duration=5,       # SFX are typically 3–10 seconds
    temperature=1.0,
    cfg_coef=3.0,
)

prompts = [
    "keyboard typing, subtle office ambience",
    "crowd applause, conference room",
    "notification chime, clean and bright",
    "rain on a window, thunder in the distance",
]

print(f"\nGenerating {len(prompts)} sound effects × {model.duration}s...")
t0 = time.time()
wavs = model.generate(prompts, progress=True)
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s ({model.duration / (elapsed/len(prompts)):.2f}x realtime per clip)")

# ── Save ──────────────────────────────────────────────────────────────────────
for i, (wav, prompt) in enumerate(zip(wavs, prompts)):
    arr = np.array(wav[0]).T          # [channels, samples] → [samples, channels]
    if arr.shape[1] == 1:
        arr = arr[:, 0]
    path = f"audiogen_output_{i}.wav"
    sf.write(path, arr, model.sample_rate)
    print(f"Saved: {path}  — \"{prompt}\"")
