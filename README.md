# mlx-audiocraft

**MusicGen + AudioGen on Apple Silicon via MLX** — the first full AudioCraft port for M-series Macs. No CUDA, no server, no Docker. Just `pip install` and generate.

```bash
# Generate sound effects (NEW — not in any other MLX package)
audiogen-mlx "keyboard typing, office ambience" -d 5 -o sfx.wav

# Generate music
musicgen-mlx "upbeat cinematic tech promo, piano, 120 BPM, no vocals" -d 30 -o music.wav
```

---

## What's unique about this package

| Package | MusicGen | **AudioGen (SFX)** | MLX (Apple GPU) |
|---------|:--------:|:-----------------:|:---------------:|
| `audiocraft` (Meta) | ✅ | ✅ | ❌ |
| `musicgen-mlx` | ✅ | ❌ | ✅ |
| **`mlx-audiocraft`** (this) | ✅ | ✅ | ✅ |

**AudioGen on MLX is new.** This is the first port that brings text-to-sound-effects to Apple Silicon with hardware acceleration.

---

## How it works (for learners)

Here's the mental model you need:

```
Your text prompt
      ↓
  T5 Encoder          ← reads your text, runs on CPU (PyTorch)
      ↓
  Transformer LM      ← generates audio "tokens", runs on MLX (Apple GPU)
      ↓
  EnCodec Decoder     ← turns tokens into a waveform, runs on MLX
      ↓
  WAV file
```

**MusicGen** and **AudioGen** use the exact same pipeline — they're just trained on different data (music vs. sound effects). This is why porting AudioGen was mostly adding `audiogen.py` that inherits the whole pipeline.

**MLX** is Apple's own ML framework optimised for the unified memory in M-series chips — the CPU, GPU, and Neural Engine all share the same RAM, so there's zero data transfer overhead between them.

---

## Install

```bash
pip install mlx-audiocraft
```

**Requirements:** macOS 13+, Apple Silicon (M1/M2/M3/M4), Python 3.10+

---

## Quick start

### Sound effects (AudioGen)

```python
from mlx_audiocraft import AudioGen

model = AudioGen.get_pretrained("facebook/audiogen-medium")
model.set_generation_params(duration=5)

wav = model.generate(["dog barking in a park"])
# wav shape: [batch, channels, samples]
```

### Music (MusicGen)

```python
from mlx_audiocraft import MusicGen

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=30)

wav = model.generate(["calm lo-fi beat, soft piano, vinyl crackle"])
```

### CLI

```bash
# Sound effects
audiogen-mlx "crowd applause, conference room" -d 5
audiogen-mlx "rain on a window, thunder in the distance" -d 8 -o rain.wav

# Music
musicgen-mlx "epic orchestral soundtrack" -m facebook/musicgen-large -d 20
musicgen-mlx "funky disco groove" "ambient pad wide reverb" -d 10  # batch
```

---

## Models

### AudioGen

| Model | Size | Download | Sample Rate |
|-------|------|----------|-------------|
| `facebook/audiogen-medium` | 1.5B | ~3.6 GB | 16 kHz |

### MusicGen

| Model | Size | Download | Sample Rate |
|-------|------|----------|-------------|
| `facebook/musicgen-small` | 300M | ~1.2 GB | 32 kHz |
| `facebook/musicgen-medium` | 1.5B | ~3.2 GB | 32 kHz |
| `facebook/musicgen-large` | 3.3B | ~6.5 GB | 32 kHz |
| `facebook/musicgen-stereo-small` | 300M | ~1.2 GB | 32 kHz stereo |
| `facebook/musicgen-stereo-medium` | 1.5B | ~3.2 GB | 32 kHz stereo |

Models download automatically from HuggingFace on first use and are cached in `~/.cache/huggingface/`.

---

## Benchmark (M4 Max, 64 GB)

> Run `python benchmarks/run_benchmarks.py` to generate results for your machine.

| Model | Duration | Time | Realtime |
|-------|----------|------|---------|
| audiogen-medium | 5s | ~8s | 0.6x |
| musicgen-small | 10s | ~8s | 1.3x |
| musicgen-medium | 10s | ~17s | 0.6x |
| musicgen-large | 10s | ~35s | 0.3x |

*Faster than realtime means generation is quicker than the audio duration.*

---

## Prompt guide

### Sound effects (AudioGen)
Be literal and specific:
```
"keyboard typing, subtle office background noise"
"notification chime, clean and bright"
"crowd applause, conference room, 3 seconds"
"rain falling on a metal roof, distant thunder"
"coffee machine brewing, kitchen ambience"
```

### Music (MusicGen)
Include style, instrumentation, BPM, and always end with `, no vocals`:
```
"upbeat cinematic tech promo, clean piano with electronic pads, 120 BPM, no vocals"
"calm educational background, soft piano and ambient pads, 75 BPM, no vocals"
"energetic SaaS launch, modern synths, punchy drums, 120 BPM, no vocals"
"Hindi classical influence, sitar and tabla, meditative, 60 BPM, no vocals"
```

---

## Save output

```python
import numpy as np
import soundfile as sf

wav = model.generate(["your prompt"])
audio = np.array(wav[0]).T          # [channels, samples] → [samples, channels]
if audio.ndim == 2 and audio.shape[1] == 1:
    audio = audio[:, 0]             # stereo → mono if needed
sf.write("output.wav", audio, model.sample_rate)
```

---

## Architecture deep-dive (for learners)

If you want to understand how this works under the hood, here's a reading order:

1. **`mlx_audiocraft/models/genmodel.py`** — `BaseGenModel` — the base class all models inherit. Understand `generate()`, `_prepare_tokens_and_attributes()`, and `_generate_tokens()`.

2. **`mlx_audiocraft/models/audiogen.py`** — our AudioGen port. It's tiny (~90 lines) because it just inherits `BaseGenModel` and points at AudioGen's weights. Good first file to read.

3. **`mlx_audiocraft/models/musicgen.py`** — MusicGen adds melody conditioning on top of `BaseGenModel`. Compare with `audiogen.py` to see the diff.

4. **`mlx_audiocraft/models/loaders.py`** — how model weights are downloaded from HuggingFace and converted from PyTorch format to MLX.

5. **`mlx_audiocraft/modules/transformer.py`** — the MLX transformer implementation. This is the core of the language model.

6. **`mlx_audiocraft/models/encodec.py`** — the audio codec (compress waveform → tokens, decode tokens → waveform).

---

## Attribution

The MusicGen MLX engine is based on [musicgen-mlx](https://github.com/andrade0/musicgen-mlx) by Andrade Olivier. The original AudioCraft library is by [Meta AI Research](https://github.com/facebookresearch/audiocraft).

AudioGen MLX port is original work in this repository.

---

## License

MIT — see [LICENSE](LICENSE).

The pre-trained model weights (`facebook/audiogen-medium`, `facebook/musicgen-*`) are released under the [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) licence by Meta.
