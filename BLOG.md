# I Ported AudioGen to Apple Silicon — Here's Everything I Learned

*How we built mlx-audiocraft: the first package that runs both MusicGen and AudioGen (text-to-sound-effects) on M-series Macs using MLX.*

![Cover illustration — sound waves flowing through a neural network](blog-images/cover.png)

> **Try it live:** [theashishmaurya.github.io/mlx-audiocraft](https://theashishmaurya.github.io/mlx-audiocraft/) — all audio samples below are playable there too.

---

## Why I built this

I was building a video production pipeline on my M4 Mac — AI-narrated videos, background music, the works. The music generation side was solved: ACE-Step and a couple of MLX-based MusicGen ports exist. But every time I needed a sound effect — a keyboard click, ambient office noise, a notification chime — I hit a wall.

**AudioGen** is Meta's model for text-to-sound-effects. It's part of their AudioCraft library and it's genuinely good. The problem: AudioCraft as a pip package is completely broken on Apple Silicon in 2026.

Try it yourself:

```bash
pip install audiocraft
```

You get this cascade of failures:

1. `av==11.0.0` tries to build against your system ffmpeg. ffmpeg 7.x renamed `AV_OPT_TYPE_CHANNEL_LAYOUT` to `AV_OPT_TYPE_CHLAYOUT`. The build explodes with 40 lines of C errors.
2. Even if you pin ffmpeg, `xformers` won't build because Apple clang doesn't support `-fopenmp`.
3. Even if you work around that, the whole thing runs on CPU at 0.03x realtime — painful for a 30-second audio clip.

So the choices were: run a Linux VM, rent a cloud GPU, or port it to MLX properly.

I chose the third option. This post is the full story of how that went.

---

## What is MLX and why does it matter for this

![Unified memory — CPU and GPU sharing the same memory pool on Apple Silicon](blog-images/mlx-unified-memory.png)

MLX is Apple's own machine learning framework, released in late 2023. The key thing to understand is the memory model.

On a normal computer, the CPU has its RAM and the GPU has its VRAM. Copying data between them takes time — this is why PyTorch has `.to("cuda")` and `.to("cpu")` calls everywhere. Every time you move a tensor between CPU and GPU you pay a transfer cost.

On Apple Silicon, **there is no separate VRAM**. The CPU, GPU, and Neural Engine all share the same physical memory pool. An array in MLX is already in a place where every compute unit can access it directly, with zero copying overhead.

MLX is designed from scratch for this architecture. It's lazy by default (computations don't happen until you `mx.eval()`), it has automatic differentiation, and it has Python bindings that feel a lot like NumPy crossed with PyTorch.

For our use case this means: once weights are loaded into MLX, every forward pass runs on the GPU with no transfer overhead. No `.to()` calls. No "device" concept at all.

---

## The AudioCraft architecture (what we're actually porting)

Before diving into the code, it's worth understanding what AudioCraft actually *is* under the hood, because this shapes everything about the port.

AudioCraft has two models you've probably heard of:

- **MusicGen** — generates music from text prompts
- **AudioGen** — generates sound effects and environmental audio from text prompts

Here's the surprising thing: **they use the exact same architecture**. Same transformer language model. Same audio codec. The only differences are what they were trained on and a few config values.

The pipeline for both looks like this:

![Generation pipeline — text flows through T5 encoder, transformer, and EnCodec decoder to produce audio](blog-images/pipeline.png)

```
Your text prompt
      │
      ▼
  T5 Encoder              ← reads your text, outputs a sequence of embeddings
      │                      runs on CPU (PyTorch, ~3s startup then cached)
      ▼
  Transformer LM           ← auto-regressive token generation
      │                      conditioned on T5 embeddings via cross-attention
      │                      runs on MLX (Apple GPU)
      ▼
  EnCodec Decoder          ← turns discrete tokens back into a waveform
                             runs on MLX (Apple GPU)
```

**T5** is Google's text encoder. It reads your prompt and produces a dense numerical representation of what you asked for. This runs in PyTorch on CPU — there's no MLX T5 port yet and it's not the bottleneck anyway.

**The Transformer LM** is where the heavy lifting happens. It's an auto-regressive decoder that predicts audio tokens one step at a time. Each step is conditioned on: (a) all previously generated tokens, and (b) the T5 embeddings of your prompt via cross-attention. This is what runs on the MLX GPU and where the speedup matters.

**EnCodec** is Meta's neural audio codec. Think of it like a lossy audio compressor, but the "compression" and "decompression" are learned neural networks. The encoder turns a waveform into a sequence of discrete integer tokens. The decoder turns those tokens back into audio. The LM generates the tokens; EnCodec decodes them.

### AudioGen vs MusicGen: what actually differs

| | MusicGen | AudioGen |
|--|----------|----------|
| Sample rate | 32 kHz | 16 kHz |
| Codebooks | 4–8 | 4 |
| Conditioning | Text + optional melody | Text only |
| Training data | Music | Sound effects & environments |
| Default clip length | 30s | 5–10s |
| CFG coefficient | 3.0 | 3.0 |

That's it. The transformer architecture is identical. The EnCodec architecture is identical. The only differences are the trained weights and a few config values. This observation is what made the port tractable.

---

## The existing work we built on

Before writing a single line, I looked for what already existed.

**[musicgen-mlx](https://github.com/andrade0/musicgen-mlx)** by Andrade Olivier is a solid MLX port of MusicGen. It has a full MLX implementation of the transformer LM, MLX EnCodec, HuggingFace weight loading with PyTorch→MLX conversion, and a `BaseGenModel` class with text conditioning and sliding-window generation.

What it **didn't** have was AudioGen. The plan became: fork musicgen-mlx, add AudioGen, publish as a new package with proper attribution.

---

## How we ported AudioGen: the actual code

![BaseGenModel is the parent — both MusicGen and AudioGen inherit from it](blog-images/inheritance.png)

I started by reading Meta's original AudioGen code to understand what overrides it makes on top of the base model:

```python
# Meta's original audiogen.py (simplified)
class AudioGen(BaseGenModel):
    def get_pretrained(name="facebook/audiogen-medium"):
        ...
    def set_generation_params(self, duration=5.0, ...):
        ...
    # _prepare_tokens_and_attributes: NOT overridden
    # _generate_tokens: NOT overridden
    # generate(): NOT overridden
```

The key insight: **AudioGen overrides almost nothing.** `BaseGenModel` already implements text-only conditioning and sliding-window generation — exactly what AudioGen needs. MusicGen overrides `_prepare_tokens_and_attributes` for melody conditioning; AudioGen doesn't need that.

Our complete `audiogen.py` — about 90 lines including docstrings:

```python
class AudioGen(BaseGenModel):
    """AudioGen: text-to-sound-effects on Apple Silicon via MLX."""

    def __init__(self, name, compression_model, lm, max_duration=None):
        super().__init__(name, compression_model, lm, max_duration)
        self.set_generation_params(duration=5)  # SFX default: 5s, not 30s

    @staticmethod
    def get_pretrained(name="facebook/audiogen-medium") -> "AudioGen":
        from .loaders import load_compression_model, load_lm_model
        lm = load_lm_model(name)
        compression_model = load_compression_model(name)
        max_duration = None
        if hasattr(lm, 'cfg') and hasattr(lm.cfg, 'dataset'):
            max_duration = getattr(lm.cfg.dataset, 'segment_duration', None)
        return AudioGen(name, compression_model, lm, max_duration)

    def set_generation_params(self, use_sampling=True, top_k=250, top_p=0.0,
                               temperature=1.0, duration=5.0, cfg_coef=3.0,
                               two_step_cfg=False, extend_stride=3.0):
        assert extend_stride < self.max_duration
        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            "use_sampling": use_sampling, "temp": temperature,
            "top_k": top_k, "top_p": top_p, "cfg_coef": cfg_coef,
            "two_step_cfg": two_step_cfg, "cfg_coef_beta": None,
        }

    # Everything else inherited from BaseGenModel — no overrides needed.
```

---

## Results — listen for yourself

![Audio waveforms — the final output: clean, natural-sounding audio generated entirely on-device](blog-images/results-audio.png)

All clips below were generated locally on an M4 Mac. No cloud API. No server. Just `pip install mlx-audiocraft`.

### Sound Effects (AudioGen) — 5s each, 16 kHz

**Mechanical keyboard typing, quiet office ambience**
<audio controls src="docs/audio/sfx_keyboard.wav" style="width:100%;margin:8px 0 16px"></audio>

**Rain falling on a metal roof, distant rolling thunder**
<audio controls src="docs/audio/sfx_rain.wav" style="width:100%;margin:8px 0 16px"></audio>

**Crowd applause in a conference room**
<audio controls src="docs/audio/sfx_applause.wav" style="width:100%;margin:8px 0 16px"></audio>

**Notification chime, clean bright tone**
<audio controls src="docs/audio/sfx_chime.wav" style="width:100%;margin:8px 0 16px"></audio>

**Coffee machine brewing, kitchen background ambience**
<audio controls src="docs/audio/sfx_coffee.wav" style="width:100%;margin:8px 0 16px"></audio>

### Music (MusicGen) — 15s each, 32 kHz

**"upbeat cinematic tech promo, clean piano with electronic pads, building momentum, 120 BPM, no vocals"**
<audio controls src="docs/audio/music_cinematic.wav" style="width:100%;margin:8px 0 16px"></audio>

**"calm lo-fi beat, soft warm piano, subtle vinyl crackle, mellow bass, 80 BPM, no vocals"**
<audio controls src="docs/audio/music_lofi.wav" style="width:100%;margin:8px 0 16px"></audio>

> Note: audio tags render on most blog platforms (Hashnode, your own site). On GitHub's markdown renderer, click the links above to open the [live demo site](https://theashishmaurya.github.io/mlx-audiocraft/) where all players are embedded.

---

### Generation speed (M4 Max, 64 GB)

| Model | Prompt duration | Wall time | Realtime ratio |
|-------|----------------|-----------|----------------|
| `audiogen-medium` | 5s SFX | ~29s | 0.17x |
| `musicgen-small` | 10s music | ~15s | 0.68x |
| `musicgen-medium` | 10s music | ~17s | 0.60x |
| `musicgen-large` | 10s music | ~35s | 0.29x |

For context: CPU-only AudioCraft takes several minutes for a 5-second clip. The MLX port bottleneck is the auto-regressive transformer loop itself — 250 sequential forward passes for a 5s clip — not the hardware.

---

## Getting the packaging right

One thing I wanted to do from the start was make this a proper package that anyone could install with a single command:

```toml
[project]
name = "mlx-audiocraft"
dependencies = [
    "mlx>=0.17",
    "torch",          # CPU-only, for T5 text encoder
    "transformers",
    "huggingface-hub",
    "omegaconf",
    "soundfile",
]

[project.scripts]
musicgen-mlx = "mlx_audiocraft.cli:musicgen_main"
audiogen-mlx = "mlx_audiocraft.cli:audiogen_main"
```

Publishing uses GitHub Actions OIDC trusted publishing — no API token stored anywhere, triggered automatically on GitHub releases:

```yaml
on:
  release:
    types: [published]
jobs:
  publish:
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - run: pip install build && python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

---

## What I learned

**The inheritance insight was the unlock.** Once I understood that `BaseGenModel` already implemented everything AudioGen needed, the port went from "big ML project" to "add one file and update two imports."

**MLX weight loading is the fiddly part.** Getting 847 weight tensors to load with the right shapes and transpositions took careful reading of the conversion code. The transformer math is nearly identical to PyTorch — the weight format translation is where the complexity lives.

**Packaging matters as much as the code.** Writing `audiogen.py` took an afternoon. Getting `pip install mlx-audiocraft` to work cleanly with proper entry points, pyproject.toml, OIDC publishing, and a clear README took about as long again.

**Apple Silicon is a real ML platform now.** M4 Max at 0.17x realtime for a 1.5B-parameter model, offline, on a laptop. Not H100 speed, but for inference workloads that don't need real-time generation it's more than good enough — and the unified memory model is genuinely nicer to work with than juggling CUDA devices.

---

## What's next

- Stereo AudioGen
- Streaming generation API
- Fine-tuning support (MLX supports training, not just inference)
- Community benchmark results — run `python benchmarks/run_benchmarks.py` and open a PR

---

## Installation

```bash
pip install mlx-audiocraft
```

**Requirements:** macOS 13+, Apple Silicon (M1/M2/M3/M4), Python 3.10+

```bash
audiogen-mlx "keyboard typing, office ambience" -d 5 -o sfx.wav
musicgen-mlx "upbeat cinematic, piano, 120 BPM, no vocals" -d 30 -o music.wav
```

Source: **[github.com/theashishmaurya/mlx-audiocraft](https://github.com/theashishmaurya/mlx-audiocraft)**  
Live demo: **[theashishmaurya.github.io/mlx-audiocraft](https://theashishmaurya.github.io/mlx-audiocraft/)**

---

## Attribution

The MusicGen MLX engine is based on **[musicgen-mlx](https://github.com/andrade0/musicgen-mlx)** by Andrade Olivier — excellent work, go star that repo.

The original AudioCraft library is by **[Meta AI Research](https://github.com/facebookresearch/audiocraft)**. Pre-trained weights (`facebook/audiogen-medium`, `facebook/musicgen-*`) are under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — research and personal use only.

The AudioGen MLX port is original work in this repo, MIT licensed.

---

*Built by [Ashish Maurya](https://github.com/theashishmaurya). Questions or contributions welcome — open an issue or PR.*
