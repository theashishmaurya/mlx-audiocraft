"""Command-line entry points for mlx-audiocraft.

musicgen-mlx "upbeat cinematic track, piano, 120 BPM" -d 30 -o music.wav
audiogen-mlx "dog barking, park ambience" -d 5 -o sfx.wav
"""

import argparse
import time
from pathlib import Path


def _parse_common(parser: argparse.ArgumentParser):
    parser.add_argument("prompt", nargs="+", help="Text description(s) to generate from")
    parser.add_argument("-m", "--model", default=None, help="HuggingFace model ID or short name")
    parser.add_argument("-d", "--duration", type=float, default=None,
                        help="Duration in seconds (default: 30 for music, 5 for SFX)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output WAV path. Multiple prompts → output_0.wav, output_1.wav, ...")
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-coef", type=float, default=3.0)
    parser.add_argument("--no-open", action="store_true", help="Don't open file after generation")
    return parser


def _save_and_report(wavs, sample_rate: int, output: str, prompts: list, t0: float, no_open: bool):
    import numpy as np
    import soundfile as sf
    import subprocess
    import sys

    output_path = Path(output) if output else None
    saved = []

    for i, (wav, prompt) in enumerate(zip(wavs, prompts)):
        # wav is [channels, samples] — soundfile wants [samples, channels] or [samples]
        arr = np.array(wav)
        if arr.ndim == 3:
            arr = arr[0]          # remove batch dim if present
        arr = arr.T               # [channels, samples] → [samples, channels]
        if arr.shape[1] == 1:
            arr = arr[:, 0]       # mono

        if output_path is None:
            path = Path(f"output_{i}.wav") if len(prompts) > 1 else Path("output.wav")
        elif len(prompts) > 1:
            path = output_path.parent / f"{output_path.stem}_{i}{output_path.suffix}"
        else:
            path = output_path

        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), arr, sample_rate)
        saved.append(path)

        duration = arr.shape[0] / sample_rate
        elapsed = time.time() - t0
        ratio = duration / elapsed
        print(f"\n✓ {path}  ({duration:.1f}s audio in {elapsed:.1f}s — {ratio:.2f}x realtime)")
        print(f"  Prompt: \"{prompt}\"")

    if not no_open and len(saved) == 1:
        if sys.platform == "darwin":
            subprocess.run(["open", str(saved[0])], check=False)

    return saved


def musicgen_main():
    parser = argparse.ArgumentParser(
        description="Generate music from text on Apple Silicon via MLX")
    _parse_common(parser)
    args = parser.parse_args()

    from .models import MusicGen

    model_name = args.model or "facebook/musicgen-medium"
    duration = args.duration or 30.0

    print(f"Loading {model_name}...")
    t0 = time.time()
    model = MusicGen.get_pretrained(model_name)
    model.set_generation_params(
        duration=duration,
        top_k=args.top_k,
        temperature=args.temperature,
        cfg_coef=args.cfg_coef,
    )
    print(f"Model ready ({time.time()-t0:.1f}s). Generating {len(args.prompt)} clip(s)...")

    t0 = time.time()
    wavs = model.generate(args.prompt, progress=True)
    _save_and_report(wavs, model.sample_rate, args.output, args.prompt, t0, args.no_open)


def audiogen_main():
    parser = argparse.ArgumentParser(
        description="Generate sound effects from text on Apple Silicon via MLX")
    _parse_common(parser)
    args = parser.parse_args()

    from .models import AudioGen

    model_name = args.model or "facebook/audiogen-medium"
    duration = args.duration or 5.0

    print(f"Loading {model_name}...")
    t0 = time.time()
    model = AudioGen.get_pretrained(model_name)
    model.set_generation_params(
        duration=duration,
        top_k=args.top_k,
        temperature=args.temperature,
        cfg_coef=args.cfg_coef,
    )
    print(f"Model ready ({time.time()-t0:.1f}s). Generating {len(args.prompt)} clip(s)...")

    t0 = time.time()
    wavs = model.generate(args.prompt, progress=True)
    _save_and_report(wavs, model.sample_rate, args.output, args.prompt, t0, args.no_open)
