"""Benchmark mlx-audiocraft generation speed on Apple Silicon.

Measures realtime ratio for each model on this machine.
Run once after install:

    python benchmarks/run_benchmarks.py

Results are saved to benchmarks/results.json for the README.
"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BENCHMARK_PROMPT = "dog barking in a park, birds chirping"
MUSIC_PROMPT = "upbeat cinematic track, piano, 120 BPM, no vocals"
DURATION = 5.0  # keep short for benchmarking

AUDIOGEN_MODELS = [
    "facebook/audiogen-medium",
]

MUSICGEN_MODELS = [
    "facebook/musicgen-small",
    "facebook/musicgen-medium",
]


def get_chip_info() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True)
        return result.stdout.strip() or platform.processor()
    except Exception:
        return platform.processor()


def get_memory_gb() -> float:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True)
        return int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        return 0.0


def benchmark_model(model_cls, model_id: str, prompt: str, duration: float) -> dict:
    print(f"\n  Loading {model_id}...", end=" ", flush=True)
    t_load = time.time()
    model = model_cls.get_pretrained(model_id)
    load_time = time.time() - t_load
    print(f"{load_time:.1f}s")

    model.set_generation_params(duration=duration)

    # Warm-up run (1 s) to populate MLX caches
    print(f"  Warm-up (1s)...", end=" ", flush=True)
    model.set_generation_params(duration=1.0)
    model.generate([prompt], progress=False)
    print("done")

    model.set_generation_params(duration=duration)
    print(f"  Benchmarking ({duration}s)...", end=" ", flush=True)
    t_gen = time.time()
    wavs = model.generate([prompt], progress=False)
    gen_time = time.time() - t_gen
    realtime_ratio = duration / gen_time

    print(f"{gen_time:.1f}s ({realtime_ratio:.2f}x realtime)")

    return {
        "model": model_id,
        "duration_s": duration,
        "generation_time_s": round(gen_time, 2),
        "realtime_ratio": round(realtime_ratio, 3),
        "load_time_s": round(load_time, 2),
        "sample_rate": model.sample_rate,
    }


def main():
    from mlx_audiocraft import AudioGen, MusicGen

    print("=" * 60)
    print("mlx-audiocraft benchmark")
    print("=" * 60)

    chip = get_chip_info()
    memory = get_memory_gb()
    print(f"Chip:   {chip}")
    print(f"Memory: {memory:.0f} GB unified")
    print(f"Python: {sys.version.split()[0]}")

    results = {
        "chip": chip,
        "memory_gb": round(memory, 1),
        "python": sys.version.split()[0],
        "audiogen": [],
        "musicgen": [],
    }

    print("\n── AudioGen (sound effects) ──")
    for model_id in AUDIOGEN_MODELS:
        try:
            r = benchmark_model(AudioGen, model_id, BENCHMARK_PROMPT, DURATION)
            results["audiogen"].append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            results["audiogen"].append({"model": model_id, "error": str(e)})

    print("\n── MusicGen (music) ──")
    for model_id in MUSICGEN_MODELS:
        try:
            r = benchmark_model(MusicGen, model_id, MUSIC_PROMPT, DURATION)
            results["musicgen"].append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            results["musicgen"].append({"model": model_id, "error": str(e)})

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")

    # Pretty print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for section, label in [("audiogen", "AudioGen"), ("musicgen", "MusicGen")]:
        print(f"\n{label}:")
        for r in results[section]:
            if "error" in r:
                print(f"  {r['model']:<40} ERROR: {r['error']}")
            else:
                rt = r["realtime_ratio"]
                faster = "faster" if rt >= 1.0 else "slower"
                print(f"  {r['model']:<40} {rt:.2f}x realtime ({faster} than real-time)")


if __name__ == "__main__":
    main()
