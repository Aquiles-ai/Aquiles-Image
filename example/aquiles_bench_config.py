"""
Example: generate Aquiles-Image benchmark configs programmatically.

BenchConfig validates the parameters and writes a JSON file that can be
executed later with:

    aquiles-image bench serve --config-bench <file>.json

`request_rate=None` means unlimited rate (burst / saturation mode).
A finite value sends requests following a Poisson process.
"""

import json

from aquilesimage.bench import BenchConfig, MixedProfile

config_eager = BenchConfig.uniform(
    size="1024x1024",
    host="127.0.0.1",
    port=5500,
    num_prompts=100,
    request_rate=5.0,
    warmup=3,
    seed=42,
    label="eager-baseline",
)

path = config_eager.save_config("bench_eager.json")
print(f"Bench config saved to: {path}")

config_piecewise = BenchConfig(
    host="127.0.0.1",
    port=5500,
    api_key="dummy-api-key",
    num_prompts=200,
    max_concurrency=8,
    timeout_s=600,
    profile=MixedProfile(
        sizes={
            "1024x1024": 0.6,
            "1536x1024": 0.25,
            "512x512": 0.15,
        },
        n=(1, 4),
    ),
    warmup=3,
    seed=42,
    metadata={"commit": "dev", "gpu": "H100"},
    label="piecewise-mixed",
)

for warning in config_piecewise.collect_warnings():
    print(f"warning: {warning}")

path = config_piecewise.save_config("bench_piecewise.json")
print(f"Bench config saved to: {path}")

with open(path, "r", encoding="utf-8") as f:
    reloaded = BenchConfig.model_validate_json(f.read())

print(f"Reloaded profile: {reloaded.profile.type}, "
      f"sizes: {list(reloaded.profile.sizes)}, "
      f"label: {reloaded.label}")
