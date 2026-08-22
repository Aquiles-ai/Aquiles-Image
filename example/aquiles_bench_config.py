"""
Example: generate Aquiles-Image benchmark configs programmatically.

BenchConfig validates the parameters and writes JSON files that can be
executed later with:

    aquiles-image bench serve --config-bench <file>.json

`request_rate=None` means unlimited rate (burst / saturation mode).
A finite value sends requests following a Poisson process.

The bench does not choose the model; it measures whatever the running
server exposes. Launch the server first:

    aquiles-image serve --model "stabilityai/stable-diffusion-3.5-medium" --api-key dummy-api-key
"""

from aquilesimage.bench import BenchConfig, MixedProfile

MODEL = "stabilityai/stable-diffusion-3.5-medium"

BASE = dict(
    host="127.0.0.1",
    port=5500,
    api_key="dummy-api-key",
    num_prompts=100,
    warmup=3,
    timeout_s=600,
    seed=42,
    metadata={"expected_model": MODEL},
)


# Experiment B — continuous vs partial batching.
# Identical requests (same resolution, n=1, same num_prompts); only the
# arrival pattern changes, so any difference comes from how batches form.
b_full_batches = BenchConfig.uniform(
    size="1024x1024",
    label="sd35m-b-full-batches",
    request_rate=None,
    max_concurrency=8,
    **BASE,
)

b_sparse_batches = BenchConfig.uniform(
    size="1024x1024",
    label="sd35m-b-sparse-batches",
    request_rate=0.5,
    **BASE,
)


# Production pair — same load pattern, only the shape distribution differs.
prod_uniform = BenchConfig.uniform(
    size="1024x1024",
    label="sd35m-prod-uniform",
    request_rate=5.0,
    max_concurrency=8,
    **BASE,
)

prod_mixed = BenchConfig(
    label="sd35m-prod-mixed",
    request_rate=5.0,
    max_concurrency=8,
    profile=MixedProfile(
        sizes={
            "1024x1024": 0.6,
            "1536x1024": 0.25,
            "512x512": 0.15,
        },
        n=(1, 4),
    ),
    **BASE,
)


ALL = [b_full_batches, b_sparse_batches, prod_uniform, prod_mixed]

for config in ALL:
    for warning in config.collect_warnings():
        print(f"warning: {warning}")
    path = config.save_config(f"{config.label}.json")
    print(f"Bench config saved to: {path}")
