import typer
from pathlib import Path
from typing import Optional
import sys
import os

app = typer.Typer()

bench_app = typer.Typer(help="Benchmark utilities for Aquiles-Image servers.")

from aquilesimage.cli.branding import (
    config_view,
    download_status,
    err,
    hint,
    info,
    model_not_found,
    ok,
    startup_banner,
    validation_error,
    warn,
)

@app.command("hello")
def greet(name: str = typer.Option(..., help="Name to greet")):
    ok(f"Hello, {name}!")


@app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", help="Host where Aquiles-Image will be executed"),
    port: int = typer.Option(5500, help="Port where Aquiles-Image will be executed"),
    model: Optional[str] = typer.Option(None, help="The model to use for image generation."),
    api_key: Optional[str] = typer.Option(None, help="API KEY enabled to make requests"),
    max_concurrent_infer: Optional[int] = typer.Option(None, help="Maximum concurrent inferences"),
    block_request: Optional[bool] = typer.Option(None, "--block-request/--no-block-request", help="Block requests during maximum concurrent inferences"),
    force: bool = typer.Option(False, "--force", is_flag=True, help="Force overwrite existing configuration"),
    no_load_model: bool = typer.Option(False, "--no-load-model", is_flag=True, help="Not loading the model simply allows for faster development without having to load the model constantly."),
    set_steps: Optional[int] = typer.Option(None, help="Set the steps that the model will use"),
    auto_pipeline: Optional[bool] = typer.Option(None, "--auto-pipeline/--no-auto-pipeline", help="Load a model that is compatible with diffusers but is not mentioned in the Aquiles-Image documentation"),
    device_map: Optional[str] = typer.Option(None, help="Device map option in which to load the model (Only compatible with diffusers/FLUX.2-dev-bnb-4bit)"),
    dist_inference: Optional[bool] = typer.Option(None, "--dist-inference/--no-dist-inference", help="Use distributed inference"),
    max_batch_size: Optional[int] = typer.Option(None, help="Maximum number of requests to group in a single batch for inference"),
    batch_timeout: Optional[float] = typer.Option(None, help="Maximum time (in seconds) to wait before processing a batch even if not full"),
    worker_sleep: Optional[float] = typer.Option(None, help="Time (in seconds) the worker sleeps between checking for new batch requests"),
    auto_pipeline_type: Optional[str] = typer.Option(None, help="You must specify the AutoPipeline type with '--auto-pipeline-type t2i (Text to Image) or i2i (Image to Image)'"),
    username: Optional[str] = typer.Option(None, help="Username for the playground (enables playground if set along with --password)"),
    password: Optional[str] = typer.Option(None, help="Password for the playground (enables playground if set along with --username)"),
    guidance_scale: Optional[float] = typer.Option(None, help="Guidance scale value for image generation"),
    seed: Optional[int] = typer.Option(None, help="Seed for reproducible image generation"),
    load_lora: Optional[bool] = typer.Option(None, "--load-lora/--no-load-lora", help="Enable LoRA loading from a config file"),
    lora_config: Optional[str] = typer.Option(None, "--lora-config", help="Path to the LoRA config JSON file (relative or absolute)"),
    mode: Optional[str] = typer.Option(None,"--mode", help="Compilation mode: 'eager' applies diffusers' base optimizations only (default behavior); 'piecewise' additionally compiles the pipeline per-shape via warmup compilation."),
    cpu_offload: Optional[bool] = typer.Option(None, "--cpu-offload/--no-cpu-offload", help="Enable CPU offloading for SD3/SD3.5 pipelines (StableDiffusion3Pipeline) to reduce VRAM usage. Other pipelines ignore this option.")
):
    """Start the Aquiles-Image server."""

    if auto_pipeline_type is not None and auto_pipeline_type not in ("t2i", "i2i"):
        err("--auto-pipeline-type must be 't2i' or 'i2i'")
        raise typer.Exit(code=1)

    if load_lora and lora_config is None:
        err("--load-lora requires --lora-config to be specified")
        raise typer.Exit(code=1)

    try:
        from aquilesimage.configs import (
            load_config_cli,
            configs_image_serve,
            config_file_exists,
            create_basic_config_if_not_exists
        )
        from aquilesimage.models import ConfigsServe
        from aquilesimage.utils import _build_allowed_users
    except ImportError as e:
        err(f"Error importing configuration modules: {e}")
        raise typer.Exit(code=1)

    config_exists = config_file_exists()

    if not config_exists:
        if model:
            info(f"No configuration found. Creating basic configuration with model: {model}")
            try:
                if no_load_model:
                    create_basic_config_if_not_exists(model, False)
                else:
                    create_basic_config_if_not_exists(model)
            except Exception as e:
                err(f"Error creating basic configuration: {e}")
                raise typer.Exit(code=1)
        else:
            try:
                create_basic_config_if_not_exists()
            except Exception as e:
                err(f"Error creating default configuration: {e}")
                raise typer.Exit(code=1)

    try:
        conf = load_config_cli()
    except Exception as e:
        err(f"Error loading configuration: {e}")
        raise typer.Exit(code=1)

    model_from_config = conf.get("model")
    final_model = model or model_from_config

    if not final_model:
        err("No model specified. Use --model parameter or configure one first.")
        hint("aquiles-image serve --model <model-id>")
        raise typer.Exit(code=1)

    config_needs_update = any([
        model is not None,
        api_key is not None,
        max_concurrent_infer is not None,
        block_request is not None,
        no_load_model,
        set_steps is not None,
        auto_pipeline is not None,
        device_map is not None,
        dist_inference is not None,
        max_batch_size is not None,
        batch_timeout is not None,
        worker_sleep is not None,
        auto_pipeline_type is not None,
        username is not None,
        password is not None,
        guidance_scale is not None,
        seed is not None,
        load_lora is not None,
        lora_config is not None,
        mode is not None,
        cpu_offload is not None
    ])

    if config_needs_update:
        try:
            existing_api_keys = conf.get("allows_api_keys", [""])

            if api_key:
                existing_api_keys = [api_key] if api_key not in existing_api_keys else existing_api_keys

            updated_conf = ConfigsServe(
                model=final_model,
                allows_api_keys=existing_api_keys,
                max_concurrent_infer=max_concurrent_infer if max_concurrent_infer is not None else conf.get("max_concurrent_infer"),
                block_request=block_request if block_request is not None else conf.get("block_request"),
                load_model=False if no_load_model else conf.get("load_model", True),
                steps_n=set_steps if set_steps is not None else conf.get("steps_n"),
                auto_pipeline=auto_pipeline if auto_pipeline is not None else conf.get("auto_pipeline"),
                device_map=device_map if device_map is not None else conf.get("device_map"),
                dist_inference=dist_inference if dist_inference is not None else conf.get("dist_inference"),
                max_batch_size=max_batch_size if max_batch_size is not None else conf.get("max_batch_size"),
                batch_timeout=batch_timeout if batch_timeout is not None else conf.get("batch_timeout"),
                worker_sleep=worker_sleep if worker_sleep is not None else conf.get("worker_sleep"),
                auto_pipeline_mode=auto_pipeline_type if auto_pipeline_type is not None else conf.get("auto_pipeline_mode"),
                guidance_scale=guidance_scale if guidance_scale is not None else conf.get("guidance_scale"),
                seed=seed if seed is not None else conf.get("seed"),
                allows_users=_build_allowed_users(username, password, conf),
                load_lora=load_lora if load_lora is not None else conf.get("load_lora"),
                lora_config_path=lora_config if lora_config is not None else conf.get("lora_config_path"),
                mode=mode if mode is not None else conf.get("mode", "eager"),
                cpu_offload=cpu_offload if cpu_offload is not None else conf.get("cpu_offload")
            )

            configs_image_serve(updated_conf, force=True)
            ok("Configuration updated successfully.")

        except Exception as e:
            err(f"Error updating configuration: {e}")
            raise typer.Exit(code=1)

    try:
        import uvicorn
    except ImportError as e:
        err(f"Error importing uvicorn: {e}")
        raise typer.Exit(code=1)

    try:
        from aquilesimage.main import app as fastapi_app
    except TypeError as e:
        err("Error loading application (Pydantic validation). This might be caused by invalid configuration values.")
        hint("aquiles-image configs --reset")
        raise typer.Exit(code=1)
    except Exception as e:
        err(f"Error loading application: {e}")
        raise typer.Exit(code=1)

    startup_banner(host, port, final_model, len(conf), dev_mode=no_load_model)

    try:
        from aquilesimage.utils.rich_logging import uvicorn_log_config
        uvicorn.run(fastapi_app, host=host, port=port, log_config=uvicorn_log_config())
    except KeyboardInterrupt:
        info("Server stopped by user.")
    except Exception as e:
        err(f"Error starting server: {e}")
        raise typer.Exit(code=1)


@app.command("configs")
def configs(
    show: bool = typer.Option(False, "--show", is_flag=True, help="Show current configuration"),
    reset: bool = typer.Option(False, "--reset", is_flag=True, help="Reset configuration to defaults"),
):
    """Manage Aquiles-Image configuration."""
    try:
        from aquilesimage.configs import load_config_cli, clear_config_cache
        import json
    except ImportError as e:
        err(f"Error importing required modules: {e}")
        raise typer.Exit(code=1)

    if reset:
        if typer.confirm("Are you sure you want to reset the configuration?"):
            try:
                clear_config_cache()
                ok("Configuration reset successfully.")
            except Exception as e:
                err(f"Error resetting configuration: {e}")
        return

    if show:
        try:
            conf = load_config_cli()
            if conf:
                info("Current configuration:")
                config_view(conf)
            else:
                warn("No configuration found.")
        except Exception as e:
            err(f"Error loading configuration: {e}")
        return

    typer.echo(typer.get_current_context().get_help())


@app.command("validate")
def validate():
    """Validate current configuration."""
    try:
        from aquilesimage.configs import load_config_cli
        from aquilesimage.models import ConfigsServe
    except ImportError as e:
        err(f"Error importing required modules: {e}")
        raise typer.Exit(code=1)

    try:
        conf = load_config_cli()

        if not conf:
            err("No configuration found.")
            raise typer.Exit(code=1)

        ConfigsServe(**conf)
        ok("Configuration is valid.")

    except typer.Exit:
        raise
    except Exception as e:
        validation_error(str(e))
        hint("aquiles-image configs --reset")
        raise typer.Exit(code=1)

@app.command("gguf-download")
def gguf_download(
    model_id: str = typer.Option(..., help="Model ID from the Aquiles GGUF registry (e.g. 'flux1-dev-q4k')"),
):
    """Download a GGUF model from the Aquiles registry."""
    try:
        from aquilesimage.utils.gguf_utils import verify_registry, AQUILES_GGUF_REGISTRY
        from huggingface_hub import hf_hub_download
        import json
    except ImportError as e:
        err(f"Error importing required modules: {e}")
        raise typer.Exit(code=1)
 
    try:
        verify_registry()
    except Exception as e:
        err(f"Error verifying registry: {e}")
        raise typer.Exit(code=1)
 
    try:
        with open(AQUILES_GGUF_REGISTRY, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except Exception as e:
        err(f"Error reading registry: {e}")
        raise typer.Exit(code=1)
 
    if model_id not in registry:
        model_not_found(model_id, registry.keys())
        raise typer.Exit(code=1)

    entry = registry[model_id]

    info(f"Downloading '{model_id}' from {entry['gguf_repo']}/{entry['gguf_file']}")
    try:
        with download_status("downloading model weights"):
            path = hf_hub_download(
                repo_id=entry["gguf_repo"],
                filename=entry["gguf_file"],
            )
        ok(f"Downloaded to: {path}")
    except Exception as e:
        err(f"Error downloading GGUF file: {e}")
        raise typer.Exit(code=1)
 
 
@app.command("gguf-update")
def gguf_update():
    """Update the local Aquiles GGUF registry from HuggingFace."""
    try:
        from aquilesimage.utils.gguf_utils import update_registry
    except ImportError as e:
        err(f"Error importing required modules: {e}")
        raise typer.Exit(code=1)
 
    info("Updating GGUF registry...")
    try:
        update_registry()
        ok("Registry updated successfully.")
    except Exception as e:
        err(f"Error updating registry: {e}")
        raise typer.Exit(code=1)

@bench_app.command("serve")
def bench_serve(
    config_bench: Path = typer.Option(
        ..., "--config-bench",
        help="Path to the bench config JSON generated with BenchConfig.save_config()"
    ),
    host: Optional[str] = typer.Option(None, help="Override target server host"),
    port: Optional[int] = typer.Option(None, help="Override target server port"),
    api_key: Optional[str] = typer.Option(None, help="Override API key"),
    label: Optional[str] = typer.Option(None, help="Override run label"),
    num_prompts: Optional[int] = typer.Option(None, help="Override number of prompts"),
    result_dir: Optional[str] = typer.Option(None, help="Override directory for result files"),
):
    """Run an online serving benchmark against a running Aquiles-Image server."""
    try:
        from aquilesimage.bench.models import BenchConfig
        from aquilesimage.bench.runner import run_bench
    except ImportError as e:
        err(f"Error importing bench modules: {e}")
        hint("pip install 'aquiles-image[bench]'")
        raise typer.Exit(code=1)

    if not config_bench.is_file():
        err(f"Bench config not found: {config_bench}")
        raise typer.Exit(code=1)

    try:
        cfg = BenchConfig.model_validate_json(config_bench.read_text(encoding="utf-8"))
    except Exception as e:
        err(f"Invalid bench config: {e}")
        raise typer.Exit(code=1)

    overrides = {
        k: v for k, v in {
            "host": host,
            "port": port,
            "api_key": api_key,
            "label": label,
            "num_prompts": num_prompts,
            "result_dir": result_dir,
        }.items() if v is not None
    }
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    for warning in cfg.collect_warnings():
        warn(warning)

    info(f"target=http://{cfg.host}:{cfg.port} num_prompts={cfg.num_prompts} label={cfg.label}")

    try:
        run_bench(cfg)
    except Exception as e:
        err(f"Benchmark failed: {e}")
        raise typer.Exit(code=1)


app.add_typer(bench_app, name="bench")


def cli():
    app()


if __name__ == "__main__":
    cli()