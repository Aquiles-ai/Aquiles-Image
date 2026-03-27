import typer
from typing import Optional
import sys

app = typer.Typer()


# Equivalente a @click.group() — se ejecuta antes de cualquier subcomando
@app.callback()
def cli():
    """A sample CLI application."""
    pass


@app.command("hello")
def greet(name: str = typer.Option(..., help="Name to greet")):
    typer.echo(f"Hello, {name}!")


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
):
    """Start the Aquiles-Image server."""

    if auto_pipeline_type is not None and auto_pipeline_type not in ("t2i", "i2i"):
        typer.echo("X Error: --auto-pipeline-type must be 't2i' or 'i2i'.", err=True)
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
        typer.echo(f"X Error importing configuration modules: {e}", err=True)
        raise typer.Exit(code=1)

    config_exists = config_file_exists()

    if not config_exists:
        if model:
            typer.echo(f"No configuration found. Creating basic configuration with model: {model}")
            try:
                if no_load_model:
                    create_basic_config_if_not_exists(model, False)
                else:
                    create_basic_config_if_not_exists(model)
            except Exception as e:
                typer.echo(f"X Error creating basic configuration: {e}", err=True)
                raise typer.Exit(code=1)
        else:
            try:
                create_basic_config_if_not_exists()
            except Exception as e:
                typer.echo(f"X Error creating default configuration: {e}", err=True)
                raise typer.Exit(code=1)

    try:
        conf = load_config_cli()
    except Exception as e:
        typer.echo(f"X Error loading configuration: {e}", err=True)
        raise typer.Exit(code=1)

    model_from_config = conf.get("model")
    final_model = model or model_from_config

    if not final_model:
        typer.echo("X Error: No model specified. Use --model parameter or configure one first.", err=True)
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
                allows_users=_build_allowed_users(username, password, conf)
            )

            configs_image_serve(updated_conf, force=True)
            typer.echo("Configuration updated successfully.")

        except Exception as e:
            typer.echo(f"X Error updating configuration: {e}", err=True)
            raise typer.Exit(code=1)

    try:
        import uvicorn
    except ImportError as e:
        typer.echo(f"X Error importing uvicorn: {e}", err=True)
        raise typer.Exit(code=1)

    try:
        from aquilesimage.main import app as fastapi_app
    except TypeError as e:
        typer.echo(f"X Error loading application (Pydantic validation): {e}", err=True)
        typer.echo("X This might be caused by invalid configuration values.", err=True)
        typer.echo("X Try running: aquiles-image configs --reset", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"X Error loading application: {e}", err=True)
        import traceback
        typer.echo(f"X Traceback: {traceback.format_exc()}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\nStarting Aquiles-Image server:")
    typer.echo(f"   Host: {host}")
    typer.echo(f"   Port: {port}")
    typer.echo(f"   Model: {final_model}")
    typer.echo(f"   Config: {len(conf)} settings loaded")
    typer.echo(f"\nServer will be available at: http://{host}:{port}")
    if no_load_model:
        typer.echo("\nAquiles-Image server in dev mode without loading the model")

    try:
        uvicorn.run(fastapi_app, host=host, port=port)
    except KeyboardInterrupt:
        typer.echo("\nServer stopped by user.")
    except Exception as e:
        typer.echo(f"X Error starting server: {e}", err=True)
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
        typer.echo(f"Error importing required modules: {e}", err=True)
        raise typer.Exit(code=1)

    if reset:
        if typer.confirm("Are you sure you want to reset the configuration?"):
            try:
                clear_config_cache()
                typer.echo("Configuration reset successfully.")
            except Exception as e:
                typer.echo(f"Error resetting configuration: {e}", err=True)
        return

    if show:
        try:
            conf = load_config_cli()
            if conf:
                typer.echo("Current configuration:")
                typer.echo(json.dumps(conf, indent=2, ensure_ascii=False))
            else:
                typer.echo("No configuration found.")
        except Exception as e:
            typer.echo(f"Error loading configuration: {e}", err=True)
        return

    typer.echo(typer.get_current_context().get_help())


@app.command("validate")
def validate():
    """Validate current configuration."""
    try:
        from aquilesimage.configs import load_config_cli
        from aquilesimage.models import ConfigsServe
    except ImportError as e:
        typer.echo(f"Error importing required modules: {e}", err=True)
        raise typer.Exit(code=1)

    try:
        conf = load_config_cli()

        if not conf:
            typer.echo("No configuration found.", err=True)
            raise typer.Exit(code=1)

        ConfigsServe(**conf)
        typer.echo("Configuration is valid.")

    except Exception as e:
        typer.echo(f"Configuration validation failed: {e}", err=True)
        raise typer.Exit(code=1)


def cli():
    app()


if __name__ == "__main__":
    cli()