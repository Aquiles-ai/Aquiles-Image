import click
from typing import Optional
import sys

@click.group()
def cli():
    """A sample CLI application."""
    pass

@cli.command("hello")
@click.option("--name")
def greet(name):
    click.echo(f"Hello, {name}!")

@cli.command("serve")
@click.option("--host", default="0.0.0.0", help="Host where Aquiles-Image will be executed")
@click.option("--port", type=int, default=5500, help="Port where Aquiles-Image will be executed")
@click.option("--model", type=str, help="The model to use for image generation.")
@click.option("--api-key", type=str, help="API KEY enabled to make requests")
@click.option("--max-concurrent-infer", type=int, help="Maximum concurrent inferences")
@click.option("--block-request/--no-block-request", default=None, help="Block requests during maximum concurrent inferences")
@click.option("--force", is_flag=True, help="Force overwrite existing configuration")
@click.option("--no-load-model", is_flag=True, help="Not loading the model simply allows for faster development without having to load the model constantly.")
@click.option("--set-steps", type=int, default=None, help="Set the steps that the model will use")
@click.option("--auto-pipeline/--no-auto-pipeline", default=None, help="Load a model that is compatible with diffusers but is not mentioned in the Aquiles-Image documentation")
@click.option("--device-map", type=str, default=None, help="Device map option in which to load the model (Only compatible with diffusers/FLUX.2-dev-bnb-4bit)")
@click.option("--dist-inference/--no-dist-inference", default=None, help="Use distributed inference (Not yet implemented)")
@click.option("--max-batch-size", type=int, default=None, help="Maximum number of requests to group in a single batch for inference")
@click.option("--batch-timeout", type=float, default=None, help="Maximum time (in seconds) to wait before processing a batch even if not full")
@click.option("--worker-sleep", type=float, default=None, help="Time (in seconds) the worker sleeps between checking for new batch requests")
def serve(host: str, port: int, model: Optional[str], api_key: Optional[str], 
         max_concurrent_infer: Optional[int], block_request: Optional[bool], force: bool, 
         no_load_model: bool, set_steps: Optional[int], auto_pipeline: Optional[bool], 
         device_map: Optional[str], dist_inference: Optional[bool], max_batch_size: Optional[int], 
         batch_timeout: Optional[float], worker_sleep: Optional[float]):
    """Start the Aquiles-Image server."""
    try:
        from aquilesimage.configs import (
            load_config_cli, 
            configs_image_serve, 
            config_file_exists,
            create_basic_config_if_not_exists
        )
        from aquilesimage.models import ConfigsServe
    except ImportError as e:
        click.echo(f"X Error importing configuration modules: {e}", err=True)
        sys.exit(1)

    config_exists = config_file_exists()
    
    if not config_exists:
        if model:
            click.echo(f"No configuration found. Creating basic configuration with model: {model}")
            try:
                if no_load_model:
                    created = create_basic_config_if_not_exists(model, False)
                else:
                    created = create_basic_config_if_not_exists(model)
            except Exception as e:
                click.echo(f"X Error creating basic configuration: {e}", err=True)
                sys.exit(1)
        else:
            try:
                created = create_basic_config_if_not_exists()
            except Exception as e:
                click.echo(f"X Error creating default configuration: {e}", err=True)
                sys.exit(1)

    try:
        conf = load_config_cli()
    except Exception as e:
        click.echo(f"X Error loading configuration: {e}", err=True)
        sys.exit(1)

    model_from_config = conf.get("model")
    final_model = model or model_from_config

    if not final_model:
        click.echo("X Error: No model specified. Use --model parameter or configure one first.", err=True)
        sys.exit(1)

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
        worker_sleep is not None
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
                worker_sleep=worker_sleep if worker_sleep is not None else conf.get("worker_sleep")
            )

            configs_image_serve(updated_conf, force=True)
            click.echo("Configuration updated successfully.")
            
        except Exception as e:
            click.echo(f"X Error updating configuration: {e}", err=True)
            sys.exit(1)

    try:
        import uvicorn
    except ImportError as e:
        click.echo(f"X Error importing uvicorn: {e}", err=True)
        sys.exit(1)
    
    try:
        from aquilesimage.main import app
    except TypeError as e:
        click.echo(f"X Error loading application (Pydantic validation): {e}", err=True)
        click.echo(f"X This might be caused by invalid configuration values.", err=True)
        click.echo(f"X Try running: aquiles-image configs --reset", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"X Error loading application: {e}", err=True)
        import traceback
        click.echo(f"X Traceback: {traceback.format_exc()}", err=True)
        sys.exit(1)

    click.echo(f"\nStarting Aquiles-Image server:")
    click.echo(f"   Host: {host}")
    click.echo(f"   Port: {port}")
    click.echo(f"   Model: {final_model}")
    click.echo(f"   Config: {len(conf)} settings loaded")
    click.echo(f"\nServer will be available at: http://{host}:{port}")
    if no_load_model:
        click.echo(f"\nAquiles-Image server in dev mode without loading the model")
    
    try:
        uvicorn.run(app, host=host, port=port)
    except KeyboardInterrupt:
        click.echo("\nServer stopped by user.")
    except Exception as e:
        click.echo(f"X Error starting server: {e}", err=True)
        sys.exit(1)

@cli.command("configs")
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--reset", is_flag=True, help="Reset configuration to defaults")
def configs(show: bool, reset: bool):
    """Manage Aquiles-Image configuration."""
    try:
        from aquilesimage.configs import load_config_cli, clear_config_cache
        import json
    except ImportError as e:
        click.echo(f"Error importing required modules: {e}", err=True)
        sys.exit(1)

    if reset:
        if click.confirm("Are you sure you want to reset the configuration?"):
            try:
                clear_config_cache() 
                click.echo("Configuration reset successfully.")
            except Exception as e:
                click.echo(f"Error resetting configuration: {e}", err=True)
        return

    if show:
        try:
            conf = load_config_cli()
            if conf:
                click.echo("Current configuration:")
                click.echo(json.dumps(conf, indent=2, ensure_ascii=False))
            else:
                click.echo("No configuration found.")
        except Exception as e:
            click.echo(f"Error loading configuration: {e}", err=True)
        return

    ctx = click.get_current_context()
    click.echo(ctx.get_help())


@cli.command("validate")
def validate():
    """Validate current configuration."""
    try:
        from aquilesimage.configs import load_config_cli
        from aquilesimage.models import ConfigsServe
    except ImportError as e:
        click.echo(f"Error importing required modules: {e}", err=True)
        sys.exit(1)

    try:
        conf = load_config_cli()
        
        if not conf:
            click.echo("❌ No configuration found.", err=True)
            sys.exit(1)
            
        validated_conf = ConfigsServe(**conf)
        click.echo("Configuration is valid.")
        
    except Exception as e:
        click.echo(f"❌ Configuration validation failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()