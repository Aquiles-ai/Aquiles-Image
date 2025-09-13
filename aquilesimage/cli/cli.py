import click

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
@click.option("--api-key", type=str, default=None, help="API KEY enabled to make requests")
@click.option("--max-concurrent-infer", type=int, default=50, help="Maximum concurrent inferences")
@click.option("--block-request", type=bool, default=False, help="Block requests during the maximum concurrent inferences")
def serve(host, port, api_key, max_concurrent_infer, block_request):
    import uvicorn
    from aquilesimage.main import app
    uvicorn.run(app, host=host, port=port)

@cli.command("configs")
def configs():
    pass

if __name__ == "__main__":
    cli()