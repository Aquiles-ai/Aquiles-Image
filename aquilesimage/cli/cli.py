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
@click.option("--host", default="0.0.0.0", help="Host where Aquiles-RAG will be executed")
@click.option("--port", type=int, default=5500, help="Port where Aquiles-RAG will be executed")
def serve(host, port):
    import uvicorn
    from aquilesimage.main import app
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    cli()