import click

@click.group()
def cli():
    """A sample CLI application."""
    pass

@cli.command("hello")
@click.option("--name")
def greet(name):
    click.echo(f"Hello, {name}!")

if __name__ == "__main__":
    cli()