from __future__ import annotations

import json
from contextlib import nullcontext

import typer

BANNER = (
    " ▄▀█ █▀█ █░█ ▀ █░ ██▀ █▀ ▄▄ ▀█▀ ▄▀▄▀▄ ▄▀▄ ▄▀░ ██▀\n"
    " █▀█ ▀▀█ ▀▄█ █ █▄ █▄▄ ▄█ ░░ ▄█▄ █░▀░█ █▀█ ▀▄█ █▄▄"
)

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    RICH = True
except ImportError:
    RICH = False

_console = None


def get_console():
    global _console
    if _console is None:
        _console = Console(highlight=False)
    return _console


def ok(msg: str) -> None:
    if RICH:
        get_console().print(f"[green]✓[/green] {msg}")
    else:
        typer.echo(f"OK: {msg}")


def err(msg: str) -> None:
    if RICH:
        get_console().print(f"[bold red]✗[/bold red] {msg}")
    else:
        typer.echo(f"X Error: {msg}", err=True)


def warn(msg: str) -> None:
    if RICH:
        get_console().print(f"[yellow]![/yellow] {msg}")
    else:
        typer.echo(f"Warning: {msg}", err=True)


def info(msg: str) -> None:
    if RICH:
        get_console().print(f"[grey62]›[/grey62] {msg}")
    else:
        typer.echo(msg)


def hint(msg: str) -> None:
    info(f"Try: {msg}")


def startup_banner(host: str, port: int, model: str, n_settings: int, dev_mode: bool = False) -> None:
    if not RICH:
        typer.echo(BANNER)
        typer.echo("\nStarting Aquiles-Image server:")
        typer.echo(f"   Host: {host}")
        typer.echo(f"   Port: {port}")
        typer.echo(f"   Model: {model}")
        typer.echo(f"   Config: {n_settings} settings loaded")
        if dev_mode:
            typer.echo("\nDev mode enabled — model loading skipped, responses are mocked")
        typer.echo(f"\nServer will be available at: http://{host}:{port}")
        return

    console = get_console()
    if console.size.width >= 56:
        console.print(Panel(
            Text(BANNER, style="bold magenta", justify="center"),
            box=box.HEAVY,
            border_style="magenta",
            padding=(0, 2),
        ))

    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_column(style="grey62", justify="right", width=8)
    t.add_column()
    t.add_row("host", str(host))
    t.add_row("port", str(port))
    t.add_row("model", Text(model, style="bold white"))
    t.add_row("config", f"{n_settings} settings loaded")
    console.print(t)
    console.print()
    if dev_mode:
        console.print("[yellow]![/yellow] Dev mode enabled — model loading skipped, responses are mocked")
    console.print(
        f"[green]✓[/green] Server will be available at "
        f"[bold underline]http://{host}:{port}[/bold underline]"
    )


def config_view(conf: dict) -> None:
    payload = json.dumps(conf, indent=4, ensure_ascii=False, default=str)
    if RICH:
        syntax = Syntax(payload, "json", theme="ansi_dark", background_color="default")
        get_console().print(Panel(syntax, box=box.ROUNDED, border_style="grey62"))
    else:
        typer.echo(payload)


def validation_error(details: str) -> None:
    err("Configuration validation failed:")
    if RICH:
        get_console().print(Panel(details, box=box.ROUNDED, border_style="red"))
    else:
        typer.echo(details, err=True)


def download_status(description: str):
    if RICH:
        return get_console().status(f"[magenta]{description}[/magenta]")
    typer.echo(f"{description}...")
    return nullcontext()


def model_not_found(model_id: str, available) -> None:
    err(f"Model '{model_id}' not found in registry.")
    info(f"Available: {', '.join(sorted(available))}")
