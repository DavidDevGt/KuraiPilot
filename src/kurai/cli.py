"""CLI de KuraiPilot. Subcomandos: convert, preview, live, bench, doctor.

convert/preview/live/bench se implementan por fases (docs/07); doctor funciona hoy
y es el primer comando a correr en una máquina nueva.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from kurai import __version__
from kurai.config import JobConfig, load_preset
from kurai.probe import probe

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

FASE_PENDIENTE = "[yellow]Aún no implementado[/] — fase pendiente del roadmap (docs/07-roadmap.md)."


@app.callback()
def _main(
    version: Annotated[bool, typer.Option("--version", help="Muestra la versión y sale.")] = False,
) -> None:
    if version:
        console.print(f"kurai {__version__}")
        raise typer.Exit()


@app.command()
def doctor() -> None:
    """Verifica el entorno: ffmpeg, NVDEC/NVENC, GPU, Ollama."""
    r = probe()
    table = Table(title="kurai doctor", show_header=False)
    table.add_column(style="bold")
    table.add_column()

    def mark(ok: bool, extra: str = "") -> str:
        return f"[green]✓[/] {extra}" if ok else f"[red]✗[/] {extra}"

    table.add_row("ffmpeg", mark(r.ffmpeg_path is not None, r.ffmpeg_version or ""))
    table.add_row("hwaccel cuda", mark(r.hwaccel_cuda))
    table.add_row("NVDEC (h264_cuvid)", mark(r.nvdec_h264))
    table.add_row("NVENC (h264_nvenc)", mark(r.nvenc_h264))
    if r.gpu_name:
        table.add_row(
            "GPU", f"[green]✓[/] {r.gpu_name} · {r.vram_free_mb}/{r.vram_total_mb} MB libres"
        )
    table.add_row(
        "Ollama (opcional)",
        f"[green]✓[/] {r.ollama_vision_model}" if r.ollama_vision_model else "[dim]—[/]",
    )
    console.print(table)

    for w in r.warnings:
        console.print(f"[yellow]⚠[/] {w}")
    for e in r.errors:
        console.print(f"[red]✗[/] {e}")

    if not r.can_convert:
        raise typer.Exit(code=1)
    verdict = "pipeline por hardware completo" if r.hw_pipeline else "convertirá por software"
    console.print(f"\n[bold green]OK[/] — {verdict}.")


@app.command()
def convert(
    input_file: Annotated[Path, typer.Argument(exists=True, readable=True)],
    preset: Annotated[str, typer.Option("--preset", "-p")] = "retro",
    cols: Annotated[int, typer.Option("--cols")] = 160,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    auto: Annotated[
        bool, typer.Option("--auto", help="Scene Analyst sugiere preset por escena.")
    ] = False,
) -> None:
    """Convierte un video a video ASCII (Fase 0)."""
    cfg = JobConfig(preset=load_preset(preset), cols=cols, output=output, auto_scene=auto)
    console.print(f"Input: {input_file} · preset [bold]{cfg.preset.name}[/] · {cfg.cols} cols")
    console.print(FASE_PENDIENTE)
    raise typer.Exit(code=2)


@app.command()
def preview(
    input_file: Annotated[Path, typer.Argument(exists=True, readable=True)],
    port: Annotated[int, typer.Option("--port")] = 8420,
) -> None:
    """Preview interactivo en el navegador (Fase 0.5)."""
    console.print(FASE_PENDIENTE)
    raise typer.Exit(code=2)


@app.command()
def live(
    input_file: Annotated[Path | None, typer.Argument()] = None,
    webcam: Annotated[bool, typer.Option("--webcam")] = False,
) -> None:
    """Reproduce en ASCII directo en la terminal (Fase 0.5)."""
    console.print(FASE_PENDIENTE)
    raise typer.Exit(code=2)


@app.command()
def bench(
    check: Annotated[bool, typer.Option("--check", help="Falla si hay regresión >10%.")] = False,
) -> None:
    """Benchmark sobre los fixtures (docs/05 §6). Se construye ANTES que el pipeline."""
    console.print(FASE_PENDIENTE)
    raise typer.Exit(code=2)
