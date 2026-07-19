"""CLI de KuraiPilot. Subcomandos: convert, preview, live, bench, doctor.

doctor es el primer comando a correr en una máquina nueva. Los presets con
componentes de fases futuras (docs/07) fallan con mensaje claro, no traceback.
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


@app.callback(invoke_without_command=True)
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
    """Convierte un video a video ASCII."""
    import time

    from pydantic import ValidationError
    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

    from kurai.engine.decode import DecodeError, probe_video
    from kurai.engine.pipeline import run_job

    try:
        cfg = JobConfig(preset=load_preset(preset), cols=cols, output=output, auto_scene=auto)
    except ValidationError:
        console.print("[red]✗[/] --cols debe estar entre 20 y 600.")
        raise typer.Exit(code=1) from None
    try:
        meta = probe_video(input_file)
    except DecodeError as e:
        console.print(f"[red]✗[/] {e}")
        raise typer.Exit(code=1) from None

    console.print(
        f"Input: [bold]{input_file.name}[/] ({meta.width}x{meta.height} · "
        f"{meta.fps:g} fps · {meta.duration_s:.1f}s · "
        f"{'con' if meta.has_audio else 'sin'} audio) · "
        f"preset [bold]{cfg.preset.name}[/] · {cfg.cols} cols"
    )

    start = time.perf_counter()
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as bar:
            task = bar.add_task("Convirtiendo", total=meta.n_frames)
            result = run_job(
                input_file, cfg, on_progress=lambda done, _: bar.update(task, completed=done)
            )
    except NotImplementedError as e:
        console.print(
            f"[yellow]El preset '{cfg.preset.name}' necesita componentes de {e} "
            f"(docs/07-roadmap.md). Hoy: retro, detallado o nitido.[/]"
        )
        raise typer.Exit(code=2) from None
    except DecodeError as e:
        console.print(f"[red]✗[/] {e}")
        raise typer.Exit(code=1) from None

    wall = time.perf_counter() - start
    speed = meta.duration_s / wall if wall > 0 else 0.0
    console.print(f"[bold green]✓[/] {result} ({wall:.1f}s · {speed:.1f}× tiempo real)")


@app.command()
def preview(
    input_file: Annotated[Path, typer.Argument(exists=True, readable=True)],
    port: Annotated[int, typer.Option("--port")] = 8420,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
) -> None:
    """Preview interactivo en el navegador (WebGL, solo localhost)."""
    from kurai.engine.decode import DecodeError, probe_video

    try:
        from kurai.preview.server import serve
    except ImportError:
        console.print("[red]✗[/] El preview necesita sus dependencias: uv sync --extra preview")
        raise typer.Exit(code=1) from None

    # Validar el input ANTES de levantar el server: un archivo que no es video
    # falla acá con mensaje claro, no con un navegador colgado en "conectando".
    try:
        meta = probe_video(input_file)
    except DecodeError as e:
        console.print(f"[red]✗[/] {e}")
        raise typer.Exit(code=1) from None

    url = f"http://127.0.0.1:{port}"
    console.print(
        f"({meta.width}x{meta.height} · {meta.fps:g} fps · {meta.duration_s:.1f}s)", style="dim"
    )
    console.print(f"Preview de [bold]{input_file.name}[/] en {url} (Ctrl-C para salir)")
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    serve(input_file, port=port)


@app.command()
def live(
    input_file: Annotated[Path, typer.Argument(exists=True, readable=True)],
    preset: Annotated[str, typer.Option("--preset", "-p")] = "retro",
    cols: Annotated[
        int | None, typer.Option("--cols", min=2, help="Ancho máximo; default: el del terminal.")
    ] = None,
) -> None:
    """Reproduce en ASCII directo en la terminal (30 fps, sin audio)."""
    from kurai.engine.decode import DecodeError
    from kurai.engine.live import run_live

    # cols acá es el tope de ancho del TERMINAL (lo consume run_live/max_cols);
    # el cols del JobConfig es el del export y no aplica en live.
    cfg = JobConfig(preset=load_preset(preset))
    try:
        shown, skipped = run_live(input_file, cfg, max_cols=cols)
    except DecodeError as e:
        console.print(f"[red]✗[/] {e}")
        raise typer.Exit(code=1) from None
    except NotImplementedError as e:
        console.print(
            f"[yellow]El preset necesita componentes de {e}. Hoy: retro, detallado o nitido.[/]"
        )
        raise typer.Exit(code=2) from None
    except KeyboardInterrupt:
        console.print("Interrumpido.")
        return
    msg = f"{shown} frames mostrados"
    if skipped:
        msg += f" · {skipped} saltados (terminal lento)"
    console.print(msg)


@app.command()
def bench(
    check: Annotated[bool, typer.Option("--check", help="Falla si hay regresión >10%.")] = False,
    accept: Annotated[
        bool, typer.Option("--accept", help="Guarda este run como baseline aceptado.")
    ] = False,
) -> None:
    """Benchmark (docs/05 §6): passthrough (techo de I/O) + retro (gate ≥4×)."""
    import tempfile

    from kurai.bench import (
        check_regression,
        ensure_bench_clip,
        load_accepted,
        run_passthrough,
        run_retro,
        save,
    )

    r = probe()
    if not r.can_convert:
        console.print("[red]✗[/] ffmpeg no disponible — correr `kurai doctor`")
        raise typer.Exit(code=1)
    use_nvenc = r.hw_pipeline and not r.gpu_disabled_by_env
    encoder = "h264_nvenc" if use_nvenc else "libx264"

    console.print(f"Clip de referencia (1080p30 · 10 s) · encoder [bold]{encoder}[/]")
    clip = ensure_bench_clip()

    results = []
    with tempfile.TemporaryDirectory(prefix="kurai-bench-") as tmp:
        for name, runner_fn in (("passthrough", run_passthrough), ("retro", run_retro)):
            result = runner_fn(clip, Path(tmp), use_nvenc)
            results.append(result)
            console.print(
                f"  {name:<12} {result.frames} frames · {result.wall_seconds}s → "
                f"[bold green]{result.speed_factor}×[/] tiempo real"
            )

    save(results, accept=accept)
    if accept:
        console.print("[green]✓[/] Baselines aceptados en bench/results/accepted.json")
        return

    baselines = load_accepted()
    failed = False
    for result in results:
        baseline = baselines.get(result.mode)
        if baseline is None:
            console.print(f"[yellow]⚠[/] Sin baseline para '{result.mode}' — kurai bench --accept")
            failed = True
            continue
        failure = check_regression(result, baseline)
        if failure is not None:
            console.print(f"[red]✗[/] {result.mode}: {failure}")
            failed = True
        else:
            console.print(f"[green]✓[/] {result.mode}: sin regresión ({baseline.speed_factor}×)")
    if failed and check:
        raise typer.Exit(code=1)
