#!/usr/bin/env python3
"""A/B ciego retro vs. detallado para el gate de saliencia (docs/06 §4, docs/07 Fase 1).

Reproduce cada par {clip}__retro.mp4 / {clip}__detallado.mp4 de output/ en
orden aleatorio, con las opciones etiquetadas solo como "A"/"B" — la
resolución retro/detallado se guarda en el voto pero nunca se muestra durante
la sesión. Correr `vote` de nuevo sin --session abre una sesión ciega nueva:
así el mismo evaluador puede acumular las ≥3 sesiones que exige el gate.

Uso:
    uv run python tools/ab_review.py vote --evaluator ana
    uv run python tools/ab_review.py vote --evaluator ana --session 2
    uv run python tools/ab_review.py report
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
EVAL_DIR = ROOT / "docs" / "evaluations"
VOTES_FILE = EVAL_DIR / "ab_saliencia_votes.jsonl"
THRESHOLD = 0.60
MIN_EVALUATORS = 3


@dataclass
class Vote:
    ts: str
    evaluator: str
    session: str
    clip: str
    choice: str  # "retro" | "detallado" | "tie" | "skip"
    commit: str


def discover_pairs(clips_dir: Path) -> list[str]:
    """Nombres de clip con ambos presets generados en clips_dir."""
    names = []
    for retro in sorted(clips_dir.glob("*__retro.mp4")):
        name = retro.name.removesuffix("__retro.mp4")
        if (clips_dir / f"{name}__detallado.mp4").is_file():
            names.append(name)
    return names


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_votes(votes_file: Path) -> list[Vote]:
    if not votes_file.is_file():
        return []
    votes = []
    for line in votes_file.read_text().splitlines():
        line = line.strip()
        if line:
            votes.append(Vote(**json.loads(line)))
    return votes


def append_vote(votes_file: Path, vote: Vote) -> None:
    votes_file.parent.mkdir(parents=True, exist_ok=True)
    with votes_file.open("a") as f:
        f.write(json.dumps(asdict(vote)) + "\n")


def _player_env() -> dict[str, str]:
    """SDL (audio y, en Wayland, video) necesita XDG_RUNTIME_DIR válido; en
    sesiones sin escritorio completo (SSH, systemd sin login) suele faltar."""
    env = os.environ.copy()
    runtime_dir = env.get("XDG_RUNTIME_DIR")
    if not runtime_dir or not Path(runtime_dir).is_dir():
        fallback = Path(tempfile.gettempdir()) / f"xdg-runtime-{os.getuid()}"
        fallback.mkdir(mode=0o700, exist_ok=True)
        env["XDG_RUNTIME_DIR"] = str(fallback)
    return env


def play(player: str, path: Path, label: str) -> None:
    print(f"  reproduciendo opción {label}... ('q' en la ventana para cortar)")
    # -an: el audio es idéntico entre presets (ADR-003) y no aporta a la
    # comparación visual — además evita depender de PulseAudio/SDL_audio.
    cmd = [
        player,
        "-autoexit",
        "-an",
        "-window_title",
        f"Opción {label}",
        "-loglevel",
        "quiet",
        str(path),
    ]
    subprocess.run(cmd, check=False, env=_player_env())


def prompt_choice() -> str:
    valid = {"a", "b", "empate", "e", "r", "s", "q"}
    prompt = "¿Preferís [a] o [b]? (a/b/empate/r=repetir/s=saltar/q=salir): "
    while True:
        ans = input(prompt).strip().lower()
        if ans in valid:
            return ans
        print("  respuesta no válida.")


def run_vote(args: argparse.Namespace) -> int:
    clips_dir = Path(args.clips_dir)
    player = args.player
    if shutil.which(player) is None:
        print(f"No se encontró '{player}' en PATH. Instalá ffmpeg (trae ffplay) o pasá --player.")
        return 1

    pairs = discover_pairs(clips_dir)
    if not pairs:
        print(f"No hay pares __retro.mp4 / __detallado.mp4 en {clips_dir}")
        return 1

    session = args.session or datetime.now().strftime("%Y%m%dT%H%M%S")
    votes = load_votes(VOTES_FILE)
    voted = {(v.evaluator, v.session, v.clip) for v in votes}
    pending = [c for c in pairs if (args.evaluator, session, c) not in voted]
    if not pending:
        print("Ya votaste todos los clips en esta sesión.")
        return 0

    random.shuffle(pending)
    commit = git_commit()
    print(f"Sesión ciega '{session}' — evaluador '{args.evaluator}' — {len(pending)} clips.")
    print("Las opciones se muestran en orden aleatorio; la etiqueta real no se revela acá.\n")

    for clip in pending:
        mapping = [
            ("retro", clips_dir / f"{clip}__retro.mp4"),
            ("detallado", clips_dir / f"{clip}__detallado.mp4"),
        ]
        random.shuffle(mapping)
        (name_a, path_a), (name_b, path_b) = mapping

        print(f"Clip: {clip}")
        while True:
            play(player, path_a, "A")
            play(player, path_b, "B")
            ans = prompt_choice()
            if ans != "r":
                break

        if ans == "q":
            print("Sesión interrumpida — los clips restantes quedan pendientes.")
            break

        choice = {"a": name_a, "b": name_b, "empate": "tie", "e": "tie", "s": "skip"}[ans]
        vote = Vote(
            ts=datetime.now(UTC).isoformat(),
            evaluator=args.evaluator,
            session=session,
            clip=clip,
            choice=choice,
            commit=commit,
        )
        append_vote(VOTES_FILE, vote)
        print("  registrado.\n")

    print(f"Votos guardados en {VOTES_FILE.relative_to(ROOT)}")
    return 0


def run_report(args: argparse.Namespace) -> int:
    votes = load_votes(VOTES_FILE)
    if not votes:
        print("No hay votos registrados todavía — corré 'vote' primero.")
        return 1

    decided = [v for v in votes if v.choice in ("retro", "detallado")]
    ties = [v for v in votes if v.choice == "tie"]
    skips = [v for v in votes if v.choice == "skip"]
    detallado_votes = sum(1 for v in decided if v.choice == "detallado")
    pct = detallado_votes / len(decided) if decided else 0.0

    evaluators = sorted({v.evaluator for v in votes})
    sessions_per_evaluator: dict[str, set[str]] = {}
    for v in votes:
        sessions_per_evaluator.setdefault(v.evaluator, set()).add(v.session)
    max_sessions_single = max((len(s) for s in sessions_per_evaluator.values()), default=0)

    enough_evaluators = len(evaluators) >= MIN_EVALUATORS or max_sessions_single >= MIN_EVALUATORS
    gate_pass = enough_evaluators and pct >= THRESHOLD

    by_clip: dict[str, dict[str, int]] = {}
    for v in votes:
        d = by_clip.setdefault(v.clip, {"detallado": 0, "retro": 0, "tie": 0})
        if v.choice in d:
            d[v.choice] += 1

    lines = [
        f"# A/B saliencia — {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"- Commit: `{git_commit()}`",
        f"- Votos decididos: {len(decided)} (empates: {len(ties)}, saltados: {len(skips)})",
        f"- Evaluadores: {', '.join(evaluators)} ({len(evaluators)})",
        f"- Preferencia por `detallado` (saliencia): "
        f"**{pct:.0%}** ({detallado_votes}/{len(decided)})",
        f"- Gate (≥{THRESHOLD:.0%}, ≥{MIN_EVALUATORS} evaluadores o sesiones ciegas): "
        f"{'✅ PASA' if gate_pass else '❌ NO PASA'}",
        "",
        "## Por clip",
        "",
        "| Clip | detallado | retro | empate |",
        "|---|---|---|---|",
    ]
    for clip, d in sorted(by_clip.items()):
        lines.append(f"| {clip} | {d['detallado']} | {d['retro']} | {d['tie']} |")
    lines += [
        "",
        "## Decisión",
        "",
        "(completar a mano: adoptar saliencia / abandonar / repetir con más evaluadores)",
        "",
    ]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    default_out = EVAL_DIR / f"{datetime.now().strftime('%Y-%m-%d')}-ab-saliencia.md"
    out_path = Path(args.out) if args.out else default_out
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n-> {out_path.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_vote = sub.add_parser("vote", help="Corre una sesión ciega de votación.")
    p_vote.add_argument("--evaluator", required=True, help="Nombre/iniciales de quien vota.")
    p_vote.add_argument("--session", default=None, help="ID de sesión (default: timestamp nuevo).")
    p_vote.add_argument("--clips-dir", default=str(OUTPUT_DIR))
    p_vote.add_argument("--player", default="ffplay")
    p_vote.set_defaults(func=run_vote)

    p_report = sub.add_parser("report", help="Agrega los votos y escribe docs/evaluations/.")
    p_report.add_argument("--out", default=None)
    p_report.set_defaults(func=run_report)

    args = parser.parse_args()
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrumpido — los votos ya registrados quedaron guardados.")
        raise SystemExit(130) from None
