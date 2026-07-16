#!/usr/bin/env python3
"""Descarga los modelos ONNX declarados en models/manifest.toml y verifica SHA-256.

Sin red, el sistema arranca en modo determinista puro (docs/03 §6). Un modelo
con hash que no coincide se borra y el script falla: nunca se carga un modelo
no verificado.

Uso: uv run python tools/fetch_models.py [--only saliency]
"""

from __future__ import annotations

import hashlib
import sys
import tomllib
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MANIFEST = MODELS_DIR / "manifest.toml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    with MANIFEST.open("rb") as f:
        manifest = tomllib.load(f)
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None

    for name, spec in manifest.get("models", {}).items():
        if only and name != only:
            continue
        dest = MODELS_DIR / spec["filename"]
        if dest.exists() and sha256(dest) == spec["sha256"]:
            print(f"✓ {name} ya presente y verificado")
            continue
        if not spec.get("url"):
            print(f"⚠ {name}: sin URL en el manifest (pendiente de pinear) — saltado")
            continue
        print(f"↓ {name} desde {spec['url']}")
        urllib.request.urlretrieve(spec["url"], dest)
        actual = sha256(dest)
        if actual != spec["sha256"]:
            dest.unlink()
            print(
                f"✗ {name}: SHA-256 no coincide (esperado {spec['sha256'][:12]}…, "
                f"obtenido {actual[:12]}…). Borrado."
            )
            return 1
        print(f"✓ {name} descargado y verificado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
