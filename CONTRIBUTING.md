# Contribuir a KuraiPilot

Guía operativa para desarrolladores y equipos. Las decisiones de arquitectura no viven acá — viven en [docs/](./docs/README.md) y son normativas; esta guía es el *cómo trabajamos*.

## Setup en 3 comandos

```bash
make setup    # entorno base con uv (CPU-only alcanza para casi todo el desarrollo)
make hooks    # activa el pre-push local (corre make check)
make doctor   # te dice qué tiene tu máquina (ffmpeg, NVDEC/NVENC, GPU, Ollama)
```

**No necesitás GPU para contribuir**: toda la suite corre con `KURAI_DISABLE_GPU=1` (así corre CI). La GPU solo hace falta para trabajar en el AI Sidecar (Fase 1+, `make setup-gpu`) y para performance ([§ Performance](#performance-solo-en-la-máquina-de-referencia)).

## Flujo de trabajo

Trunk-based con ramas cortas:

1. Rama desde `main`: `tipo/descripcion-corta` (`feat/bench-passthrough`, `fix/vfr-metadata`).
2. Commits convencionales: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `chore:`. Cuerpo en español si aporta contexto.
3. PR contra `main`. La plantilla de PR pide lo que el reviewer necesita; completala de verdad.
4. Merge con squash. `main` siempre está en verde y siempre es instalable.

Configuración recomendada de branch protection en GitHub (el owner la aplica una vez): requerir el workflow `CI` en verde, 1 review aprobado, historial lineal, sin push directo a `main`.

## Los tres gates (en orden de cercanía)

| Gate | Cuándo | Qué corre |
| --- | --- | --- |
| `make check` | Antes de commitear | ruff + mypy strict + import-linter + pytest |
| Hook pre-push | Al pushear (automático con `make hooks`) | Lo mismo |
| CI (Actions) | En el PR | Lo mismo, en matrix 3.12/3.13, CPU-only |

Si CI está rojo, el PR no se revisa. Si necesitás ayuda para ponerlo en verde, pedila en el PR mismo.

## Dónde está cada verdad (leer antes de tocar)

- **Vas a implementar una etapa del pipeline** → [docs/02](./docs/02-pipeline-spec.md) define su contrato; tu PR referencia la sección que implementa.
- **Vas a tocar cualquier componente con modelo** → [docs/04](./docs/04-ai-components.md) es normativo (presupuestos de latencia/VRAM, scheduling). Cambiarlo exige actualizar el doc en el mismo PR.
- **Querés cambiar una decisión estructural** → se escribe un ADR nuevo en [docs/adr/](./docs/adr/) que supersede al anterior; los ADRs aceptados no se editan. El ADR se discute en su propio PR, antes del PR de implementación.
- **Tu tarea tiene gate de fase** → [docs/07](./docs/07-roadmap.md); el criterio de aceptación del gate va en la descripción del PR.

## Reglas que el tooling hace cumplir (no negociables en review)

1. **Capas de arquitectura**: `import-linter` valida en CI que los imports respeten las capas de [docs/01 §3](./docs/01-architecture-overview.md) (p. ej. `engine` jamás importa `cli`; `ai` jamás importa `engine`). Si tu cambio necesita romper una capa, el problema es el diseño del cambio — hablalo antes de escribir código.
2. **Vectorización estricta** en etapas 2–8 (ADR-006): sin loops por celda/píxel en Python. El reviewer rechaza aunque funcione.
3. **mypy strict** en `src/kurai`: sin `Any` en firmas propias, sin `# type: ignore` sin comentario que explique por qué.
4. **Marcadores de test declarados**: `--strict-markers` activo; un marcador nuevo se agrega a `pyproject.toml` con su descripción.

## Reglas que el reviewer hace cumplir

- **Golden files** ([docs/06 §1](./docs/06-testing-and-evaluation.md)): un cambio en `tests/golden/` exige que el commit explique el cambio de algoritmo que lo justifica. Golden cambiado sin justificación = regresión, se rechaza.
- **Presets**: preset nuevo o modificado toca tres lugares en el mismo PR: `presets/*.toml`, la tabla de `docs/02 §10`, y `SPEC_TABLE` en `tests/test_config.py` (el sync es manual a propósito — obliga a mirar la spec).
- **Determinismo** (G4): nada de aleatoriedad sin semilla, tiempo del sistema, u orden de iteración no determinista en el pipeline.
- **Mensajes al usuario del CLI en español; código e identificadores en inglés.**

## Performance: solo en la máquina de referencia

CI **no** mide performance (ADR-001: no hay CI con GPU). Si tu PR toca el hot path (etapas 2–8, transferencias, colas):

1. Corré `kurai bench --check` en la máquina de referencia (o pedile al owner que lo corra).
2. Adjuntá el JSON de resultado al PR.
3. Regresión >10% en speed factor bloquea el merge salvo decisión explícita documentada en el PR.

## Trabajo con agentes (Claude Code, etc.)

Los agentes siguen [CLAUDE.md](./CLAUDE.md), que apunta a las mismas reglas de esta guía. Un PR generado con asistencia de IA se revisa igual que cualquier otro — quien lo abre es responsable de su contenido (el mismo principio de Diligence de [IDEA.md](./IDEA.md) aplicado a nosotros).

## Ownership

El mapa de ownership por área está en [.github/CODEOWNERS](./.github/CODEOWNERS). Regla general: los docs normativos (`docs/`, `docs/adr/`) requieren review del owner de arquitectura; el resto, del owner del área tocada.
