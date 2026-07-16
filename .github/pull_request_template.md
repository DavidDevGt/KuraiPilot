## Qué cambia y por qué

<!-- Una o dos frases. Si implementa una etapa/gate, referenciá la sección: "Implementa docs/02 E4" / "Gate de Fase 1 (docs/07)". -->

## Checklist

- [ ] `make check` en verde localmente (ruff + mypy + import-linter + pytest)
- [ ] Si toca la spec del pipeline o componentes de IA: `docs/02` / `docs/04` actualizados en este PR
- [ ] Si cambia un preset: `presets/*.toml` + tabla `docs/02 §10` + `SPEC_TABLE` en `tests/test_config.py`, los tres
- [ ] Si cambia un golden file: el mensaje de commit justifica el cambio de algoritmo
- [ ] Si toca el hot path (etapas 2–8): resultado de `kurai bench --check` de la máquina de referencia adjunto
- [ ] Si revierte/cambia una decisión estructural: ADR nuevo en `docs/adr/` (discutido en PR aparte)

## Cómo se probó

<!-- Tests nuevos/modificados, o el comando manual con su salida. "Pasa CI" no es respuesta. -->
