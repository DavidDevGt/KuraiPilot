# KuraiPilot

Conversor **local-first** de cualquier video a video ASCII renderizado — audio intacto, reproducible en cualquier player. Ningún frame sale de tu máquina.

```bash
make setup && make doctor   # prepara el entorno y verifica ffmpeg/GPU
kurai convert video.mp4     # (Fase 0, en construcción)
```

- **Visión de producto**: [IDEA.md](./IDEA.md) · **Estado del arte**: [INVESTIGATION.md](./INVESTIGATION.md)
- **Arquitectura** (normativa): [docs/](./docs/README.md) · **Roadmap y gates**: [docs/07-roadmap.md](./docs/07-roadmap.md)
- **Para agentes/contribuidores**: [CLAUDE.md](./CLAUDE.md)

Principio rector: determinista por defecto, IA por elección — el pipeline completo funciona con todos los componentes de IA apagados, y cada componente de IA debe ganarse su lugar con métrica o A/B ciego ([ADR-002](./docs/adr/ADR-002-deterministic-core.md)).
