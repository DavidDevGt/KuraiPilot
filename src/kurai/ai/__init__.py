"""AI Sidecar + Scene Analyst (docs/04-ai-components.md, NORMATIVO).

Regla de gobernanza: no se agrega ni cambia un modelo sin actualizar docs/04
y el ADR correspondiente. Hot path: ≤10 ms/invocación, ≤2 GB VRAM agregado.
Todo componente de acá degrada a determinista si falla — nunca aborta el job.
"""
