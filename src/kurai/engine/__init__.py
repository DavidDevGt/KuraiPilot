"""Engine: las 9 etapas del pipeline (docs/02-pipeline-spec.md, normativo).

Regla de vectorización estricta (ADR-006): en las etapas 2-8 está prohibido
iterar por píxel o por celda en Python. Un `for` sobre celdas se rechaza en
revisión aunque funcione.
"""
