# 06 — Testing & Evaluation

Estrategia de calidad en dos planos: **corrección** (el pipeline hace lo que la spec dice — tests automáticos) y **calidad percibida** (el resultado se ve bien — evaluación con métricas y A/B). La distinción importa porque los componentes de IA no pueden validarse solo con asserts: necesitan la disciplina de Discernment definida en [IDEA.md](../IDEA.md) — ningún componente de IA se promociona porque "se ve bien" a ojo de quien lo implementó.

## 1. Golden files sobre el artefacto canónico

- El objeto testeado es la **CharMatrix** ([01 §5](./01-architecture-overview.md)), no el video final: comparar mp4s es frágil (el encoder mete no-determinismo); comparar matrices `uint8` es exacto.
- `tests/golden/` guarda, por cada (fixture × preset determinista), la CharMatrix de frames de muestra (frames 0, N/2, N-1 y los 2 posteriores a cada corte de escena) como `.npz` comprimido.
- **Garantía G4**: el test corre el pipeline dos veces sobre el mismo input y exige igualdad bit a bit entre corridas, además de igualdad contra el golden. Cubre presets `retro` y `detallado` completos; para `alta-fidelidad`, cubre las etapas deterministas con las IA mockeadas (la CNN y el flow reales se validan en §3-4, no con golden exacto — un cambio de versión de ORT puede mover un logit sin que sea un bug).
- Actualizar un golden requiere justificación en el commit: un golden que cambia sin cambio intencional de algoritmo es una regresión, no un test flaky.

## 2. Tests unitarios y de contrato por etapa

Cobertura mínima obligatoria por etapa (pytest, sin GPU en CI — camino `KURAI_DISABLE_GPU`):

- **E1**: metadatos de rotación respetados; VFR→CFR produce `n_frames` esperado; audio extraído bit-idéntico (hash del stream vs. `ffmpeg -c:a copy` directo).
- **E2**: corrección de aspecto — un círculo perfecto de entrada produce una CharMatrix cuyo bounding box de caracteres no-espacio es ~cuadrado (±1 celda). Luma de un frame gris uniforme al 50% cae en el nivel medio de la rampa.
- **E4**: monotonía — luma creciente nunca produce índice de rampa decreciente. Gamma aplicada una sola vez.
- **E6**: Bayer es idéntico entre frames para input estático (cero flicker inducido, por construcción). FS serpentine reproduce el golden de un gradiente sintético.
- **E7**: propiedad de histéresis — input con ruido gaussiano σ < h/2 sobre fondo estático produce **cero** cambios de carácter tras el frame 1. Corte de escena resetea (el frame post-corte es idéntico al de un job que empieza ahí).
- **E8**: render de una CharMatrix con todos los glifos reproduce el atlas exactamente (identidad del fancy indexing).
- **E9**: `n_frames_out == n_frames_in`; duración del output = duración del input ±1 frame; stream de audio del output pasa `ffmpeg -f null` sin errores y su hash coincide con el extraído en E1.
- **Degradación** ([02 §11](./02-pipeline-spec.md)): con modelos ausentes (`KURAI_MODELS_DIR` vacío), el preset `alta-fidelidad` completa el job con warnings y produce exactamente el output de `detallado`-sin-cnn — se testea por igualdad de CharMatrix.

## 3. Métrica de flicker (cuantitativa, automatizada)

La métrica que gobierna la Etapa 7 y su escalera de complejidad ([04 §4](./04-ai-components.md)):

```
FCR (Flicker Change Rate) = cambios de char_idx por celda por segundo,
                            medido SOLO en celdas estáticas
```

- **Celdas estáticas** se definen por ground truth del fixture, no por heurística: los fixtures de flicker son clips sintéticos (fondo fijo + objeto en movimiento con máscara conocida) y un clip real de cámara en trípode con región estática anotada a mano una vez.
- **Umbrales**: FCR en zona estática con histéresis ≤ **0.05** (un cambio cada 20 s por celda) **[target]**; sin anti-flicker el baseline típico es >2. El preset con flow se acepta solo si su FCR en el fixture de paneo mejora ≥ 50% sobre histéresis sola — si no, el flow no entra ([IDEA.md](../IDEA.md), Discernment: si la métrica no baja frente a la versión sin IA, el componente se descarta).
- Corre en CI sobre los fixtures sintéticos (baratos); el clip real corre en `kurai bench`.

## 4. Set de evaluación curado y A/B

- `tests/fixtures/` contiene clips cortos (5-10 s, licencia libre, versionados con Git LFS o descargados por hash) cubriendo los 4 dominios definidos en IDEA.md: **retrato** (rostro, el caso clave de saliencia), **deporte** (movimiento rápido, estrés de E7), **paisaje** (gradientes amplios, estrés de dithering), **animación** (áreas planas + líneas duras, estrés de E5). Más 2 sintéticos: gradiente + círculo, y el fixture de flicker.
- **Protocolo A/B para gates de IA** (saliencia en Fase 1, CNN en Fase 2): `tools/ab_review.py` genera pares (con/sin el componente, orden aleatorio, sin etiquetas) y registra la elección. Gate: ≥ 60% de preferencia agregada sobre el set completo, con al menos 3 evaluadores (o el mismo evaluador en sesiones ciegas separadas, dado que el proyecto es unipersonal — imperfecto, pero infinitamente mejor que decidir con el par etiquetado a la vista).
- El resultado del A/B se registra en `docs/evaluations/` con fecha, commit y decisión — es el registro de Discernment del proyecto.

## 5. Performance como test

La regresión de performance es un fallo de test: `kurai bench` en modo `--check` compara contra el último resultado aceptado en `bench/results/` y falla si algún preset cae >10% en speed factor o alguna etapa excede su presupuesto de [05 §3](./05-performance-and-capacity.md). Corre pre-merge en la máquina de referencia (no hay CI con GPU equivalente; el bench es local por diseño — [ADR-001](./adr/ADR-001-local-first.md)).

## 6. Qué NO se testea automáticamente (y qué se hace en su lugar)

- **Calidad estética absoluta**: no hay métrica; se cubre con el protocolo A/B y el set curado.
- **El Scene Analyst** ([04 §6](./04-ai-components.md)): su output es una sugerencia opcional de un modelo no determinista — se testea el **contrato** (JSON parseable o descarte limpio, timeout respetado, auto-desactivación por VRAM) con Ollama mockeado, nunca el contenido de la sugerencia.
- **Compatibilidad universal de inputs** ("cualquier video"): imposible de enumerar; se cubre con el contrato de E1 (todo lo que ffmpeg decodifica) + un corpus creciente de casos reales que fallaron, cada bug de input se convierte en fixture.
