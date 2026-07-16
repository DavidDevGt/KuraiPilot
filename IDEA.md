# IDEA.md — Video a ASCII

> Pulido de la idea original: "un sistema donde le paso cualquier video y lo puede decodificar y renderizar como video ASCII". Ver [INVESTIGATION.md](./INVESTIGATION.md) para el respaldo técnico de cada decisión de esta sección.

## El pitch

Un sistema al que subís un video cualquiera y te devuelve el mismo video, pero renderizado enteramente en caracteres ASCII/Unicode — con audio intacto, listo para reproducir o compartir como cualquier archivo de video normal. No un script de terminal que hay que instalar y correr con flags: un producto con vista previa instantánea en el navegador y export final descargable.

Lo que lo diferencia de la docena de conversores de GitHub que ya existen (sección 2 de INVESTIGATION.md) no es "convertir mejor a gris" — el mapeo de luminancia a carácter está resuelto desde los 70s. El diferencial está en tres cosas que hoy nadie resuelve bien juntas: **densidad de detalle que respeta al sujeto de la escena, ausencia de parpadeo entre frames, y una experiencia de producto terminada** en vez de una herramienta de línea de comandos.

## La pregunta que ordena todo: ¿dónde entra la IA, y dónde estorba?

La idea original preguntaba "no sé si para que el producto quede excelente necesite inteligencia artificial". Esa es exactamente la pregunta que el framework 4D de Anthropic (Delegation, Description, Discernment, Diligence) está hecho para responder — no a nivel de "¿uso IA para escribir el código?", sino a nivel de arquitectura: **¿qué parte del pipeline se delega a un modelo, y cuál se resuelve con una función determinista?**

### Delegation — trazando la línea

La investigación (sección 4 de INVESTIGATION.md) deja un mapa bastante claro. Regla general: **todo lo que tiene una respuesta matemáticamente correcta y verificable se queda determinista; solo lo que requiere "criterio" sobre la escena se delega a un modelo.**

| Etapa del pipeline | ¿Quién la resuelve? | Por qué |
|---|---|---|
| Decode, resize, aspect-ratio de fuente monoespaciada | Determinista (ffmpeg/OpenCV) | Es aritmética. Un modelo aquí sería más lento y no más correcto. |
| Luminancia → carácter, rampa de densidad | Determinista | Función bien definida, sin ambigüedad. |
| Dithering (Floyd-Steinberg / Bayer) | Determinista | Algoritmo de 1976, sigue siendo el estándar. |
| Anti-flicker básico (histéresis por umbral) | Determinista | Resuelve ~80% del parpadeo a costo casi nulo — no hace falta optical flow para esto. |
| Selección de carácter por textura compleja (pelo, follaje) | **IA opcional** (CNN classifier) | Ganancia real pero marginal sobre edge-detection clásico; solo justificado en "modo calidad alta". |
| Densidad de detalle no uniforme (foco en el sujeto) | **IA** (saliency / segmentación class-agnostic) | Es la mejora con más impacto visual y ningún método determinista la resuelve bien — un humano *sabe* dónde mirar, una grilla fija no. |
| Anti-flicker avanzado en cámara en movimiento | **IA opcional** (optical flow) | Solo si la histéresis simple no basta (paneos, cámara en mano). |
| Upscaling previo | **IA opcional** (super-resolución) | Solo tiene sentido si la entrada es de menor resolución que la grilla de salida — si no, no aporta nada y se descarta automáticamente. |
| Generar el ASCII con un modelo generativo end-to-end | **Nunca** | No hay nada que "inventar": el output correcto es una función determinada de la entrada. Un generativo aquí sería más lento, menos fiel, y no reproducible. |

Esto también responde la pregunta original de forma directa: **no, el producto no necesita IA para funcionar y verse bien** — el pipeline determinista de la tabla ya produce algo comparable a las mejores herramientas existentes (Chafa, AsciiArtist). La IA se justifica solo en dos puntos concretos (saliencia y, opcionalmente, anti-flicker avanzado), y ahí sí mueve la aguja de "bueno" a "se siente compuesto por alguien, no generado por una fórmula".

### Description — cómo el sistema le habla al usuario (y viceversa)

En el 4D framework, Description es la competencia de comunicarse bien con la IA. Acá se invierte: el sistema es el que necesita comunicarse bien con el usuario, porque este no es un chat de prompt libre — es una herramienta con perillas. Las decisiones de diseño de UX deben traducir el pipeline técnico a un vocabulario que el usuario entienda sin necesitar leer INVESTIGATION.md:

- **Presets, no parámetros crudos**: "Retro terminal" (monocromo, rampa corta, sin saliencia), "Detallado" (color, rampa larga, saliencia activada), "Alta fidelidad" (saliencia + CNN de textura + anti-flicker avanzado, más lento).
- **Sliders con feedback visual inmediato** sobre densidad de caracteres y paleta, aprovechando que el pipeline base corre en tiempo real en WebGL (sección 3 de INVESTIGATION.md) para vista previa antes del export pesado.
- **Transparencia de costo**: si el usuario activa un modo con IA (saliencia, CNN, optical flow), el sistema debe indicar que el procesamiento será más lento — la comunicación tiene que ser honesta sobre el trade-off, no ocultarlo detrás de un preset con nombre bonito.

### Discernment — cómo sabemos que el resultado es bueno

El blog es enfático: no aceptar el resultado de la IA porque sí. Aplicado acá, eso significa que cada componente con IA necesita una forma de evaluarse que no sea "a mí me pareció que se veía bien":

- **Saliencia**: medir si la detección del sujeto principal coincide con el foco esperado en un set de video de prueba curado a mano (retratos, deportes, paisajes, animación) — no basta con probarlo en tres videos random y asumir que generaliza.
- **Anti-flicker**: métrica cuantificable — varianza del carácter elegido por celda entre frames consecutivos en zonas de la imagen que deberían estar estáticas. Si la varianza no baja frente a la versión sin IA, el componente no está aportando y se descarta, por más sofisticado que sea el método.
- **Comparación con el pipeline determinista como baseline**: cada modo con IA se evalúa contra la versión sin IA correspondiente. Si la diferencia perceptible no justifica el costo de cómputo extra, el modo no se vuelve default — se queda opcional o se descarta. Esto evita el error que describe el blog de "adoptar la etiqueta sin el impacto real": no se agrega IA al producto para poder decir que el producto usa IA.

### Diligence — responsabilidad sobre lo que el sistema produce

Esta es la parte que el pitch original no contemplaba y que vale la pena resolver antes de escalar el producto, no después:

- **Derechos sobre el video de entrada**: el usuario sube contenido que no necesariamente le pertenece. El sistema no debe alojar ni redistribuir el video original más allá del tiempo necesario para procesarlo, y los términos de uso deben dejar claro que la responsabilidad de tener derechos sobre el contenido es de quien lo sube.
- **Uso indebido para ocultar procedencia**: convertir un video a ASCII no es una técnica de evasión de detección de deepfakes ni de moderación de contenido, pero vale la pena tenerlo presente en el diseño (por ejemplo, no optimizar el producto para "que sea irreconocible al pasar por sistemas de verificación").
- **Consumo de cómputo**: los modos con IA (saliencia, CNN, optical flow) tienen huella energética real si se ofrecen a escala. Vale la pena que el preset por defecto sea el pipeline determinista (barato) y que los modos con IA sean elección explícita del usuario, no el comportamiento por defecto silencioso.
- **Transparencia**: si en algún punto se usa un modelo generativo para *cualquier* parte del resultado (no recomendado por el análisis de Delegation, pero por si cambia en el futuro), el output debería declararlo — coherente con el espíritu de Diligence del blog: asumir responsabilidad por lo que se produce con IA, no esconderlo.

## Arquitectura propuesta (alto nivel)

```
Input video
   │
   ▼
[1] Decode + extract audio (ffmpeg)
   │
   ▼
[2] Resize a grilla (corrige aspect-ratio de fuente monoespaciada)
   │
   ▼
[3] (opcional) Saliencia — modelo ligero class-agnostic (MobileSAM/U2Net)
   │     → mapa de densidad de detalle por región
   ▼
[4] Luminancia → carácter (rampa configurable por preset)
   │
   ▼
[5] (opcional) Refinamiento de carácter por textura — CNN classifier
   │
   ▼
[6] Dithering (Floyd-Steinberg / Bayer)
   │
   ▼
[7] Anti-flicker (histéresis simple, o optical flow si "alta fidelidad")
   │
   ▼
[8] Render: texto → imagen por frame (color/monocromo según preset)
   │
   ▼
[9] Re-encode con ffmpeg + mux audio original (-c:a copy)
   │
   ▼
Output: video renderizado + preview interactivo (WebGL, cliente)
```

Los pasos 3, 5 y 7 son los únicos con IA, y los tres son desactivables — el pipeline debe funcionar completo y producir un resultado sólido con todos ellos apagados.

## MVP y fases

**Fase 0 — Pipeline determinista (sin IA)**
Decode, grilla, luminancia, rampa configurable, dithering, anti-flicker por histéresis, export a video con audio. Esto solo ya iguala o supera a la mayoría de las herramientas de GitHub relevadas en la investigación. Vista previa en navegador vía Canvas/WebGL.

**Fase 1 — Saliencia**
Agregar detección de sujeto y densidad de detalle no uniforme. Es el componente de mayor impacto visual identificado en la investigación y el primer candidato real a usar IA.

**Fase 2 — Modo "alta fidelidad"**
CNN de refinamiento de carácter por textura + anti-flicker con optical flow, como preset opt-in más lento, evaluado explícitamente contra el baseline (Discernment) antes de promoverlo a default de nada.

**Fase 3 — Export enriquecido**
Formatos adicionales (asciinema `.cast`, HTML/Canvas embebible, SVG de frame destacado) según demanda real de usuarios, no especulativa.

## Métricas de éxito

- **Calidad**: comparación ciega A/B contra el pipeline sin IA en el set de prueba curado (retratos, deportes, paisajes, animación) — ¿la mayoría prefiere el resultado con saliencia?
- **Performance**: tiempo de procesamiento por segundo de video de entrada, en cada preset, para fijar expectativas realistas de UX ("esto puede tardar X minutos").
- **Fidelidad de audio**: audio del output debe ser bit-idéntico o perceptualmente idéntico al original (verificar que el remux no recodifica innecesariamente).

## Riesgos y preguntas abiertas

- ¿Procesamiento server-side, client-side (WebGL/WebGPU), o híbrido (preview en cliente, export pesado en servidor)? La investigación sugiere híbrido como el punto óptimo, pero define el costo de infraestructura del producto.
- ¿Cuál es el límite de duración/resolución de video soportado en el tier gratuito, dado el costo de cómputo de los modos con IA?
- ¿Vale la pena un modo "en vivo" (webcam a ASCII en tiempo real), dado que hay precedente (HasciiCam) y encaja naturalmente con el pipeline WebGL de la Fase 0?
- Falta validar con usuarios reales si "densidad de detalle por sujeto" (Fase 1) es percibido como mejora antes de invertir en el modelo de saliencia — es una hipótesis fuerte de este documento, no un hecho confirmado.
