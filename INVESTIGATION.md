# INVESTIGATION.md — Video a ASCII: estado del arte

> Investigación técnica de soporte para [IDEA.md](./IDEA.md). Cubre el pipeline clásico de conversión, herramientas existentes, el enfoque de tiempo real en GPU, y dónde la IA aporta valor real (vs. dónde es ruido).

## 1. El pipeline base (sin IA)

Todo conversor de video a ASCII, sea de 1990 o de este año, resuelve el mismo problema en el mismo orden:

1. **Decode**: extraer frames del video (ffmpeg, OpenCV, libav).
2. **Resize / sampling**: reducir cada frame a una grilla de N columnas × M filas — una celda por carácter de salida. Esta reducción es el paso que más determina la calidad percibida: si se reduce mal (nearest-neighbor puro) se pierde información que ningún mapeo posterior recupera.
3. **Cuantización de brillo**: convertir cada celda a luminancia (típicamente `Y = 0.2126R + 0.7152G + 0.0722B` o la aproximación clásica `0.299R+0.587G+0.114B`), porque el ojo humano pesa el verde mucho más que el azul.
4. **Mapeo carácter**: cada rango de brillo se asocia a un carácter de una "rampa" ordenada por densidad visual percibida, ej. `" .:-=+*#%@"` (10 niveles) o rampas más largas de 70 caracteres para más gradiente.
5. **Color (opcional)**: si el destino soporta ANSI/24-bit, se pinta cada carácter con el color original de la celda (foreground) y a veces también el fondo.
6. **Dithering (opcional)**: Floyd-Steinberg u ordered/Bayer dithering para difuminar el error de cuantización entre celdas vecinas y evitar bandas duras de brillo — técnica de 1976, sigue siendo el estándar de facto.
7. **Render + encode**: salida a terminal (stdout con secuencias ANSI), a imagen/video (quemar el texto sobre frames y re-encodear con ffmpeg), o a un canvas/HTML interactivo.

Ninguno de estos pasos requiere IA. Es procesamiento de señal determinista y barato en CPU.

## 2. Herramientas y proyectos existentes

| Proyecto | Enfoque | Notas relevantes |
|---|---|---|
| [video-to-ascii (joelibaceta)](https://github.com/joelibaceta/video-to-ascii) | Python + OpenCV | Reproduce video en terminal, mapea luminancia a caracteres, referencia canónica del pipeline básico. |
| [ASCII-Video (AlexEidt)](https://github.com/AlexEidt/ASCII-Video) | Go, tiempo real | Renderer explícitamente enfocado en performance — decodifica y renderiza en vivo, útil como referencia de arquitectura de bajo overhead. |
| [HasciiCam](https://dyne.org/software/hasciicam/) | C, captura V4L2 en vivo | Toma YUV420 y usa el canal Y directamente como luminancia (evita conversión RGB→gray), renderiza con el motor AA-lib. Precedente de 2000s del caso de uso "webcam a ASCII en vivo". |
| [Chafa](https://hpjansson.org/chafa/) | C, terminal moderno | El más sofisticado de los "clásicos": soporta Unicode más allá de ASCII puro, protocolos gráficos de terminal modernos (sixel, kitty), y dithering configurable (Floyd-Steinberg / Bayer / ninguno). Buen benchmark de calidad máxima sin IA. |
| [libcaca](http://caca.zoy.org/wiki/libcaca) | C, librería | Motor usado por mplayer `-vo caca` para reproducir video como ASCII/ANSI a pantalla completa; incluye dithering serpentine Floyd-Steinberg por canal. |
| [AsciiArtist (JuliaPoo)](https://github.com/JuliaPoo/AsciiArtist) | Python, edge-aware | En vez de solo luminancia, hace detección de bordes y elige el carácter cuya "forma" (trazo diagonal, vertical, horizontal) mejor calza con la orientación del borde local. Resultado notablemente más nítido en líneas y contornos que el mapeo por brillo puro — sigue siendo determinista, no ML. |
| [collidingscopes/ascii](https://collidingscopes.github.io/ascii/) (HN, ago 2024) | JS + Canvas, web | Conversor de video en el navegador, client-side, sin backend. Valida que el caso de uso "subo un video, lo veo en ASCII" es viable enteramente en cliente para videos cortos/resolución moderada. |
| [media-to-ascii (spoorn)](https://github.com/spoorn/media-to-ascii) | CLI, Rust | Soporta imagen y video, salida a archivo de video o consola — arquitectura de dos modos (preview interactivo vs. export) relevante para el producto. |

**Conclusión de esta comparación:** el espacio de "conversores clásicos" está saturado y bien resuelto para el caso CLI/terminal. El diferencial real de un producto nuevo no está en "convertir mejor a gris" — está en (a) UX de producto terminado (subís un video, no un script), (b) calidad de salida como *video* (no solo terminal), y (c) las mejoras que la IA sí puede aportar y que ningún proyecto de esta lista resuelve bien todavía (sección 4).

## 3. Tiempo real en GPU (shaders)

Para reproducir ASCII en vivo a 30-60fps sin cuello de botella en CPU, el patrón de la industria (creative coding / VJ tooling) es mover el mapeo carácter-por-celda a un fragment shader:

- **Enfoque de sprite sheet**: se pre-renderiza el set de caracteres de la rampa como un atlas de texturas. El shader calcula luminancia por celda y hace lookup de la textura correspondiente — sin generar geometría de texto en CPU por frame.
- **Enfoque procedural**: los caracteres se dibujan matemáticamente dentro del shader (SDF o funciones de distancia) en vez de usar bitmaps, lo que permite escalar sin pérdida y animar transiciones entre densidades.
- **Referencias de implementación**: [Three.js `AsciiEffect`](https://threejs.org) (el ejemplo canónico, pensado originalmente para escenas 3D), [Codrops "Efecto"](https://tympanus.net/codrops/2026/01/04/efecto-building-real-time-ascii-and-dithering-effects-with-webgl-shaders/) (WebGL + dithering combinados como post-proceso, 2026), [Codrops OGL ASCII shader](https://tympanus.net/codrops/2024/11/13/creating-an-ascii-shader-using-ogl/).
- Como cada celda es independiente (no hay dependencia entre vecinos salvo en dithering con error diffusion, que es inherentemente secuencial), el mapeo base paraleliza perfecto en GPU. El dithering Floyd-Steinberg clásico *no* paraleliza igual de bien porque cada píxel depende del error del anterior — para GPU conviene usar variantes ordered/Bayer o dithering por bloque.

**Implicación de arquitectura**: si el producto quiere una vista previa interactiva en el navegador (arrastrar un video y verlo en ASCII al instante, ajustar densidad/paleta con sliders), un pipeline WebGL/WebGPU en cliente es viable y ya tiene precedentes probados. El *export* final a archivo de video pesado (4K, minutos de duración) sigue siendo mejor resuelto server-side con ffmpeg.

## 4. Dónde la IA aporta algo que el pipeline clásico no puede

Esta es la pregunta central para el producto: ¿qué justifica IA más allá de "se ve más de moda"? Investigando papers y prácticas actuales, hay cuatro candidatos con justificación técnica real, y uno que es más hype que sustancia.

### 4.1 Selección de carácter por clasificación (CNN) — mejora marginal sobre edge-detection clásico
Existe investigación (["Evaluating Machine Learning Approaches for ASCII Art Generation", arXiv 2503.14375](https://arxiv.org/pdf/2503.14375); [proyecto ascii-net](https://github.com/a-metz/ascii-net)) que entrena una CNN para clasificar qué carácter representa mejor un parche de imagen, usando como target imágenes distorsionadas del propio carácter (para emular cómo se ve un borde real). Reportan ~89% de accuracy en la clasificación.

**Evaluación honesta**: el enfoque determinista de edge-detection + matching de orientación (como hace AsciiArtist, sección 2) ya captura la mayor parte de esta ganancia a costo casi cero. La CNN gana en casos de textura compleja (pelo, follaje, patrones) donde la orientación de borde único no basta. Es una mejora real pero de rendimiento decreciente — vale la pena como *modo de calidad alta opcional*, no como el pipeline por defecto.

### 4.2 Detección de saliencia / segmentación semántica — candidato con más impacto visual real
Ninguna herramienta de la sección 2 hace esto: **densidad de detalle no uniforme**. Hoy, todos los conversores usan grilla fija (misma resolución de caracteres en toda la imagen). Un modelo de *salient object detection* (no necesita ser segmentación semántica completa con clases — basta con un modelo class-agnostic tipo el de [Apple ML Research](https://machinelearning.apple.com/research/salient-object-segmentation) o SAM) puede identificar el sujeto principal (rostro, objeto en foco) y:
- Asignar más "resolución de caracteres" (celdas más pequeñas, rampa más larga) al sujeto.
- Simplificar el fondo (celdas grandes, rampa corta o directamente espacios).

Esto imita lo que un artista ASCII humano hace intuitivamente y es la diferencia más notoria entre un ASCII "mecánico" y uno que se siente compuesto. Es factible en tiempo casi real con modelos ligeros (MobileSAM, U2Net) corriendo una vez por frame o incluso cada N frames con tracking entre medio.

### 4.3 Coherencia temporal (el problema que nadie resuelve bien todavía)
Es el hallazgo más importante de esta investigación: **cuantizar cada frame de forma independiente produce "chisporroteo"** (flicker) — un píxel en el umbral entre dos niveles de brillo salta de carácter en cada frame aunque la escena esté casi estática, porque ruido de sensor o compresión empuja el valor de un lado a otro del umbral. Ninguno de los proyectos listados en la sección 2 lo menciona como problema resuelto.

La literatura de *video style transfer* (no específica de ASCII, pero directamente aplicable) sí lo resuelve: se usa optical flow para calcular cómo se mueve cada región entre frame t-1 y t, se "warpea" (proyecta) el resultado anterior sobre el frame actual, y se penaliza/suaviza la diferencia entre el resultado esperado por warping y el resultado nuevo (ver ["Coherent Online Video Style Transfer"](https://arxiv.org/pdf/1703.09211), ["Fast Coherent Video Style Transfer via Flow Errors Reduction"](https://www.mdpi.com/2076-3417/14/6/2630)). Aplicado a ASCII, esto se traduce en: no cambiar el carácter de una celda salvo que el cambio de brillo/contenido supere un umbral de histéresis ponderado por el movimiento óptico local.

**Nota importante**: una versión *simple* y no-IA de esto (histéresis por umbral simple, sin optical flow) ya elimina el 80% del flicker con costo casi nulo. El optical flow completo es la versión "IA" que vale la pena solo si el histéresis simple no basta (cámara en movimiento, paneos).

### 4.4 Super-resolución / upscaling previo — útil pero de valor acotado
Si el video de entrada es de baja resolución o muy comprimido, un modelo de super-resolución (Real-ESRGAN y similares) antes de la reducción a grilla puede recuperar bordes que el bloque de compresión JPEG/H.264 difuminó. Tiene sentido como paso *opcional* solo cuando la resolución de entrada es menor a la grilla de salida deseada — en el caso contrario (video 4K reducido a 200 columnas) no aporta nada, porque ya se está perdiendo información, no recuperándola.

### 4.5 Lo que NO vale la pena (el caso de "IA porque sí")
- **Generar el video ASCII con un modelo generativo (difusión/GAN) en vez de mapear determinísticamente**: no hay ganancia — el output "correcto" de ASCII es una función bien definida de la imagen de entrada, no algo con ambigüedad creativa que un modelo generativo deba "inventar". Sería más lento, menos fiel al original, y no determinista entre corridas.
- **LLM para "describir" la escena y elegir estilo**: interesante como *feature de producto* (ver IDEA.md, capa de Description), pero no resuelve nada del pipeline de renderizado en sí.

## 5. Formatos de salida a considerar

| Formato | Caso de uso | Complejidad |
|---|---|---|
| Terminal en vivo (ANSI stream) | Demos, CLI tool, nostalgia retro | Baja — ya resuelto por herramientas existentes |
| Video renderizado (mp4/webm con texto "quemado" sobre frames) | Compartir en redes, el caso de uso principal del producto | Media — requiere renderizar texto a imagen por frame y re-encodear con audio original |
| Grabación de sesión de terminal (asciinema `.cast`) | Embeber en blogs/docs, reproducción fiel de texto real (no video) | Baja, formato ya estandarizado |
| HTML/Canvas interactivo exportable | Portfolio, arte generativo, permite reproducir con controles | Media-alta, pero reutiliza el pipeline WebGL de la sección 3 |
| SVG por frame / spritesheet | Impresión, arte estático de un frame destacado | Baja |

El caso de uso descrito por el usuario ("le paso cualquier video y lo convierte a un video renderizado en ASCII") apunta principalmente al segundo formato (video real, reproducible en cualquier player), con el cuarto como diferenciador de producto para la vista previa web.

## 6. Riesgos técnicos a anticipar

- **Costo de cómputo para 4K / videos largos**: procesar frame por frame a resolución completa antes de reducir es caro. Conviene reducir resolución *antes* de cualquier paso costoso (edge detection, saliencia), no después.
- **Audio**: es fácil olvidarlo — el pipeline debe extraer y re-muxear el audio original sin recodificarlo innecesariamente (ffmpeg `-c:a copy`).
- **Legibilidad vs. fidelidad**: hay una tensión de producto, no solo técnica, entre "que se vea como el video original" y "que se vea bien como ASCII" — rampas de caracteres más largas y color agregan fidelidad pero reducen la estética "retro terminal" que mucha gente busca. Debe ser un parámetro expuesto al usuario, no una decisión fija del sistema.
- **Fuentes monoespaciadas y aspect ratio**: los caracteres no son cuadrados (son más altos que anchos, ~1:2), así que el muestreo de la grilla debe corregir el aspect ratio o la imagen sale deformada — error común en implementaciones ingenuas.

## Fuentes consultadas

- [video-to-ascii (joelibaceta)](https://github.com/joelibaceta/video-to-ascii)
- [ASCII-Video (AlexEidt)](https://github.com/AlexEidt/ASCII-Video)
- [HasciiCam](https://dyne.org/software/hasciicam/)
- [Chafa: Terminal Graphics for the 21st Century](https://hpjansson.org/chafa/)
- [Libcaca study — Colour dithering](http://caca.zoy.org/study/part6.html)
- [AsciiArtist (JuliaPoo)](https://github.com/JuliaPoo/AsciiArtist)
- [Show HN: turn videos into ASCII art (collidingscopes)](https://news.ycombinator.com/item?id=41389326)
- [media-to-ascii (spoorn)](https://github.com/spoorn/media-to-ascii)
- [Efecto: Building Real-Time ASCII and Dithering Effects with WebGL Shaders — Codrops](https://tympanus.net/codrops/2026/01/04/efecto-building-real-time-ascii-and-dithering-effects-with-webgl-shaders/)
- [Creating an ASCII Shader Using OGL — Codrops](https://tympanus.net/codrops/2024/11/13/creating-an-ascii-shader-using-ogl/)
- [Evaluating Machine Learning Approaches for ASCII Art Generation (arXiv 2503.14375)](https://arxiv.org/pdf/2503.14375)
- [ascii-net (a-metz)](https://github.com/a-metz/ascii-net)
- [Fast Class-Agnostic Salient Object Segmentation — Apple ML Research](https://machinelearning.apple.com/research/salient-object-segmentation)
- [Coherent Online Video Style Transfer (arXiv 1703.09211)](https://arxiv.org/pdf/1703.09211)
- [Fast Coherent Video Style Transfer via Flow Errors Reduction](https://www.mdpi.com/2076-3417/14/6/2630)
- [Floyd–Steinberg dithering — Wikipedia](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering)
