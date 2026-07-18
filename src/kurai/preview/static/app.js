// Cliente del preview (docs/03 §3): recibe la CharMatrix por WebSocket y la
// proyecta con WebGL2 en UN pass — fullscreen quad + 3 texturas (char_idx,
// fg, atlas de glifos). Sin build step ni dependencias: ES modules vanilla.

const MONO = [102 / 255, 1.0, 102 / 255];

const VS = `#version 300 es
out vec2 v_uv;
void main() {
  // Triángulo que cubre la pantalla (sin buffers)
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  v_uv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
precision highp usampler2D;
in vec2 v_uv;
out vec4 outColor;
uniform usampler2D u_charIdx;   // (rows, cols) R8UI
uniform sampler2D u_fg;         // (rows, cols) RGB
uniform sampler2D u_atlas;      // (glyphH, nGlyphs*glyphW) R8
uniform vec2 u_grid;            // cols, rows
uniform vec2 u_glyph;           // glyphW, glyphH
uniform float u_nGlyphs;
uniform vec3 u_mono;            // rgb, o -1 si modo fg
void main() {
  vec2 cell = floor(v_uv * u_grid);
  vec2 inCell = fract(v_uv * u_grid);           // 0..1 dentro de la celda
  ivec2 cellI = ivec2(cell);
  uint idx = texelFetch(u_charIdx, cellI, 0).r;
  // Píxel del glifo dentro del atlas horizontal
  vec2 g = (vec2(float(idx), 0.0) + inCell) * u_glyph;
  float ink = texelFetch(u_atlas, ivec2(int(g.x), int(g.y)), 0).r;
  vec3 tint = (u_mono.r < 0.0) ? texelFetch(u_fg, cellI, 0).rgb : u_mono;
  outColor = vec4(tint * ink, 1.0);
}`;

class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext("webgl2");
    if (!gl) throw new Error("WebGL2 no disponible en este navegador");
    this.gl = gl;
    this.canvas = canvas;
    const prog = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, VS], [gl.FRAGMENT_SHADER, FS]]) {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS))
        throw new Error(gl.getShaderInfoLog(sh));
      gl.attachShader(prog, sh);
    }
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    this.u = {};
    for (const name of ["u_charIdx", "u_fg", "u_atlas", "u_grid", "u_glyph", "u_nGlyphs", "u_mono"])
      this.u[name] = gl.getUniformLocation(prog, name);
    this.texChar = this._makeTex();
    this.texFg = this._makeTex();
    this.texAtlas = this._makeTex();
    gl.uniform1i(this.u.u_charIdx, 0);
    gl.uniform1i(this.u.u_fg, 1);
    gl.uniform1i(this.u.u_atlas, 2);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  }

  _makeTex() {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  setMeta(meta, atlasBytes) {
    const gl = this.gl;
    this.meta = meta;
    this.canvas.width = meta.cols * meta.glyph_w;
    this.canvas.height = meta.rows * meta.glyph_h;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.uniform2f(this.u.u_grid, meta.cols, meta.rows);
    gl.uniform2f(this.u.u_glyph, meta.glyph_w, meta.glyph_h);
    gl.uniform1f(this.u.u_nGlyphs, meta.n_glyphs);
    // Atlas (n, h, w) → tira horizontal (h, n*w) para texelFetch simple
    const { n_glyphs: n, glyph_h: gh, glyph_w: gw } = meta;
    const strip = new Uint8Array(gh * n * gw);
    for (let g = 0; g < n; g++)
      for (let y = 0; y < gh; y++)
        for (let x = 0; x < gw; x++)
          strip[y * n * gw + g * gw + x] = atlasBytes[g * gh * gw + y * gw + x];
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.texAtlas);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, n * gw, gh, 0, gl.RED, gl.UNSIGNED_BYTE, strip);
    this.setColorMode(meta.color_mode);
  }

  setColorMode(mode) {
    const gl = this.gl;
    if (mode === "mono") gl.uniform3f(this.u.u_mono, ...MONO);
    else gl.uniform3f(this.u.u_mono, -1, -1, -1);
  }

  drawFrame(rows, cols, charIdx, fg) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texChar);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8UI, cols, rows, 0, gl.RED_INTEGER, gl.UNSIGNED_BYTE, charIdx);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.texFg);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, cols, rows, 0, gl.RGB, gl.UNSIGNED_BYTE, fg);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

// ------------------------------------------------------------------ wiring

const $ = (id) => document.getElementById(id);
const renderer = new Renderer($("canvas"));
const ws = new WebSocket(`ws://${location.host}/ws`);
ws.binaryType = "arraybuffer";

let meta = null;
let playing = false;
let lastConfigSent = 0;

function sendConfig(partial) {
  lastConfigSent = performance.now();
  ws.send(JSON.stringify({ type: "config", ...partial }));
}

ws.onmessage = (ev) => {
  if (typeof ev.data === "string") {
    const msg = JSON.parse(ev.data);
    if (msg.type === "meta") {
      meta = msg;
      const atlas = Uint8Array.from(atob(msg.atlas_b64), (c) => c.charCodeAt(0));
      renderer.setMeta(msg, atlas);
      $("seek").max = msg.n_frames - 1;
      $("status").textContent = `${msg.cols}×${msg.rows} celdas · ${msg.fps.toFixed(2)} fps`;
    } else if (msg.type === "state") {
      playing = msg.playing;
      $("playpause").textContent = playing ? "⏸" : "▶";
      $("seek").value = msg.frame;
    }
  } else {
    const dv = new DataView(ev.data);
    $("seek").value = dv.getUint32(0, true); // el slider sigue al playback
    const rows = dv.getUint16(4, true);
    const cols = dv.getUint16(6, true);
    const n = rows * cols;
    const charIdx = new Uint8Array(ev.data, 9, n);
    const fg = new Uint8Array(ev.data, 9 + n, n * 3);
    renderer.drawFrame(rows, cols, charIdx, fg);
    if (lastConfigSent) {
      const ms = (performance.now() - lastConfigSent).toFixed(0);
      $("status").textContent = `${meta.cols}×${meta.rows} celdas · ajuste reflejado en ${ms} ms`;
      lastConfigSent = 0;
    }
  }
};

ws.onclose = () => { $("status").textContent = "desconectado — recargá la página"; };

$("playpause").onclick = () => ws.send(JSON.stringify({ type: playing ? "pause" : "play" }));
$("seek").oninput = (e) => ws.send(JSON.stringify({ type: "seek", frame: +e.target.value }));
$("cols").oninput = (e) => { $("cols-val").textContent = e.target.value; sendConfig({ cols: +e.target.value }); };
$("gamma").oninput = (e) => { $("gamma-val").textContent = e.target.value; sendConfig({ gamma: +e.target.value }); };
$("ramp").onchange = (e) => sendConfig({ ramp: e.target.value });
$("color").onchange = (e) => { sendConfig({ color: e.target.value }); };
$("cols-val").textContent = $("cols").value;
$("gamma-val").textContent = $("gamma").value;
