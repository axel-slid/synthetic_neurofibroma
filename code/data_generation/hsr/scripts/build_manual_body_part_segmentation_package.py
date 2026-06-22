#!/usr/bin/env python3
"""Build an offline browser package for manual HSR body-part segmentation."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SEGMENTATION_ROOT = ROOT / "data" / "hsr" / "body_part_segmentation"
DEFAULT_MESH_ROOT = ROOT / "data" / "hsr" / "visualizations" / "meshes"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "hsr" / "manual_body_part_segmentation_package"

LABEL_NAMES = ["front", "back", "face", "arms", "hands", "legs", "feet", "clothes"]
LABEL_COLORS = {
    "front": "#00A6A6",
    "back": "#7B61FF",
    "face": "#FF5A36",
    "arms": "#2CA02C",
    "hands": "#F2C94C",
    "legs": "#1F77B4",
    "feet": "#D946EF",
    "clothes": "#8A8A8A",
}


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def compact_float_list(values: np.ndarray, decimals: int) -> list[float]:
    return np.round(values.astype(np.float32), decimals=decimals).reshape(-1).tolist()


def scan_payload(npz_path: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=False)
    scan_id = str(data["scan_id"])
    vertex_colors = data["vertex_colors"].astype(np.uint8)
    vertex_labels = data["vertex_labels"].astype(np.uint8)
    return {
        "scan_id": scan_id,
        "vertex_count": int(len(data["vertices"])),
        "triangle_count": int(len(data["triangles"])),
        "vertices": compact_float_list(data["vertices"], decimals=5),
        "triangles": data["triangles"].astype(np.uint32).reshape(-1).tolist(),
        "normals": compact_float_list(data["vertex_normals"], decimals=5),
        "colors": vertex_colors.reshape(-1).tolist(),
        "labels": vertex_labels.tolist(),
        "front_sign": int(data["front_sign"]),
        "height": float(data["height"]),
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def app_index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HSR Manual Body-Part Segmentation</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div id="app">
    <aside id="sidebar">
      <header>
        <h1>HSR Segmentation</h1>
        <p>Paint body-part labels on the scan vertices, then export annotations.</p>
      </header>

      <label class="field">
        <span>Scan</span>
        <select id="scanSelect"></select>
      </label>

      <div class="segmented" role="group" aria-label="Interaction mode">
        <button id="rotateMode" type="button">Rotate</button>
        <button id="paintMode" type="button">Paint</button>
      </div>

      <label class="field">
        <span>View</span>
        <select id="viewMode">
          <option value="mixed">Labels over texture</option>
          <option value="labels">Labels only</option>
          <option value="texture">Texture only</option>
        </select>
      </label>

      <label class="field">
        <span>Brush radius <output id="brushOut"></output></span>
        <input id="brushRadius" type="range" min="4" max="90" value="26">
      </label>

      <label class="field">
        <span>Overlay opacity <output id="opacityOut"></output></span>
        <input id="overlayOpacity" type="range" min="0" max="1" step="0.05" value="0.55">
      </label>

      <section>
        <h2>Labels</h2>
        <div id="labelButtons"></div>
      </section>

      <section class="actions">
        <button id="undoBtn" type="button">Undo</button>
        <button id="redoBtn" type="button">Redo</button>
        <button id="resetScanBtn" type="button">Reset scan</button>
      </section>

      <section class="actions">
        <button id="exportBtn" type="button">Export annotations JSON</button>
        <label class="fileButton">
          Load saved JSON
          <input id="importInput" type="file" accept="application/json,.json">
        </label>
      </section>

      <section>
        <h2>Status</h2>
        <pre id="status"></pre>
      </section>
    </aside>

    <main>
      <canvas id="viewport"></canvas>
      <canvas id="overlay"></canvas>
      <div id="hud">
        <span>Wheel zoom</span>
        <span>Drag rotate</span>
        <span>Paint mode paints the visible mesh surface</span>
        <span>Drag background to rotate</span>
        <span>Keys 1-8 labels, Z undo</span>
      </div>
    </main>
  </div>

  <script src="app_data.js"></script>
  <script src="app.js"></script>
</body>
</html>
"""


def app_styles_css() -> str:
    return r""":root {
  color-scheme: light;
  --bg: #f4f6f8;
  --panel: #ffffff;
  --text: #161b22;
  --muted: #637083;
  --line: #d9e0e8;
  --accent: #1f6feb;
}

* { box-sizing: border-box; }
html, body, #app { height: 100%; margin: 0; }
body {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--text);
  background: var(--bg);
}

#app {
  display: grid;
  grid-template-columns: 330px minmax(0, 1fr);
}

#sidebar {
  overflow: auto;
  padding: 18px;
  background: var(--panel);
  border-right: 1px solid var(--line);
}

h1 { font-size: 20px; margin: 0 0 6px; }
h2 { font-size: 13px; margin: 22px 0 10px; text-transform: uppercase; color: var(--muted); letter-spacing: .04em; }
p { margin: 0 0 16px; color: var(--muted); line-height: 1.35; }

main {
  position: relative;
  min-width: 0;
  overflow: hidden;
  background: #eef2f6;
}

#viewport,
#overlay {
  display: block;
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}

#overlay {
  cursor: grab;
}

#overlay.painting { cursor: crosshair; }

#hud {
  position: absolute;
  left: 14px;
  bottom: 14px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  pointer-events: none;
}

#hud span {
  font-size: 12px;
  color: #334155;
  background: rgba(255, 255, 255, .9);
  border: 1px solid rgba(148, 163, 184, .45);
  border-radius: 6px;
  padding: 5px 8px;
}

.field {
  display: grid;
  gap: 6px;
  margin: 14px 0;
  font-size: 13px;
  color: var(--muted);
}

select, button, .fileButton {
  font: inherit;
  border: 1px solid var(--line);
  background: #fff;
  color: var(--text);
  border-radius: 6px;
  min-height: 34px;
}

select { padding: 0 10px; width: 100%; }
button, .fileButton {
  padding: 7px 10px;
  cursor: pointer;
}
button:hover, .fileButton:hover { border-color: #9fb2c7; }
button.active { background: var(--accent); color: #fff; border-color: var(--accent); }

.segmented {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin: 12px 0 4px;
}

#labelButtons {
  display: grid;
  gap: 8px;
}

.labelButton {
  display: grid;
  grid-template-columns: 18px 1fr auto;
  align-items: center;
  gap: 9px;
  text-align: left;
}

.swatch {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  border: 1px solid rgba(0, 0, 0, .18);
}

.count {
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  font-size: 12px;
}

.actions {
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
}

.fileButton {
  display: block;
  text-align: center;
}
.fileButton input { display: none; }

pre {
  white-space: pre-wrap;
  font-size: 12px;
  line-height: 1.35;
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
  min-height: 92px;
  margin: 0;
}
"""


def app_js() -> str:
    return r"""(() => {
  'use strict';

  const pkg = window.BODY_PART_SEGMENTATION_PACKAGE;
  const labelNames = pkg.labels;
  const labelColors = pkg.label_colors;
  const labelRgb = labelNames.map(name => hexToRgb01(labelColors[name]));
  const labelCss = labelNames.map(name => labelColors[name]);

  const canvas = document.getElementById('viewport');
  const overlay = document.getElementById('overlay');
  const overlayCtx = overlay.getContext('2d');
  const scanSelect = document.getElementById('scanSelect');
  const viewMode = document.getElementById('viewMode');
  const brushRadius = document.getElementById('brushRadius');
  const brushOut = document.getElementById('brushOut');
  const overlayOpacity = document.getElementById('overlayOpacity');
  const opacityOut = document.getElementById('opacityOut');
  const labelButtons = document.getElementById('labelButtons');
  const status = document.getElementById('status');
  const rotateMode = document.getElementById('rotateMode');
  const paintMode = document.getElementById('paintMode');

  const gl = canvas.getContext('webgl2', { antialias: true }) ||
    canvas.getContext('webgl', { antialias: true });
  if (!gl) {
    status.textContent = 'This browser does not provide WebGL, so the filled mesh viewer cannot start.';
    throw new Error('WebGL is unavailable.');
  }
  const isWebGL2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;
  const uintIndexExtension = isWebGL2 ? true : gl.getExtension('OES_element_index_uint');
  if (!uintIndexExtension) {
    status.textContent = 'This browser cannot draw the scan meshes because unsigned integer indices are unavailable.';
    throw new Error('OES_element_index_uint is unavailable.');
  }

  let state = null;
  state = {
    scans: pkg.scans.map(prepareScan),
    scanIndex: 0,
    activeLabel: 0,
    mode: 'paint',
    view: 'mixed',
    brushRadius: Number(brushRadius.value),
    overlayOpacity: Number(overlayOpacity.value),
    yaw: 0,
    pitch: 0,
    zoom: 1,
    panX: 0,
    panY: 0,
    pointer: null,
    dragging: false,
    dragMode: null,
    lastX: 0,
    lastY: 0,
    undo: [],
    redo: [],
    projected: null,
  };

  const program = createProgram(gl, `
    attribute vec3 aPosition;
    attribute vec3 aNormal;
    attribute vec3 aColor;
    uniform mat4 uMvp;
    uniform mat3 uNormalMat;
    varying vec3 vColor;
    void main() {
      vec3 n = normalize(uNormalMat * aNormal);
      float light = 0.68 + 0.32 * max(dot(n, normalize(vec3(0.25, 0.45, 0.86))), 0.0);
      vColor = aColor * light;
      gl_Position = uMvp * vec4(aPosition, 1.0);
    }
  `, `
    precision mediump float;
    varying vec3 vColor;
    void main() {
      gl_FragColor = vec4(vColor, 1.0);
    }
  `);

  const locations = {
    position: gl.getAttribLocation(program, 'aPosition'),
    normal: gl.getAttribLocation(program, 'aNormal'),
    color: gl.getAttribLocation(program, 'aColor'),
    mvp: gl.getUniformLocation(program, 'uMvp'),
    normalMat: gl.getUniformLocation(program, 'uNormalMat'),
  };

  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.disable(gl.CULL_FACE);

  function hexToRgb01(hex) {
    const clean = hex.replace('#', '');
    return [
      parseInt(clean.slice(0, 2), 16) / 255,
      parseInt(clean.slice(2, 4), 16) / 255,
      parseInt(clean.slice(4, 6), 16) / 255,
    ];
  }

  function prepareScan(raw) {
    const sourceVertices = Float32Array.from(raw.vertices);
    const triangles = Uint32Array.from(raw.triangles);
    const colors = Uint8Array.from(raw.colors);
    const labels = Uint8Array.from(raw.labels);
    const initialLabels = Uint8Array.from(raw.labels);
    const n = raw.vertex_count;
    const center = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      center[0] += sourceVertices[i * 3];
      center[1] += sourceVertices[i * 3 + 1];
      center[2] += sourceVertices[i * 3 + 2];
    }
    center[0] /= n; center[1] /= n; center[2] /= n;
    let maxSpan = 0;
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
      for (let axis = 0; axis < 3; axis++) {
        const v = sourceVertices[i * 3 + axis];
        if (v < min[axis]) min[axis] = v;
        if (v > max[axis]) max[axis] = v;
      }
    }
    for (let axis = 0; axis < 3; axis++) maxSpan = Math.max(maxSpan, max[axis] - min[axis]);
    const positions = new Float32Array(sourceVertices.length);
    for (let i = 0; i < n; i++) {
      positions[i * 3] = (sourceVertices[i * 3] - center[0]) / maxSpan;
      positions[i * 3 + 1] = (sourceVertices[i * 3 + 2] - center[2]) / maxSpan;
      positions[i * 3 + 2] = -(sourceVertices[i * 3 + 1] - center[1]) / maxSpan;
    }

    let normals;
    if (Array.isArray(raw.normals) && raw.normals.length === sourceVertices.length) {
      const sourceNormals = Float32Array.from(raw.normals);
      normals = new Float32Array(sourceNormals.length);
      for (let i = 0; i < n; i++) {
        const nx = sourceNormals[i * 3];
        const ny = sourceNormals[i * 3 + 1];
        const nz = sourceNormals[i * 3 + 2];
        const len = Math.hypot(nx, ny, nz) || 1;
        normals[i * 3] = nx / len;
        normals[i * 3 + 1] = nz / len;
        normals[i * 3 + 2] = -ny / len;
      }
    } else {
      normals = computeNormals(positions, triangles);
    }
    const colorData = new Float32Array(n * 3);
    const scan = { ...raw, sourceVertices, positions, triangles, normals, colors, labels, initialLabels, colorData, buffers: null };
    refreshScanColors(scan, false);
    return scan;
  }

  function activeScan() { return state.scans[state.scanIndex]; }

  function createShader(glContext, type, source) {
    const shader = glContext.createShader(type);
    glContext.shaderSource(shader, source);
    glContext.compileShader(shader);
    if (!glContext.getShaderParameter(shader, glContext.COMPILE_STATUS)) {
      const message = glContext.getShaderInfoLog(shader);
      glContext.deleteShader(shader);
      throw new Error(message);
    }
    return shader;
  }

  function createProgram(glContext, vertexSource, fragmentSource) {
    const vertexShader = createShader(glContext, glContext.VERTEX_SHADER, vertexSource);
    const fragmentShader = createShader(glContext, glContext.FRAGMENT_SHADER, fragmentSource);
    const linkedProgram = glContext.createProgram();
    glContext.attachShader(linkedProgram, vertexShader);
    glContext.attachShader(linkedProgram, fragmentShader);
    glContext.linkProgram(linkedProgram);
    glContext.deleteShader(vertexShader);
    glContext.deleteShader(fragmentShader);
    if (!glContext.getProgramParameter(linkedProgram, glContext.LINK_STATUS)) {
      const message = glContext.getProgramInfoLog(linkedProgram);
      glContext.deleteProgram(linkedProgram);
      throw new Error(message);
    }
    return linkedProgram;
  }

  function computeNormals(positions, triangles) {
    const normals = new Float32Array(positions.length);
    for (let t = 0; t < triangles.length; t += 3) {
      const ia = triangles[t] * 3;
      const ib = triangles[t + 1] * 3;
      const ic = triangles[t + 2] * 3;
      const ax = positions[ia], ay = positions[ia + 1], az = positions[ia + 2];
      const bx = positions[ib], by = positions[ib + 1], bz = positions[ib + 2];
      const cx = positions[ic], cy = positions[ic + 1], cz = positions[ic + 2];
      const abx = bx - ax, aby = by - ay, abz = bz - az;
      const acx = cx - ax, acy = cy - ay, acz = cz - az;
      const nx = aby * acz - abz * acy;
      const ny = abz * acx - abx * acz;
      const nz = abx * acy - aby * acx;
      normals[ia] += nx; normals[ia + 1] += ny; normals[ia + 2] += nz;
      normals[ib] += nx; normals[ib + 1] += ny; normals[ib + 2] += nz;
      normals[ic] += nx; normals[ic + 1] += ny; normals[ic + 2] += nz;
    }
    for (let i = 0; i < normals.length; i += 3) {
      const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
      normals[i] /= len;
      normals[i + 1] /= len;
      normals[i + 2] /= len;
    }
    return normals;
  }

  function uploadScan(scan) {
    if (scan.buffers) return;
    scan.buffers = {
      position: gl.createBuffer(),
      normal: gl.createBuffer(),
      color: gl.createBuffer(),
      index: gl.createBuffer(),
    };

    gl.bindBuffer(gl.ARRAY_BUFFER, scan.buffers.position);
    gl.bufferData(gl.ARRAY_BUFFER, scan.positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, scan.buffers.normal);
    gl.bufferData(gl.ARRAY_BUFFER, scan.normals, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, scan.buffers.color);
    gl.bufferData(gl.ARRAY_BUFFER, scan.colorData, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, scan.buffers.index);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, scan.triangles, gl.STATIC_DRAW);
  }

  function refreshScanColors(scan, updateGpu = true) {
    const opacity = state ? state.overlayOpacity : 0.55;
    for (let i = 0; i < scan.vertex_count; i++) {
      const j = i * 3;
      const tr = scan.colors[j] / 255;
      const tg = scan.colors[j + 1] / 255;
      const tb = scan.colors[j + 2] / 255;
      const label = scan.labels[i];
      const lr = labelRgb[label][0];
      const lg = labelRgb[label][1];
      const lb = labelRgb[label][2];
      if (state && state.view === 'texture') {
        scan.colorData[j] = tr;
        scan.colorData[j + 1] = tg;
        scan.colorData[j + 2] = tb;
      } else if (state && state.view === 'labels') {
        scan.colorData[j] = lr;
        scan.colorData[j + 1] = lg;
        scan.colorData[j + 2] = lb;
      } else {
        scan.colorData[j] = tr * (1 - opacity) + lr * opacity;
        scan.colorData[j + 1] = tg * (1 - opacity) + lg * opacity;
        scan.colorData[j + 2] = tb * (1 - opacity) + lb * opacity;
      }
    }
    if (updateGpu && scan.buffers) {
      gl.bindBuffer(gl.ARRAY_BUFFER, scan.buffers.color);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, scan.colorData);
    }
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    overlay.width = canvas.width;
    overlay.height = canvas.height;
    overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    state.projected = null;
    render();
  }

  function bindAttribute(buffer, location, size) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
  }

  function render() {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const scan = activeScan();
    uploadScan(scan);
    const matrices = makeMatrices();

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.93, 0.95, 0.97, 1.0);
    gl.clearDepth(1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(program);
    gl.uniformMatrix4fv(locations.mvp, false, matrices.mvp);
    gl.uniformMatrix3fv(locations.normalMat, false, matrices.normalMat);

    bindAttribute(scan.buffers.position, locations.position, 3);
    bindAttribute(scan.buffers.normal, locations.normal, 3);
    bindAttribute(scan.buffers.color, locations.color, 3);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, scan.buffers.index);
    gl.drawElements(gl.TRIANGLES, scan.triangles.length, gl.UNSIGNED_INT, 0);

    drawOverlay();
    updateStatus();
  }

  function drawOverlay() {
    const rect = canvas.getBoundingClientRect();
    overlayCtx.clearRect(0, 0, rect.width, rect.height);
    if (state.mode !== 'paint' || !state.pointer) return;
    overlayCtx.beginPath();
    overlayCtx.arc(state.pointer.x, state.pointer.y, state.brushRadius, 0, Math.PI * 2);
    overlayCtx.strokeStyle = labelCss[state.activeLabel];
    overlayCtx.lineWidth = 2;
    overlayCtx.stroke();
    overlayCtx.beginPath();
    overlayCtx.arc(state.pointer.x, state.pointer.y, 2.5, 0, Math.PI * 2);
    overlayCtx.fillStyle = labelCss[state.activeLabel];
    overlayCtx.fill();
  }

  function project(scan) {
    const rect = canvas.getBoundingClientRect();
    const matrices = makeMatrices();
    const n = scan.vertex_count;
    const sx = new Float32Array(n);
    const sy = new Float32Array(n);
    const depth = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      const j = i * 3;
      const clip = transformPoint(matrices.mvp, scan.positions[j], scan.positions[j + 1], scan.positions[j + 2]);
      const invW = 1 / clip[3];
      const ndcX = clip[0] * invW;
      const ndcY = clip[1] * invW;
      const ndcZ = clip[2] * invW;
      sx[i] = (ndcX * 0.5 + 0.5) * rect.width;
      sy[i] = (1 - (ndcY * 0.5 + 0.5)) * rect.height;
      depth[i] = ndcZ;
    }
    state.projected = { sx, sy, depth };
    return state.projected;
  }

  function hasSurfaceAt(x, y) {
    const scan = activeScan();
    const proj = state.projected || project(scan);
    const hitRadius = Math.max(7, Math.min(14, state.brushRadius * 0.45));
    const r2 = hitRadius * hitRadius;
    let nearestDepth = Infinity;
    for (let i = 0; i < scan.vertex_count; i++) {
      const dx = proj.sx[i] - x;
      const dy = proj.sy[i] - y;
      if (dx * dx + dy * dy <= r2 && proj.depth[i] < nearestDepth) nearestDepth = proj.depth[i];
    }
    return Number.isFinite(nearestDepth);
  }

  function transformPoint(m, x, y, z) {
    return [
      m[0] * x + m[4] * y + m[8] * z + m[12],
      m[1] * x + m[5] * y + m[9] * z + m[13],
      m[2] * x + m[6] * y + m[10] * z + m[14],
      m[3] * x + m[7] * y + m[11] * z + m[15],
    ];
  }

  function mat4Identity() {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  }

  function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        out[col * 4 + row] =
          a[0 * 4 + row] * b[col * 4 + 0] +
          a[1 * 4 + row] * b[col * 4 + 1] +
          a[2 * 4 + row] * b[col * 4 + 2] +
          a[3 * 4 + row] * b[col * 4 + 3];
      }
    }
    return out;
  }

  function mat4Translation(tx, ty, tz) {
    const out = mat4Identity();
    out[12] = tx;
    out[13] = ty;
    out[14] = tz;
    return out;
  }

  function mat4Scale(sx, sy, sz) {
    const out = mat4Identity();
    out[0] = sx;
    out[5] = sy;
    out[10] = sz;
    return out;
  }

  function mat4RotateX(angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const out = mat4Identity();
    out[5] = c;
    out[6] = s;
    out[9] = -s;
    out[10] = c;
    return out;
  }

  function mat4RotateY(angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const out = mat4Identity();
    out[0] = c;
    out[2] = -s;
    out[8] = s;
    out[10] = c;
    return out;
  }

  function mat4Ortho(left, right, bottom, top, near, far) {
    const out = mat4Identity();
    out[0] = 2 / (right - left);
    out[5] = 2 / (top - bottom);
    out[10] = -2 / (far - near);
    out[12] = -(right + left) / (right - left);
    out[13] = -(top + bottom) / (top - bottom);
    out[14] = -(far + near) / (far - near);
    return out;
  }

  function makeMatrices() {
    const rect = canvas.getBoundingClientRect();
    const aspect = rect.width / Math.max(rect.height, 1);
    const projection = aspect >= 1
      ? mat4Ortho(-aspect, aspect, -1, 1, -3, 3)
      : mat4Ortho(-1, 1, -1 / aspect, 1 / aspect, -3, 3);
    const minDim = Math.max(Math.min(rect.width, rect.height), 1);
    const tx = (state.panX / minDim) * 2;
    const ty = -(state.panY / minDim) * 2;
    let model = mat4Multiply(mat4Translation(tx, ty, 0), mat4Scale(state.zoom, state.zoom, state.zoom));
    model = mat4Multiply(model, mat4RotateX(state.pitch));
    model = mat4Multiply(model, mat4RotateY(state.yaw));
    const mvp = mat4Multiply(projection, model);
    const normalMat = new Float32Array([
      model[0], model[1], model[2],
      model[4], model[5], model[6],
      model[8], model[9], model[10],
    ]);
    return { mvp, normalMat };
  }

  function paintAt(x, y) {
    const scan = activeScan();
    const proj = state.projected || project(scan);
    const r2 = state.brushRadius * state.brushRadius;
    let maxDepth = Infinity;
    for (let i = 0; i < scan.vertex_count; i++) {
      const dx = proj.sx[i] - x;
      const dy = proj.sy[i] - y;
      if (dx * dx + dy * dy <= r2 && proj.depth[i] < maxDepth) maxDepth = proj.depth[i];
    }
    if (!Number.isFinite(maxDepth)) return;
    const shell = 0.045 / Math.max(0.6, state.zoom);
    const changed = [];
    for (let i = 0; i < scan.vertex_count; i++) {
      const dx = proj.sx[i] - x;
      const dy = proj.sy[i] - y;
      if (dx * dx + dy * dy <= r2 && proj.depth[i] <= maxDepth + shell && scan.labels[i] !== state.activeLabel) {
        changed.push([i, scan.labels[i], state.activeLabel]);
        scan.labels[i] = state.activeLabel;
      }
    }
    if (changed.length) {
      state.undo.push({ scanIndex: state.scanIndex, changes });
      state.redo.length = 0;
      refreshScanColors(scan);
      render();
    }
  }

  function undo() {
    const action = state.undo.pop();
    if (!action) return;
    const scan = state.scans[action.scanIndex];
    for (const [idx, oldLabel] of action.changes) {
      scan.labels[idx] = oldLabel;
    }
    refreshScanColors(scan);
    state.redo.push(action);
    render();
  }

  function redo() {
    const action = state.redo.pop();
    if (!action) return;
    const scan = state.scans[action.scanIndex];
    for (const [idx, oldLabel, newLabel] of action.changes) {
      scan.labels[idx] = newLabel;
    }
    refreshScanColors(scan);
    state.undo.push(action);
    render();
  }

  function resetScan() {
    const scan = activeScan();
    scan.labels.set(scan.initialLabels);
    state.undo.length = 0;
    state.redo.length = 0;
    refreshScanColors(scan);
    render();
  }

  function counts(scan) {
    const out = new Array(labelNames.length).fill(0);
    for (const label of scan.labels) out[label]++;
    return out;
  }

  function updateStatus() {
    const scan = activeScan();
    const c = counts(scan);
    status.textContent = [
      `Package: ${pkg.package_id}`,
      `Scan: ${scan.scan_id}`,
      `Vertices: ${scan.vertex_count.toLocaleString()}`,
      `Triangles: ${scan.triangle_count.toLocaleString()}`,
      'Renderer: filled WebGL mesh',
      `Mode: ${state.mode}`,
      `Active: ${labelNames[state.activeLabel]}`,
      '',
      ...labelNames.map((name, idx) => `${name.padEnd(8)} ${String(c[idx]).padStart(7)}`),
    ].join('\n');
    document.querySelectorAll('.labelButton .count').forEach((el, idx) => {
      el.textContent = c[idx].toLocaleString();
    });
  }

  function exportAnnotations() {
    const payload = {
      schema: 'hsr_manual_body_part_segmentation_v1',
      package_id: pkg.package_id,
      exported_at: new Date().toISOString(),
      labels: labelNames,
      label_colors: labelColors,
      scans: state.scans.map(scan => ({
        scan_id: scan.scan_id,
        vertex_count: scan.vertex_count,
        vertex_labels: Array.from(scan.labels),
        label_counts: Object.fromEntries(labelNames.map((name, idx) => [name, counts(scan)[idx]])),
      })),
    };
    const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'manual_body_part_annotations.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function loadAnnotations(file) {
    const reader = new FileReader();
    reader.onload = () => {
      const payload = JSON.parse(reader.result);
      if (!Array.isArray(payload.scans)) throw new Error('Missing scans array in annotation JSON.');
      for (const saved of payload.scans) {
        const scan = state.scans.find(item => item.scan_id === saved.scan_id);
        if (!scan) continue;
        if (!Array.isArray(saved.vertex_labels) || saved.vertex_labels.length !== scan.vertex_count) {
          throw new Error(`Bad label length for ${saved.scan_id}`);
        }
        const savedLabels = Uint8Array.from(saved.vertex_labels);
        for (const label of savedLabels) {
          if (label >= labelNames.length) throw new Error(`Bad label id in ${saved.scan_id}`);
        }
        scan.labels.set(savedLabels);
        refreshScanColors(scan);
      }
      state.undo.length = 0;
      state.redo.length = 0;
      render();
    };
    reader.readAsText(file);
  }

  function initControls() {
    for (let i = 0; i < state.scans.length; i++) {
      const opt = document.createElement('option');
      opt.value = String(i);
      opt.textContent = state.scans[i].scan_id;
      scanSelect.appendChild(opt);
    }
    scanSelect.addEventListener('change', () => {
      state.scanIndex = Number(scanSelect.value);
      state.yaw = 0;
      state.pitch = 0;
      state.projected = null;
      render();
    });

    labelNames.forEach((name, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'labelButton';
      btn.innerHTML = `<span class="swatch" style="background:${labelColors[name]}"></span><span>${idx + 1}. ${name}</span><span class="count"></span>`;
      btn.addEventListener('click', () => {
        state.activeLabel = idx;
        refreshLabelButtons();
      });
      labelButtons.appendChild(btn);
    });
    refreshLabelButtons();

    rotateMode.addEventListener('click', () => setMode('rotate'));
    paintMode.addEventListener('click', () => setMode('paint'));
    document.getElementById('undoBtn').addEventListener('click', undo);
    document.getElementById('redoBtn').addEventListener('click', redo);
    document.getElementById('resetScanBtn').addEventListener('click', resetScan);
    document.getElementById('exportBtn').addEventListener('click', exportAnnotations);
    document.getElementById('importInput').addEventListener('change', event => {
      if (event.target.files[0]) loadAnnotations(event.target.files[0]);
    });
    viewMode.addEventListener('change', () => {
      state.view = viewMode.value;
      for (const scan of state.scans) refreshScanColors(scan);
      render();
    });
    brushRadius.addEventListener('input', () => {
      state.brushRadius = Number(brushRadius.value);
      brushOut.textContent = `${state.brushRadius}px`;
      drawOverlay();
    });
    overlayOpacity.addEventListener('input', () => {
      state.overlayOpacity = Number(overlayOpacity.value);
      opacityOut.textContent = `${Math.round(state.overlayOpacity * 100)}%`;
      for (const scan of state.scans) refreshScanColors(scan);
      render();
    });
    brushOut.textContent = `${state.brushRadius}px`;
    opacityOut.textContent = `${Math.round(state.overlayOpacity * 100)}%`;
    setMode('paint');
  }

  function refreshLabelButtons() {
    document.querySelectorAll('.labelButton').forEach((btn, idx) => {
      btn.classList.toggle('active', idx === state.activeLabel);
    });
    updateStatus();
  }

  function setMode(mode) {
    state.mode = mode;
    rotateMode.classList.toggle('active', mode === 'rotate');
    paintMode.classList.toggle('active', mode === 'paint');
    overlay.classList.toggle('painting', mode === 'paint');
    drawOverlay();
  }

  overlay.addEventListener('pointermove', event => {
    const rect = overlay.getBoundingClientRect();
    state.pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    if (!state.dragging) { drawOverlay(); return; }
    const dx = event.clientX - state.lastX;
    const dy = event.clientY - state.lastY;
    state.lastX = event.clientX;
    state.lastY = event.clientY;
    if (state.dragMode === 'paint') {
      paintAt(state.pointer.x, state.pointer.y);
    } else {
      state.yaw += dx * 0.01;
      state.pitch = Math.max(-1.25, Math.min(1.25, state.pitch + dy * 0.01));
      state.projected = null;
      render();
    }
  });

  overlay.addEventListener('pointerdown', event => {
    overlay.setPointerCapture(event.pointerId);
    state.dragging = true;
    state.lastX = event.clientX;
    state.lastY = event.clientY;
    const rect = overlay.getBoundingClientRect();
    state.pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    const paintStart = state.mode === 'paint' && !event.altKey && event.button === 0 && hasSurfaceAt(state.pointer.x, state.pointer.y);
    state.dragMode = paintStart ? 'paint' : 'rotate';
    if (state.dragMode === 'paint') paintAt(state.pointer.x, state.pointer.y);
  });

  overlay.addEventListener('pointerup', event => {
    state.dragging = false;
    state.dragMode = null;
    try { overlay.releasePointerCapture(event.pointerId); } catch (_) {}
  });

  overlay.addEventListener('pointerleave', () => {
    state.pointer = null;
    state.dragging = false;
    drawOverlay();
  });

  overlay.addEventListener('wheel', event => {
    event.preventDefault();
    const scale = Math.exp(-event.deltaY * 0.001);
    state.zoom = Math.max(0.28, Math.min(4.5, state.zoom * scale));
    state.projected = null;
    render();
  }, { passive: false });

  window.addEventListener('keydown', event => {
    if (event.key >= '1' && event.key <= '8') {
      state.activeLabel = Number(event.key) - 1;
      refreshLabelButtons();
    } else if (event.key.toLowerCase() === 'z' && (event.metaKey || event.ctrlKey)) {
      undo();
    } else if (event.key.toLowerCase() === 'y' && (event.metaKey || event.ctrlKey)) {
      redo();
    } else if (event.key === ' ') {
      setMode(state.mode === 'paint' ? 'rotate' : 'paint');
      event.preventDefault();
    }
  });

  window.addEventListener('resize', resize);
  initControls();
  resize();
})();
"""


def readme_text(package_root: Path) -> str:
    return f"""# Manual HSR Body-Part Segmentation Package

Generated: {datetime.now(timezone.utc).isoformat()}

This package is designed for a headless-server workflow:

1. Copy this folder, or `{package_root.name}.zip`, to a local machine with a browser.
2. Open `app/index.html` directly in Chrome/Edge/Firefox.
3. Paint labels on the scans.
4. Click `Export annotations JSON`.
5. Put the downloaded `manual_body_part_annotations.json` into this package's `results/` folder.
6. Copy the package or JSON file back to the server.
7. Import it with:

```bash
python code/data_generation/hsr/scripts/import_manual_body_part_segmentation.py \\
  data/hsr/manual_body_part_segmentation_package/results/manual_body_part_annotations.json
```

Labels:

{chr(10).join(f"- `{name}`: `{LABEL_COLORS[name]}`" for name in LABEL_NAMES)}

The app is static and uses embedded mesh data in `app/app_data.js`; it does not need a dev server.
The browser viewer renders filled WebGL triangles over the closed textured mesh, not a point cloud.
Raw source assets are also included in `data/raw/`.
"""


def build_package(output_root: Path, segmentation_root: Path, mesh_root: Path, overwrite: bool) -> dict[str, Any]:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "app").mkdir(exist_ok=True)
    (output_root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (output_root / "results").mkdir(exist_ok=True)

    data_dir = segmentation_root / "data"
    scan_npz_paths = sorted(data_dir.glob("*_body_part_segmentation.npz"))
    scans = [scan_payload(path) for path in scan_npz_paths]
    package_id = f"hsr_manual_body_part_segmentation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    payload = {
        "schema": "hsr_manual_body_part_segmentation_package_v1",
        "package_id": package_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels": LABEL_NAMES,
        "label_colors": LABEL_COLORS,
        "scans": scans,
    }

    write_text(output_root / "app" / "index.html", app_index_html())
    write_text(output_root / "app" / "styles.css", app_styles_css())
    write_text(output_root / "app" / "app.js", app_js())
    write_text(
        output_root / "app" / "app_data.js",
        "window.BODY_PART_SEGMENTATION_PACKAGE = "
        + json.dumps(payload, separators=(",", ":"))
        + ";\n",
    )
    write_text(output_root / "README.md", readme_text(output_root))
    write_text(output_root / "results" / "README.md", "Put exported manual_body_part_annotations.json here before importing on the server.\n")

    raw_files: list[str] = []
    for npz_path in scan_npz_paths:
        scan_id = str(np.load(npz_path, allow_pickle=False)["scan_id"])
        for source in [
            npz_path,
            data_dir / f"{scan_id}_body_part_colored_mesh.ply",
            mesh_root / f"{scan_id}_closed_textured_mesh.ply",
        ]:
            if source.exists():
                dst = output_root / "data" / "raw" / source.name
                shutil.copy2(source, dst)
                raw_files.append(root_relative(dst))
    manifest = data_dir / "manifest.json"
    if manifest.exists():
        dst = output_root / "data" / "raw" / "body_part_segmentation_manifest.json"
        shutil.copy2(manifest, dst)
        raw_files.append(root_relative(dst))

    summary = {
        "package_id": package_id,
        "package_root": root_relative(output_root),
        "app": root_relative(output_root / "app" / "index.html"),
        "zip": root_relative(output_root.with_suffix(".zip")),
        "scan_count": len(scans),
        "labels": LABEL_NAMES,
        "raw_files": raw_files,
        "result_path": root_relative(output_root / "results" / "manual_body_part_annotations.json"),
    }
    write_text(output_root / "package_manifest.json", json.dumps(summary, indent=2) + "\n")

    zip_path = output_root.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(output_root.parent))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--segmentation-root", type=Path, default=DEFAULT_SEGMENTATION_ROOT)
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = build_package(args.output_root, args.segmentation_root, args.mesh_root, args.overwrite)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
