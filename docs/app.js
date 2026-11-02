"use strict";
const TILE = 256;
const $ = id => document.getElementById(id);
let session = null, results = null;
let imgA = null, imgB = null;       // ImageData
let prob = null;                    // Float32Array, live TinyCD output, w*h
let saved = {};                     // sample-only maps in [0,1]: label / bit / changeformer
let dims = [0, 0];
let busy = false;

const setStatus = t => { $("status").textContent = t; };

async function init() {
  results = await (await fetch("data/results.json")).json();
  const tb = document.querySelector("#tbl tbody");
  for (const m of results.models) {
    const tr = document.createElement("tr");
    if (m.id === "tinycd") tr.className = "me";
    tr.innerHTML = `<td>${m.model}</td><td>${m.f1.toFixed(2)}</td><td>${m.paper_f1.toFixed(2)}</td><td>${m.precision.toFixed(2)}</td>` +
      `<td>${m.recall.toFixed(2)}</td><td>${m.params.toLocaleString()}</td><td>${m.gflops.toFixed(2)}</td>`;
    tb.appendChild(tr);
  }
  const sel = $("sample");
  for (const s of results.samples) sel.add(new Option(`${s.title} (${s.dir})`, s.dir));
  sel.onchange = () => loadSample(sel.value);
  $("fileA").onchange = $("fileB").onchange = loadUploads;
  $("run").onclick = run;
  $("th").oninput = () => { $("thOut").value = (+$("th").value).toFixed(2); render(); };
  $("view").onchange = render;
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create("tinycd.onnx", { executionProviders: ["wasm"] });
  await loadSample(results.samples[0].dir);
}

function loadImg(src) {
  return new Promise((res, rej) => {
    const i = new Image();
    i.onload = () => res(i);
    i.onerror = () => rej(new Error("Could not read that image file."));
    i.src = src;
  });
}
function toImageData(img) {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0);
  return g.getImageData(0, 0, c.width, c.height);
}
function toProb(img) {      // grayscale PNG -> [0,1]
  const d = toImageData(img).data, out = new Float32Array(d.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = d[i * 4] / 255;
  return out;
}

async function loadSample(dir) {
  setStatus("Loading sample...");
  const base = `samples/${dir}/`;
  const [a, b, l, pb, pc] = await Promise.all(["A", "B", "label", "prob_bit", "prob_changeformer"].map(n => loadImg(`${base}${n}.png`)));
  setImages(toImageData(a), toImageData(b));
  saved = { label: toProb(l), bit: toProb(pb), changeformer: toProb(pc) };
  await run();
}

async function loadUploads() {
  const fa = $("fileA").files[0], fb = $("fileB").files[0];
  if (!fa || !fb) return;
  try {
    const [a, b] = await Promise.all([fa, fb].map(f => loadImg(URL.createObjectURL(f))));
    if (a.naturalWidth !== b.naturalWidth || a.naturalHeight !== b.naturalHeight)
      throw new Error(`The two images must have the same size (got ${a.naturalWidth}x${a.naturalHeight} and ${b.naturalWidth}x${b.naturalHeight}).`);
    if (a.naturalWidth * a.naturalHeight > 2048 * 2048) throw new Error("Image too large for the browser demo (max 2048x2048).");
    setImages(toImageData(a), toImageData(b));
    saved = {};
    $("view").value = "tinycd";
    await run();
  } catch (e) { setStatus(e.message); }
}

function setImages(a, b) {
  imgA = a; imgB = b; dims = [a.width, a.height]; prob = null;
  for (const [id, im] of [["cA", a], ["cB", b]]) {
    const c = $(id); c.width = im.width; c.height = im.height;
    c.getContext("2d").putImageData(im, 0, 0);
  }
  render();
}

const reflect = (i, n) => {
  if (n === 1) return 0;
  const p = 2 * (n - 1);
  i = ((i % p) + p) % p;
  return i < n ? i : p - i;
};

// CHW float tensor in [0,1], reflect-padded at the borders (the model normalises internally).
function tileTensor(im, x0, y0) {
  const t = new Float32Array(3 * TILE * TILE), d = im.data;
  for (let y = 0; y < TILE; y++) {
    const sy = reflect(y0 + y, im.height);
    for (let x = 0; x < TILE; x++) {
      const si = (sy * im.width + reflect(x0 + x, im.width)) * 4, o = y * TILE + x;
      for (let c = 0; c < 3; c++) t[c * TILE * TILE + o] = d[si + c] / 255;
    }
  }
  return new ort.Tensor("float32", t, [1, 3, TILE, TILE]);
}

async function run() {
  if (!imgA || !session || busy) return;
  busy = true; $("run").disabled = true;
  const [w, h] = dims, out = new Float32Array(w * h), t0 = performance.now();
  const nx = Math.ceil(w / TILE), ny = Math.ceil(h / TILE);
  try {
    for (let ty = 0; ty < ny; ty++) for (let tx = 0; tx < nx; tx++) {
      setStatus(`Running TinyCD... tile ${ty * nx + tx + 1} of ${nx * ny}`);
      await new Promise(r => setTimeout(r));     // let the page repaint
      const r = await session.run({ before: tileTensor(imgA, tx * TILE, ty * TILE), after: tileTensor(imgB, tx * TILE, ty * TILE) });
      const p = r.prob.data;
      for (let y = 0; y < TILE && ty * TILE + y < h; y++) for (let x = 0; x < TILE && tx * TILE + x < w; x++)
        out[(ty * TILE + y) * w + tx * TILE + x] = p[y * TILE + x];
    }
    prob = out;
    setStatus(`Done: ${w}x${h} px, ${nx * ny} tile(s), ${((performance.now() - t0) / 1000).toFixed(2)} s in your browser.`);
  } catch (e) {
    setStatus("Inference failed: " + e.message);
  }
  busy = false; $("run").disabled = false;
  render();
}

function render() {
  if (!imgA) return;
  $("run").disabled = !session || busy;
  const [w, h] = dims, th = +$("th").value, view = $("view").value;
  const p = view === "tinycd" ? prob : saved[view];
  const c = $("cP"); c.width = w; c.height = h;
  const g = c.getContext("2d"), im = new ImageData(new Uint8ClampedArray(imgB.data), w, h);
  if (!p) {
    g.putImageData(im, 0, 0);
    $("info").textContent = view === "tinycd" ? "" : "Not available for uploaded images.";
    return;
  }
  let n = 0;
  for (let i = 0; i < p.length; i++) if (p[i] >= th) {
    n++;
    const o = i * 4;
    im.data[o] = 0.45 * im.data[o] + 0.55 * 255;
    im.data[o + 1] = 0.45 * im.data[o + 1] + 0.55 * 40;
    im.data[o + 2] = 0.45 * im.data[o + 2] + 0.55 * 40;
  }
  g.putImageData(im, 0, 0);
  $("info").textContent = `${(100 * n / p.length).toFixed(1)}% of pixels marked as changed`;
}

init().catch(e => setStatus("Could not start: " + e.message + " (open this page over http, not as a file)"));
