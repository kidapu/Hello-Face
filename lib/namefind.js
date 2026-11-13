// NameFind: "what is this cut-out called?" in Japanese. The cut-out (the outside dimmed toward gray) goes through
// MobileCLIP2-S0's image encoder and is matched against ~6400 Japanese words (segtrain/lookup/export_web.py).
// Runs in a worker on the CPU (onnxruntime-web, CDN); the model (~23 MB) + words (~3.5 MB) come from models/namefind/
// next to this script, once, then from the browser's cache. Nothing is uploaded.
//
//   await NameFind.load(status)
//   const r = await NameFind.name(image, mask, w, h)   image: Image / canvas / video frame; mask: Uint8Array(w*h) on a w x h grid
//       -> { ja (？ when unsure), top: [[ja, score], ...], ms }
window.NameFind = (() => {
  const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.js';
  const SRC = (document.currentScript && document.currentScript.src) || location.href;
  const BASE = new URL('.', SRC).href, VQ = new URL(SRC).search, DIR = `${BASE}models/namefind/`;
  const S = 256;
  // below this the best word is as often wrong as right (COCO regions): show ？ instead
  const MIN = 0.13;
  let worker = null, loading = null, seq = 0, ready = false;
  const wait = new Map();
  const call = (type, msg = {}, transfer = []) => new Promise((res, rej) => {
    const id = ++seq; wait.set(id, { res, rej }); worker.postMessage({ id, type, ...msg }, transfer);
  });

  function load(status = () => {}) {
    if (loading) return loading;
    loading = (async () => {
      status('名前: モデルを読み込み中…');
      worker = new Worker(`${BASE}namefind_worker.js${VQ}`);
      worker.onmessage = ({ data: m }) => { const w = wait.get(m.id); if (!w) return; wait.delete(m.id); m.error ? w.rej(new Error(m.error)) : w.res(m); };
      worker.onerror = (e) => { const err = new Error(e.message || '名前の処理が止まりました'); for (const w of wait.values()) w.rej(err); wait.clear(); worker = null; loading = null; ready = false; };
      await call('load', { ort: ORT, dir: DIR });
      ready = true;
    })();
    loading.catch(() => { loading = null; });
    return loading;
  }

  let cv = null, cx = null;
  async function name(image, mask, w, h) {
    await load();
    const iw = image.videoWidth || image.naturalWidth || image.width, ih = image.videoHeight || image.naturalHeight || image.height;
    // the piece's box on the grid, plus 8 %
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    if (x1 < 0) return null;
    const mg = 0.08 * Math.max(x1 - x0 + 1, y1 - y0 + 1);
    let bx0 = Math.max(0, x0 - mg), by0 = Math.max(0, y0 - mg), bx1 = Math.min(w, x1 + 1 + mg), by1 = Math.min(h, y1 + 1 + mg);
    // the model's own preprocessing: short side to 256, the middle square
    const bw = bx1 - bx0, bh = by1 - by0, side = Math.min(bw, bh);
    bx0 += (bw - side) / 2; by0 += (bh - side) / 2;
    if (!cv) { cv = document.createElement('canvas'); cv.width = cv.height = S; cx = cv.getContext('2d', { willReadFrequently: true }); }
    const kx = iw / w, ky = ih / h;
    cx.drawImage(image, bx0 * kx, by0 * ky, side * kx, side * ky, 0, 0, S, S);
    const d = cx.getImageData(0, 0, S, S).data;
    // the mask on the 256 grid, softened (~3 px), then the outside dimmed toward gray (as in test_lookup.py)
    let m = new Float32Array(S * S);
    for (let y = 0; y < S; y++) {
      const gy = Math.min(h - 1, Math.floor(by0 + (y + 0.5) / S * side));
      for (let x = 0; x < S; x++) m[y * S + x] = mask[gy * w + Math.min(w - 1, Math.floor(bx0 + (x + 0.5) / S * side))];
    }
    for (let p = 0; p < 2; p++) m = blur(m, S, 3);
    const px = new Float32Array(3 * S * S), n = S * S;
    for (let i = 0; i < n; i++) {
      const a = 0.35 + 0.65 * m[i], g = 127 * (0.65 - 0.65 * m[i]);
      px[i] = (d[i * 4] * a + g) / 255; px[n + i] = (d[i * 4 + 1] * a + g) / 255; px[2 * n + i] = (d[i * 4 + 2] * a + g) / 255;
    }
    const r = await call('name', { pixels: px }, [px.buffer]);
    return { ja: r.top[0][1] >= MIN ? r.top[0][0] : '？', top: r.top, ms: r.ms };
  }
  function blur(a, s, r) {   // box blur, rows then columns
    const b = new Float32Array(a.length), c = new Float32Array(a.length), k = 2 * r + 1;
    for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) { let t = 0; for (let d = -r; d <= r; d++) t += a[y * s + Math.min(s - 1, Math.max(0, x + d))]; b[y * s + x] = t / k; }
    for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) { let t = 0; for (let d = -r; d <= r; d++) t += b[Math.min(s - 1, Math.max(0, y + d)) * s + x]; c[y * s + x] = t / k; }
    return c;
  }
  return { load, name, get ready() { return ready; } };
})();
