// ShapeNet: which animal a cloud's silhouette looks like, by a small CNN (segtrain/shapes/train_shape.py) run in
// plain JS (about 3 M multiply-adds per cloud, well under a millisecond or two; no WebGPU needed).
//   await ShapeNet.load(url)                      -> ShapeNet.ready
//   ShapeNet.classify(label, w, h, lab)           -> { kind: prob, ... } for the component lab of the label grid
//   ShapeNet.input(label, w, h, lab)              -> the 32 x 32 coverage the CNN sees (for the debug page)
window.ShapeNet = (() => {
  let M = null;
  // the weights sit next to this script (shapenet.json), wherever the page is
  const DEF = typeof document !== 'undefined' && document.currentScript ? new URL('shapenet.json' + new URL(document.currentScript.src).search, document.currentScript.src).href : 'shapenet.json';
  const f32 = (s) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Float32Array(u.buffer); };
  async function load(url = DEF) {
    if (M) return M;
    const j = await (await fetch(url)).json();
    M = { kinds: j.kinds, N: j.in, L: j.long, acc: j.val_acc, conv: j.conv.map((c) => ({ w: f32(c.w), b: f32(c.b), co: c.shape[0], ci: c.shape[1] })),
      fc: j.fc.map((c) => ({ w: f32(c.w), b: f32(c.b), o: c.shape[0], i: c.shape[1] })) };
    api.ready = true; api.kinds = M.kinds;
    return M;
  }
  // the component's box, fitted so its long side is L cells of the N x N input, centred; each input cell = the
  // part of it covered by the component (4 x 4 samples)
  function input(label, w, h, lab) {
    const N = M ? M.N : 32, L = M ? M.L : 28;
    let x0 = w, x1 = -1, y0 = h, y1 = -1;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (label[y * w + x] === lab) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    const X = new Float32Array(N * N); if (x1 < 0) return X;
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1, k = L / Math.max(bw, bh), ow = bw * k, oh = bh * k;
    // same placement as the training's fit(): the scaled box rounded, at the integer offset (N - size) // 2
    const rw = Math.max(1, Math.round(ow)), rh = Math.max(1, Math.round(oh)), ox = (N - rw) >> 1, oy = (N - rh) >> 1;
    for (let v = 0; v < rh; v++) for (let u = 0; u < rw; u++) {
      let c = 0;
      for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) {
        const sx = x0 + Math.floor((u + (i + 0.5) / 4) / rw * bw), sy = y0 + Math.floor((v + (j + 0.5) / 4) / rh * bh);
        if (label[sy * w + sx] === lab) c++;
      }
      X[(oy + v) * N + ox + u] = c / 16;
    }
    return X;
  }
  function conv(x, S, c) {   // 3x3, padding 1, then ReLU
    const { w, b, co, ci } = c, y = new Float32Array(co * S * S);
    for (let o = 0; o < co; o++) {
      const yo = o * S * S; y.fill(b[o], yo, yo + S * S);
      for (let i = 0; i < ci; i++) {
        const xi = i * S * S, wo = (o * ci + i) * 9;
        for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
          const wv = w[wo + ky * 3 + kx]; if (!wv) continue;
          const dy = ky - 1, dx = kx - 1;
          for (let yy = Math.max(0, -dy); yy < Math.min(S, S - dy); yy++) {
            const r = yo + yy * S, s = xi + (yy + dy) * S + dx;
            for (let xx = Math.max(0, -dx); xx < Math.min(S, S - dx); xx++) y[r + xx] += wv * x[s + xx];
          }
        }
      }
    }
    for (let i = 0; i < y.length; i++) if (y[i] < 0) y[i] = 0;
    return y;
  }
  function pool(x, C, S) {
    const T = S >> 1, y = new Float32Array(C * T * T);
    for (let c = 0; c < C; c++) for (let v = 0; v < T; v++) for (let u = 0; u < T; u++) {
      const a = c * S * S + 2 * v * S + 2 * u; y[c * T * T + v * T + u] = Math.max(x[a], x[a + 1], x[a + S], x[a + S + 1]);
    }
    return y;
  }
  function fc(x, l, relu) {
    const y = new Float32Array(l.o);
    for (let o = 0; o < l.o; o++) { let s = l.b[o]; const r = o * l.i; for (let i = 0; i < l.i; i++) s += l.w[r + i] * x[i]; y[o] = relu && s < 0 ? 0 : s; }
    return y;
  }
  function run(X) {
    let x = X, S = M.N;
    M.conv.forEach((c, i) => { x = conv(x, S, c); if (i < 3) { x = pool(x, c.co, S); S >>= 1; } });
    const z = fc(fc(x, M.fc[0], true), M.fc[1], false), mx = Math.max(...z);
    let t = 0; const e = Array.from(z, (v) => { const q = Math.exp(v - mx); t += q; return q; });
    const p = {}; M.kinds.forEach((k, i) => (p[k] = e[i] / t));
    return p;
  }
  function classify(label, w, h, lab) { return M ? run(input(label, w, h, lab)) : null; }
  const api = { load, classify, input, run, ready: false, kinds: null };
  return api;
})();
