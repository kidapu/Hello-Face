// ColorFind: a mask from "things that look like the colour you tapped", split along edges.
// For wood blocks, leaves, rocks ... anything that is not a white cloud.
// The result has the same shape as CloudFind.analyse(), so CloudFind.find() places the faces.
//
//   const det = ColorFind.analyse(source, srcW, srcH, { detW, blur, refs, tol, lumaW, edgeThr, edgeW, thrBias })
//       refs: [[L, a, b], ...]  colours to look for (from ColorFind.pick)
//       similarity to the nearest ref (CIE Lab, lightness weighted by lumaW so shading still counts)
//       -> cut along strong edges (seams between blocks, the fold between a front and a side face)
//       -> mask -> distance to the outside (px)
//   const lab = ColorFind.pick(det, x, y, r)   mean Lab around a detector pixel
window.ColorFind = (() => {
  let work = null;
  const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const f3 = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  function toLab(r, g, b) {
    r = lin(r); g = lin(g); b = lin(b);
    const x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.9505, y = 0.2126 * r + 0.7152 * g + 0.0722 * b,
          z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.089;
    const fx = f3(x), fy = f3(y), fz = f3(z);
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
  }

  function analyse(src, sw, sh, o) {
    const w = o.detW, h = Math.max(8, Math.round(w * sh / sw)), n = w * h;
    if (!work) work = document.createElement('canvas');
    if (work.width !== w || work.height !== h) { work.width = w; work.height = h; }
    const x2 = work.getContext('2d', { willReadFrequently: true });
    x2.drawImage(src, 0, 0, w, h);
    const d = x2.getImageData(0, 0, w, h).data;
    const L = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const lab = toLab(d[i * 4] / 255, d[i * 4 + 1] / 255, d[i * 4 + 2] / 255);
      L[i] = lab[0]; A[i] = lab[1]; B[i] = lab[2];
    }
    // similarity to the nearest reference colour
    const refs = o.refs && o.refs.length ? o.refs : [];
    const sim = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let best = 0;
      for (const [rl, ra, rb] of refs) {
        const dE = Math.hypot((L[i] - rl) * o.lumaW, A[i] - ra, B[i] - rb);
        best = Math.max(best, 1 - dE / o.tol);
      }
      sim[i] = best;
    }
    CloudFind.boxBlur(sim, w, h, Math.round(o.blur));
    // edges: lightness gradient (Sobel) on a lightly blurred copy
    const Ls = CloudFind.boxBlur(Float32Array.from(L), w, h, 1), edge = new Float32Array(n);
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = (Ls[i - w + 1] + 2 * Ls[i + 1] + Ls[i + w + 1]) - (Ls[i - w - 1] + 2 * Ls[i - 1] + Ls[i + w - 1]);
      const gy = (Ls[i + w - 1] + 2 * Ls[i + w] + Ls[i + w + 1]) - (Ls[i - w - 1] + 2 * Ls[i - w] + Ls[i - w + 1]);
      edge[i] = Math.hypot(gx, gy) / 8;   // lightness units per px
    }
    const thr = 0.5 + o.thrBias, mask = new Uint8Array(n), ew = Math.max(0, Math.round(o.edgeW));
    let cover = 0;
    for (let i = 0; i < n; i++) mask[i] = sim[i] > thr ? 1 : 0;
    // cut a line of width edgeW wherever the edge is strong
    if (o.edgeThr > 0) {
      const cut = new Uint8Array(n);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        if (edge[y * w + x] < o.edgeThr) continue;
        for (let j = -ew; j <= ew; j++) for (let k = -ew; k <= ew; k++) {
          const X = x + k, Y = y + j;
          if (X >= 0 && Y >= 0 && X < w && Y < h) cut[Y * w + X] = 1;
        }
      }
      for (let i = 0; i < n; i++) if (cut[i]) mask[i] = 0;
    }
    for (let i = 0; i < n; i++) cover += mask[i];
    const D = distance(mask, w, h);
    return { w, h, white: sim, mask, D, edge, L, A, B, thr, cover: cover / n };
  }

  // distance (px) to the nearest pixel outside of the mask (2-pass chamfer, the image border counts as outside)
  function distance(mask, w, h) {
    const n = w * h;
    const D = new Float32Array(n), q = Math.SQRT2;
    for (let i = 0; i < n; i++) D[i] = mask[i] ? 1e9 : 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (!D[i]) continue;
      let v = Math.min(D[i], x > 0 ? D[i - 1] + 1 : 1);
      if (y > 0) v = Math.min(v, D[i - w] + 1, x > 0 ? D[i - w - 1] + q : 1, x < w - 1 ? D[i - w + 1] + q : 1); else v = Math.min(v, 1);
      D[i] = v;
    }
    for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x; if (!D[i]) continue;
      let v = Math.min(D[i], x < w - 1 ? D[i + 1] + 1 : 1);
      if (y < h - 1) v = Math.min(v, D[i + w] + 1, x < w - 1 ? D[i + w + 1] + q : 1, x > 0 ? D[i + w - 1] + q : 1); else v = Math.min(v, 1);
      D[i] = v;
    }
    return D;
  }

  function pick(det, x, y, r = 2) {
    let l = 0, a = 0, b = 0, c = 0;
    for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) {
      const X = Math.round(x) + i, Y = Math.round(y) + j;
      if (X < 0 || Y < 0 || X >= det.w || Y >= det.h) continue;
      const k = Y * det.w + X; l += det.L[k]; a += det.A[k]; b += det.B[k]; c++;
    }
    return c ? [l / c, a / c, b / c] : null;
  }
  return { analyse, pick, toLab, distance };
})();
