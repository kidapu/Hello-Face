// CloudFind: find the round bulges of white clouds in an image / video frame.
// Shared by cloud_doodle.html (still photos) and cloud_cam.html (camera).
//
//   const det = CloudFind.analyse(source, srcW, srcH, { detW, blur, thrBias })
//       blurred whiteness -> Otsu threshold -> cloud mask -> distance to the sky (px)
//   const faces = CloudFind.find(det, { minR, maxR, maxFaces, sep, faceUp, faceScale })
//       local maxima of the distance = centres of the largest circles that fit in each bulge;
//       each face: { cx, cy, cr (circle), r (face radius), fx, fy (face centre),
//                    ears: [{x, y, nx, ny, onEdge}] | null, paws: [{x, y}] | null }   (detector px)
window.CloudFind = (() => {
  let work = null;   // reused canvas for reading pixels

  function otsu(buf) {
    const h = new Float64Array(64);
    for (const v of buf) h[Math.min(63, Math.max(0, Math.floor(v * 64)))]++;
    let tot = buf.length, sum = 0, sumB = 0, wB = 0, best = 0, thr = 0.5;
    for (let i = 0; i < 64; i++) sum += i * h[i];
    for (let i = 0; i < 64; i++) {
      wB += h[i]; if (!wB) continue;
      const wF = tot - wB; if (!wF) break;
      sumB += i * h[i];
      const mB = sumB / wB, mF = (sum - sumB) / wF, b = wB * wF * (mB - mF) ** 2;
      if (b > best) { best = b; thr = (i + 1) / 64; }
    }
    return thr;
  }
  // separable box blur, run twice (close to a gaussian); canvas filters are not on every browser
  function boxBlur(a, w, h, r) {
    if (r < 1) return a;
    const t = new Float32Array(a.length);
    for (let pass = 0; pass < 2; pass++) {
      for (let y = 0; y < h; y++) {
        let s = 0; const o = y * w;
        for (let x = -r; x <= r; x++) s += a[o + Math.min(w - 1, Math.max(0, x))];
        for (let x = 0; x < w; x++) {
          t[o + x] = s / (2 * r + 1);
          s += a[o + Math.min(w - 1, x + r + 1)] - a[o + Math.max(0, x - r)];
        }
      }
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let y = -r; y <= r; y++) s += t[Math.min(h - 1, Math.max(0, y)) * w + x];
        for (let y = 0; y < h; y++) {
          a[y * w + x] = s / (2 * r + 1);
          s += t[Math.min(h - 1, y + r + 1) * w + x] - t[Math.max(0, y - r) * w + x];
        }
      }
    }
    return a;
  }

  function analyse(src, sw, sh, o) {
    const w = o.detW, h = Math.max(8, Math.round(w * sh / sw));
    if (!work) work = document.createElement('canvas');
    if (work.width !== w || work.height !== h) { work.width = w; work.height = h; }
    const x2 = work.getContext('2d', { willReadFrequently: true });
    x2.drawImage(src, 0, 0, w, h);
    const d = x2.getImageData(0, 0, w, h).data;
    const white = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      const r = d[i * 4] / 255, g = d[i * 4 + 1] / 255, b = d[i * 4 + 2] / 255;
      const mn = Math.min(r, g, b), mx = Math.max(r, g, b);
      white[i] = Math.min(1, Math.max(0, mn - 0.8 * (mx - mn)));   // white & grey clouds high, blue sky / trees low
    }
    boxBlur(white, w, h, Math.round(o.blur));
    const thr = Math.min(0.95, Math.max(0.05, otsu(white) + o.thrBias));
    const mask = new Uint8Array(w * h);
    let cover = 0;
    for (let i = 0; i < w * h; i++) { mask[i] = white[i] > thr ? 1 : 0; cover += mask[i]; }
    // distance (px) to the nearest sky pixel; outside of the image counts as sky (2-pass chamfer)
    const D = new Float32Array(w * h), q = Math.SQRT2;
    for (let i = 0; i < w * h; i++) D[i] = mask[i] ? 1e9 : 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (!D[i]) continue;
      let v = D[i];
      v = Math.min(v, x > 0 ? D[i - 1] + 1 : 1);
      if (y > 0) v = Math.min(v, D[i - w] + 1, x > 0 ? D[i - w - 1] + q : 1, x < w - 1 ? D[i - w + 1] + q : 1); else v = Math.min(v, 1);
      D[i] = v;
    }
    for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x; if (!D[i]) continue;
      let v = D[i];
      v = Math.min(v, x < w - 1 ? D[i + 1] + 1 : 1);
      if (y < h - 1) v = Math.min(v, D[i + w] + 1, x < w - 1 ? D[i + w + 1] + q : 1, x > 0 ? D[i + w - 1] + q : 1); else v = Math.min(v, 1);
      D[i] = v;
    }
    return { w, h, white, mask, D, thr, cover: cover / (w * h) };
  }

  function find(det, o) {
    const { w, h, D } = det, minR = o.minR * w, maxR = o.maxR * w;
    const inMask = (x, y) => { const X = Math.round(x), Y = Math.round(y);
      return X >= 0 && Y >= 0 && X < w && Y < h && det.mask[Y * w + X] === 1; };
    const distAt = (x, y) => { const X = Math.round(x), Y = Math.round(y);
      return X >= 0 && Y >= 0 && X < w && Y < h ? D[Y * w + X] : 0; };
    const march = (x, y, dx, dy, maxT) => {
      for (let t = 0; t < maxT; t += 0.5) {
        const X = x + dx * t, Y = y + dy * t;
        if (X < 0 || Y < 0 || X > w - 1 || Y > h - 1) return { t, edge: true };
        if (!inMask(X, Y)) return { t, edge: false };
      }
      return { t: maxT, edge: false, far: true };
    };
    const outward = (x, y) => {
      const s = 2, f = (X, Y) => { X = Math.max(0, Math.min(w - 1, Math.round(X))); Y = Math.max(0, Math.min(h - 1, Math.round(Y)));
        return det.white[Y * w + X]; };
      const gx = f(x + s, y) - f(x - s, y), gy = f(x, y + s) - f(x, y - s), l = Math.hypot(gx, gy);
      return l < 1e-5 ? null : [-gx / l, -gy / l];
    };
    // candidates: local maxima of the distance map (5x5)
    const cand = [];
    for (let y = 2; y < h - 2; y++) for (let x = 2; x < w - 2; x++) {
      const v = D[y * w + x]; if (v < minR) continue;
      let ok = true;
      for (let j = -2; j <= 2 && ok; j++) for (let i = -2; i <= 2; i++) if (D[(y + j) * w + x + i] > v) { ok = false; break; }
      if (ok) cand.push({ x, y, r: v });
    }
    cand.sort((a, b) => b.r - a.r);
    const out = [];
    for (const c of cand) {
      if (out.length >= o.maxFaces) break;
      if (out.some((f) => Math.hypot(f.cx - c.x, f.cy - c.y) < o.sep * (f.cr + c.r))) continue;
      const r = Math.min(c.r, maxR);
      // slide the face up toward the top of the bulge while there is still room for it
      let fx = c.x, fy = c.y;
      const step = Math.max(0.5, r * 0.05);
      for (let k = 0; k < 60 && o.faceUp > 0; k++) {
        const ny = fy - step, top = march(fx, ny, 0, -1, r * 4).t;
        if (distAt(fx, ny) < r * (1 - 0.35 * o.faceUp) || top < r * (1.25 - 0.2 * o.faceUp)) break;
        fy = ny;
      }
      const fr = r * o.faceScale;
      // ears: two rays up and outward; on the outline when it is near, else on the head line
      let ears = [];
      for (const s of [-1, 1]) {
        const a = 30 * Math.PI / 180, dx = s * Math.sin(a), dy = -Math.cos(a);
        const m = march(fx, fy, dx, dy, fr * 1.25);
        if (m.edge && m.t < fr * 0.7) { ears = null; break; }
        const onEdge = !m.far && !m.edge, tt = Math.min(m.t, fr * 1.0);
        const bx = fx + dx * tt, by = fy + dy * tt, n = (onEdge && outward(bx, by)) || [dx, dy];
        let nx = 0.5 * n[0] + 0.5 * dx, ny = 0.5 * n[1] + 0.5 * dy; const l = Math.hypot(nx, ny); nx /= l; ny /= l;
        ears.push({ x: bx, y: by, nx, ny, onEdge });
      }
      // paws: two rays down, just inside the bottom outline
      let paws = [];
      for (const s of [-1, 1]) {
        const a = 28 * Math.PI / 180, dx = s * Math.sin(a), dy = Math.cos(a);
        const m = march(fx, fy, dx, dy, fr * 1.5);
        if (m.edge || m.t < fr * 0.8) { paws = null; break; }
        paws.push({ x: fx + dx * m.t, y: fy + dy * m.t });
      }
      out.push({ cx: c.x, cy: c.y, cr: c.r, r, fx, fy, ears, paws });
    }
    return out;
  }
  return { analyse, find, otsu, boxBlur };
})();
