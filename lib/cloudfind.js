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
  // max filter (square, radius r), separable
  function dilate(a, w, h, r, ry = r) {
    const t = new Float32Array(a.length), o = new Float32Array(a.length);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let m = -1e9; const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
      for (let k = x0; k <= x1; k++) m = Math.max(m, a[y * w + k]);
      t[y * w + x] = m;
    }
    for (let x = 0; x < w; x++) for (let y = 0; y < h; y++) {
      let m = -1e9; const y0 = Math.max(0, y - ry), y1 = Math.min(h - 1, y + ry);
      for (let k = y0; k <= y1; k++) m = Math.max(m, t[k * w + x]);
      o[y * w + x] = m;
    }
    return o;
  }

  function analyse(src, sw, sh, o) {
    const w = o.detW, h = Math.max(8, Math.round(w * sh / sw));
    if (!work) work = document.createElement('canvas');
    if (work.width !== w || work.height !== h) { work.width = w; work.height = h; }
    const x2 = work.getContext('2d', { willReadFrequently: true });
    x2.drawImage(src, 0, 0, w, h);
    const d = x2.getImageData(0, 0, w, h).data;
    const n = w * h, white = new Float32Array(n), lum = new Float32Array(n), sky = new Uint8Array(n), pale = new Uint8Array(n);
    let nSky = 0;
    for (let i = 0; i < n; i++) {
      const r = d[i * 4] / 255, g = d[i * 4 + 1] / 255, b = d[i * 4 + 2] / 255;
      const mn = Math.min(r, g, b), mx = Math.max(r, g, b);
      white[i] = Math.min(1, Math.max(0, mn - 0.8 * (mx - mn)));   // white & grey clouds high, blue sky / trees low
      lum[i] = 0.3 * r + 0.59 * g + 0.11 * b;
      if (b - r > 0.12 && b > 0.4 && b >= g) { sky[i] = 1; nSky++; }   // clear blue sky
      else if (b - r > 0.01 && lum[i] > 0.55) pale[i] = 1;             // pale / hazy sky near the horizon
    }
    boxBlur(white, w, h, Math.round(o.blur));
    // threshold: Otsu, but never below "a bit whiter than the typical blue sky" -- with buildings, trees and
    // road in the photo Otsu splits dark / bright instead of sky / cloud
    // with enough blue sky: a cloud is whiter than the sky around it (the sky itself gets paler toward the
    // horizon, so one global threshold either misses thin clouds or swallows the hazy sky);
    // local sky level = blur(white * sky) / blur(sky) with a big radius
    let thr = otsu(white), thrMode = 'otsu';
    const mask = new Uint8Array(n);
    if ((o.skyRel ?? true) && nSky > n * 0.03) {
      thrMode = 'sky';
      const sw = []; for (let i = 0; i < n; i++) if (sky[i]) sw.push(white[i]);
      sw.sort((a, b) => a - b);
      const sky50 = sw[sw.length >> 1], R = Math.max(3, Math.round(w / 10));
      const num = new Float32Array(n), den = new Float32Array(n);
      for (let i = 0; i < n; i++) if (sky[i]) { num[i] = white[i]; den[i] = 1; }
      boxBlur(num, w, h, R); boxBlur(den, w, h, R);
      const delta = (o.skyDelta ?? 0.14) + o.thrBias;
      // the pale haze over the horizon is not blue, so it needs its own level: per band of columns and per row,
      // a low percentile of the bright pixels (the haze itself, or the sky between clouds), carried down from
      // the rows above when a row has too few bright pixels; a cloud must be whiter than both levels
      const NB = 4, bw = Math.ceil(w / NB), lv = new Float32Array(NB * h), buf = [];
      for (let b = 0; b < NB; b++) {
        let prev = sky50;
        for (let y = 0; y < h; y++) {
          buf.length = 0;
          for (let x = b * bw; x < Math.min(w, (b + 1) * bw); x++) if (lum[y * w + x] > 0.35) buf.push(white[y * w + x]);
          if (buf.length > bw * 0.2) { buf.sort((a, c) => a - c); prev = Math.max(sky50, buf[Math.floor(buf.length * (o.hazePct ?? 0.3))]); }
          lv[b * h + y] = prev;
        }
      }
      const rr = Math.max(1, Math.round(h / 30)), lvS = new Float32Array(NB * h);
      for (let b = 0; b < NB; b++) for (let y = 0; y < h; y++) {
        let a = 0, c = 0; for (let j = -rr; j <= rr; j++) { const Y = y + j; if (Y >= 0 && Y < h) { a += lv[b * h + Y]; c++; } }
        lvS[b * h + y] = a / c;
      }
      const haze = (x, y) => {   // bilinear across the bands
        const f = Math.min(NB - 1, Math.max(0, x / bw - 0.5)), b0 = Math.floor(f), b1 = Math.min(NB - 1, b0 + 1), t = f - b0;
        return lvS[b0 * h + y] * (1 - t) + lvS[b1 * h + y] * t;
      };
      const ref = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const x = i % w, y = (i - x) / w, k = Math.min(1, den[i] / 0.05), local = den[i] > 1e-4 ? num[i] / den[i] : sky50;
        ref[i] = Math.max(k * local + (1 - k) * sky50, haze(x, y));
        mask[i] = white[i] > ref[i] + delta ? 1 : 0;
      }
      // big bright clouds: around them the "sky" is partly pale cloud, so the local level is too high;
      // when Otsu does split sky / cloud (its threshold is above the sky) its mask is added where it is
      // still clearly above the local level
      const tO = thr + o.thrBias;
      if (tO > sky50 + 0.08) for (let i = 0; i < n; i++) if (white[i] > tO && white[i] > ref[i] + delta * 0.4) mask[i] = 1;
      thr = sky50 + delta;
      // sunset / dusk: pink and grey clouds are not whiter than the sky (grey ones are darker), but they are
      // less blue than the sky next to them; the sky reference is the locally bluest colour (a max filter),
      // and its lightness is taken from the pixels that are that blue. The yellow glow over the horizon is
      // sky, not cloud. Busy pixels (windows, wires, edges of buildings) are left out.
      // auto: only for a dusk sky (greyish blue, or many warm-lit pixels); on a deep blue day sky the soft
      // edges of white clouds are less blue too and would merge neighbouring clouds
      // warm pixels are only counted above the skyline (the lowest row that is still 10% blue sky): sunlit
      // sand / ground below it is warm too and is not a sunset
      let skyline = 0;
      for (let y = 0; y < h; y++) { let c = 0; for (let x = 0; x < w; x++) c += sky[y * w + x]; if (c > w * 0.1) skyline = y; }
      let warm = 0; const sb = [];
      for (let i = 0; i < n; i++) {
        const r = d[i * 4] / 255, g = d[i * 4 + 1] / 255, b = d[i * 4 + 2] / 255;
        if (sky[i]) sb.push((b - r) / (Math.max(r, g, b) + 0.05));
        if (i < (skyline + 1) * w && r > b + 0.03 && r > g && lum[i] > 0.45) warm++;
      }
      sb.sort((a, c) => a - c);
      const dusk = warm > n * 0.04 || sb[sb.length >> 1] < 0.28;
      if (o.tint === true || ((o.tint ?? 'auto') === 'auto' && dusk)) {
        const blu = new Float32Array(n), lb = Float32Array.from(lum);
        for (let i = 0; i < n; i++) {
          const r = d[i * 4] / 255, g = d[i * 4 + 1] / 255, b = d[i * 4 + 2] / 255;
          blu[i] = (b - r) / (Math.max(r, g, b) + 0.05);
        }
        boxBlur(blu, w, h, 1); boxBlur(lb, w, h, 1);
        const Rt = Math.max(2, Math.round(w * 0.08)), bref = boxBlur(dilate(blu, w, h, Math.round(Rt * (o.tintRx ?? 1)), Math.round(Rt * (o.tintRy ?? 1))), w, h, Rt >> 1);
        // near the horizon the dusk sky itself turns pink / orange across the whole row: the bluer sky above
        // must not be the reference there, so the reference is capped at the row's own bluest pixels (90th
        // percentile of the bright ones, smoothed over a few rows) plus a margin
        const cap = o.tintRowCap ?? 0.1;
        if (cap > 0) {
          const rp = new Float32Array(h), buf = [];
          for (let y = 0; y < h; y++) {
            buf.length = 0;
            for (let x = 0; x < w; x++) if (lb[y * w + x] > 0.3) buf.push(blu[y * w + x]);
            buf.sort((a, c) => a - c);
            rp[y] = buf.length > w * 0.2 ? buf[Math.floor(buf.length * 0.9)] : 1;
          }
          const rr = Math.max(1, Math.round(h / 40));
          for (let y = 0; y < h; y++) {
            let m = -1e9; for (let j = -rr; j <= rr; j++) { const Y = y + j; if (Y >= 0 && Y < h) m = Math.max(m, rp[Y]); }
            for (let x = 0; x < w; x++) bref[y * w + x] = Math.min(bref[y * w + x], m + cap);
          }
        }
        // sky lightness: the brightest of the "locally bluest" pixels nearby (dark grey clouds can be as blue
        // as the dusk sky, but never as bright)
        let sl = new Float32Array(n);
        for (let i = 0; i < n; i++) if (blu[i] > bref[i] - (o.tintSkyTol ?? 0.03) && lb[i] > (o.tintSkyMinL ?? 0.45)) sl[i] = lb[i];
        sl = dilate(sl, w, h, Math.round(Rt * (o.tintLumR ?? 2)));
        // no blue-enough sky nearby (a pink / orange band over the horizon): carry the sky lightness down from
        // above instead of leaving 0, which the blur would turn into "the sky here is dark"
        for (let y = 1; y < h; y++) for (let x = 0; x < w; x++) if (!sl[y * w + x]) sl[y * w + x] = sl[(y - 1) * w + x];
        sl = boxBlur(sl, w, h, Rt);
        const g = new Float32Array(n);
        for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
          const i = y * w + x;
          g[i] = Math.abs(lum[i + 1] - lum[i - 1]) + Math.abs(lum[i + w] - lum[i - w]);
        }
        boxBlur(g, w, h, 1);
        const ta = o.tintBlue ?? 0.1, tl = o.tintLum ?? 0.12, tb = o.tintBias ?? 1;
        for (let i = 0; i < n; i++) {
          const r = d[i * 4] / 255, gg = d[i * 4 + 1] / 255, b = d[i * 4 + 2] / 255;
          if (b - gg < -0.03 && lum[i] > 0.8) { pale[i] = 1; continue; }   // horizon glow
          const L = sl[i] > 0.1 ? sl[i] : lb[i];
          const s = Math.hypot(Math.max(0, bref[i] - blu[i]) / ta, (lb[i] - L) / tl);
          if (s > tb + o.thrBias * 4 && lb[i] > (o.tintDark ?? 0.3) && g[i] < (o.tintTex ?? 0.12)) mask[i] = 1;
          else if (!mask[i] && blu[i] > bref[i] - 0.06 && lb[i] > 0.3) pale[i] = 1;   // the sky here
        }
        thrMode = 'tint'; if (o.debug) Object.assign(o.debug, { bref, sl, lb, blu, g });
      }
    } else {
      thr = Math.min(0.95, Math.max(0.05, thr + o.thrBias));
      for (let i = 0; i < n; i++) mask[i] = white[i] > thr ? 1 : 0;
    }
    // keep only the regions that look like clouds: mostly surrounded by blue sky, smooth inside, not tiny
    const comps = [];
    if (thrMode !== 'otsu' && (o.skyRing ?? 0.55) > 0) {
      const lab = new Int32Array(n).fill(-1), st = [];
      for (let s0 = 0; s0 < n; s0++) {
        if (!mask[s0] || lab[s0] >= 0) continue;
        const id = comps.length, px = [];
        let ring = 0, ringSky = 0, tex = 0, nIn = 0, x0 = w, x1 = 0;
        lab[s0] = id; st.push(s0);
        while (st.length) {
          const i = st.pop(), x = i % w, y = (i - x) / w; px.push(i);
          if (x < x0) x0 = x; if (x > x1) x1 = x;
          // texture inside (not on the rim): windows and lamps are busy, clouds are smooth
          if (x > 1 && x < w - 2 && y > 1 && y < h - 2 && mask[i - 2] && mask[i + 2] && mask[i - 2 * w] && mask[i + 2 * w]) {
            tex += Math.abs(lum[i + 1] - lum[i - 1]) + Math.abs(lum[i + w] - lum[i - w]); nIn++; }
          for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            const X = x + dx, Y = y + dy;
            if (X < 0 || Y < 0 || X >= w || Y >= h) continue;   // the image border does not count
            const j = Y * w + X;
            if (mask[j]) { if (lab[j] < 0) { lab[j] = id; st.push(j); } }
            else { ring++; if (sky[j] || pale[j]) ringSky++; }
          }
        }
        const c = { area: px.length, sky: ring ? ringSky / ring : 1, tex: nIn >= 8 ? tex / nIn : -1, wid: x1 - x0 + 1 };
        // a big cloud bank may sit on the skyline (less sky around it); a small bright patch must be almost all sky
        // around, else it is a lit wall / window / lamp
        const small = c.area < n * 0.006;
        c.keep = c.area >= n * (o.minArea ?? 0.0004) && c.sky >= (small ? 0.8 : (o.skyRing ?? 0.4)) 
          && (c.tex < 0 || (c.tex <= (o.texMax ?? 0.2) && (c.tex >= (thrMode === 'tint' ? (o.texMinTint ?? 0.012) : (o.texMin ?? 0.03))
            // a thin, wispy cloud is smooth too: fine when it floats in blue sky (haze and glare sit on the skyline)
            || (c.sky >= (o.wispSky ?? 0.9) && c.tex >= (o.texMinWisp ?? 0.008)))));   // too smooth = haze / glare
        if (!c.keep) for (const i of px) mask[i] = 0;
        comps.push(c);
      }
      for (let i = 0; i < n; i++) if (!mask[i]) lab[i] = -1;
      var label = lab;
    }
    let cover = 0;
    for (let i = 0; i < n; i++) cover += mask[i];
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
    // open = cloud or sky (for keeping faces off the buildings)
    const open = new Uint8Array(n);
    for (let i = 0; i < n; i++) open[i] = mask[i] || sky[i] || pale[i] ? 1 : 0;
    return { w, h, white, mask, D, thr, thrMode, sky, open, comps, label: label || null, cover: cover / (w * h) };
  }

  function find(det, o) {
    const { w, h, D } = det, minR = o.minR * w, maxR = o.maxR * w, minFace = (o.minFace ?? 0) * w;
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
    // the shape of a cloud (cached per component): how long it is (PCA), and its centre line along the long axis
    const shapes = new Map();
    const shapeOf = (lab) => {
      if (lab < 0 || !det.label) return null;
      if (shapes.has(lab)) return shapes.get(lab);
      let n = 0, mx = 0, my = 0, sxx = 0, syy = 0, sxy = 0;
      for (let i = 0; i < w * h; i++) if (det.label[i] === lab) { const x = i % w, y = (i - x) / w; n++; mx += x; my += y; }
      if (n < 20) { shapes.set(lab, null); return null; }
      mx /= n; my /= n;
      for (let i = 0; i < w * h; i++) if (det.label[i] === lab) { const x = i % w - mx, y = (i - (i % w)) / w - my; sxx += x * x; syy += y * y; sxy += x * y; }
      sxx /= n; syy /= n; sxy /= n;
      const tr = sxx + syy, dt = Math.sqrt(Math.max(0, (sxx - syy) ** 2 / 4 + sxy * sxy)), l1 = tr / 2 + dt, l2 = Math.max(1e-6, tr / 2 - dt);
      const th = 0.5 * Math.atan2(2 * sxy, sxx - syy), ux = Math.cos(th), uy = Math.sin(th);
      // centre line: bins along the long axis, the middle of the cloud across it
      const B = 2, lo = new Map(), hi = new Map();
      for (let i = 0; i < w * h; i++) if (det.label[i] === lab) {
        const x = i % w - mx, y = (i - (i % w)) / w - my, sv = x * ux + y * uy, q = -x * uy + y * ux, k = Math.round(sv / B);
        if (!lo.has(k) || q < lo.get(k)) lo.set(k, q); if (!hi.has(k) || q > hi.get(k)) hi.set(k, q);
      }
      const ks = [...lo.keys()].sort((a, c) => a - c);
      let line = ks.map((k) => { const q = (lo.get(k) + hi.get(k)) / 2; return { x: mx + k * B * ux - q * uy, y: my + k * B * uy + q * ux, half: (hi.get(k) - lo.get(k)) / 2 + 0.5 }; });
      line = line.map((p, i) => { let ax = 0, ay = 0, ah = 0, c = 0;   // smooth
        for (let j = -2; j <= 2; j++) { const q = line[i + j]; if (q) { ax += q.x; ay += q.y; ah += q.half; c++; } }
        return { x: ax / c, y: ay / c, half: ah / c }; });
      // bend (how far the middle of the centre line sits off the chord, / chord length), the thickness of the
      // middle vs the ends, and whether the ends are lifted above the middle (wings)
      const A = line[0], Z = line[line.length - 1], ch = Math.hypot(Z.x - A.x, Z.y - A.y) || 1;
      let bend = 0; for (const p of line) bend = Math.max(bend, Math.abs((p.x - A.x) * (Z.y - A.y) - (p.y - A.y) * (Z.x - A.x)) / ch);
      const third = Math.max(1, Math.floor(line.length / 3)), fifth = Math.max(1, Math.floor(line.length / 5));
      const avgH = (a) => a.reduce((t, p) => t + p.half, 0) / Math.max(1, a.length);
      const mid = line[line.length >> 1];
      const sh = { n, elong: Math.sqrt(l1 / l2), len: ks.length ? (ks[ks.length - 1] - ks[0]) * B : 0, line, thick: n / Math.max(1, ks.length * B),
        bend: bend / ch, flat: Math.abs(uy) < 0.5,
        midThick: avgH(line.slice(third, line.length - third)) / Math.max(0.5, (avgH(line.slice(0, fifth)) + avgH(line.slice(-fifth))) / 2),
        endsUp: (mid.y - (A.y + Z.y) / 2) / ch };
      shapes.set(lab, sh);
      return sh;
    };
    // candidates: local maxima of the distance map (5x5)
    // a thin or small cloud has a small inscribed circle; let its face grow toward the size of the whole cloud
    const blobK = o.blobFace ?? 0.9, eqR = (i) => det.label && det.label[i] >= 0 ? Math.sqrt(det.comps[det.label[i]].area / Math.PI) : 0;
    const cand = [];
    for (let y = 2; y < h - 2; y++) for (let x = 2; x < w - 2; x++) {
      let v = D[y * w + x]; if (v < 1.5) continue;
      const e = eqR(y * w + x);
      if (Math.max(v, e) < minR) continue;   // the cloud (or the bulge) is too small
      let ok = true;
      for (let j = -2; j <= 2 && ok; j++) for (let i = -2; i <= 2; i++) if (D[(y + j) * w + x + i] > v) { ok = false; break; }
      if (ok) cand.push({ x, y, r: Math.max(minFace, Math.min(Math.max(v, blobK * e), v * 3)), rin: v, lab: det.label ? det.label[y * w + x] : -1, e });
    }
    cand.sort((a, b) => b.r - a.r);
    const out = [];
    const n = w * h, shapeOn = o.shapeKind ?? true;
    // peek: a roundish hole of sky inside one cloud -> an animal looking through it (eyes in the hole, ear tips
    // and paws on its rim); that cloud gets no other character
    if (shapeOn && (o.peek ?? true) && det.label) {
      const seen = new Uint8Array(n), st = [];
      for (let s0 = 0; s0 < n; s0++) {
        if (det.mask[s0] || seen[s0]) continue;
        const px = []; let border = false, labs = new Map(), other = 0;
        seen[s0] = 1; st.push(s0);
        while (st.length) {
          const i = st.pop(), x = i % w, y = (i - x) / w; px.push(i);
          if (x === 0 || y === 0 || x === w - 1 || y === h - 1) border = true;
          for (const j of [i - 1, i + 1, i - w, i + w]) {
            if (j < 0 || j >= n || (Math.abs((j % w) - x) > 1)) continue;
            if (det.mask[j]) { const l = det.label[j]; if (l >= 0) labs.set(l, (labs.get(l) || 0) + 1); else other++; }
            else if (!seen[j]) { seen[j] = 1; st.push(j); }
          }
        }
        if (border || px.length < n * (o.holeMin ?? 0.008) || px.length > n * (o.holeMax ?? 0.06) || labs.size !== 1 || other) continue;
        let mx = 0, my = 0, x0 = w, x1 = 0, y0 = h, y1 = 0;
        for (const i of px) { const x = i % w, y = (i - x) / w; mx += x; my += y; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
        mx /= px.length; my /= px.length;
        const bw = x1 - x0 + 1, bh = y1 - y0 + 1, rh = Math.sqrt(px.length / Math.PI), lab = [...labs.keys()][0];
        if (Math.max(bw, bh) / Math.min(bw, bh) > 2.2 || px.length < 0.45 * bw * bh) continue;   // not roundish
        if (rh < Math.max(minR, minFace) * (o.peekMin ?? 0.9) || det.comps[lab].area < px.length * 6 || out.some((f) => f.lab === lab)) continue;
        out.push({ cx: mx, cy: my, cr: rh, r: Math.min(maxR, rh * 1.05), fx: mx, fy: my, ears: null, paws: null, arms: null,
          hint: 'peek', hole: { rx: bw / 2 / Math.min(maxR, rh * 1.05), ry: bh / 2 / Math.min(maxR, rh * 1.05) }, lab });   // hole size in face units
        if (out.length >= Math.min(2, o.maxFaces)) break;
      }
    }
    // small clouds in numbers -> a flock of sheep
    const nSmall = det.comps ? det.comps.filter((cc) => cc.keep && cc.area < n * (o.sheepMax ?? 0.015) && cc.area > n * 0.0015).length : 0;
    for (const c of cand) {
      if (out.length >= o.maxFaces) break;
      // the face must (mostly) fit in the picture
      const r0 = Math.min(c.r, maxR) * (o.faceScale ?? 1) * (o.edgeFit ?? 0.7);
      if (c.x - r0 < 0 || c.x + r0 > w || c.y - r0 < 0) continue;
      if (out.some((f) => Math.hypot(f.cx - c.x, f.cy - c.y) < o.sep * (f.cr + c.r))) continue;
      // one character per cloud (two on a very big cloud bank), so a photo reads as "a cat in the clouds"
      // rather than a crowd; perCloud 0 = no limit
      if (c.lab >= 0 && (o.perCloud ?? 1) > 0) {
        const cc = det.comps[c.lab], pc = o.perCloud ?? 1;
        let lim = cc.area > w * h * (o.bigCloud ?? 0.5) ? 2 * pc : pc;
        // a long thin bank along the horizon: one more for every ~30% of the picture's width it spans;
        // a thick cloud that just happens to be wide stays one character (mean thickness = area / width)
        if (cc.wid && cc.area / cc.wid < h * (o.thinCloud ?? 0.06)) lim = Math.max(lim, Math.min(3 * pc, pc + Math.floor(cc.wid / (w * (o.longCloud ?? 0.3)))));
        if (out.filter((f) => f.lab === c.lab).length >= lim) continue;
      }
      // the character follows the shape (o.shapeKind, default on): a long thin cloud -> a snake (its head at the
      // thicker end, the body along the cloud's centre line), a round fat one -> a bear, else a cat
      const sh = (o.shapeKind ?? true) ? shapeOf(c.lab) : null;
      let hint = null;
      if (sh) {
        if (sh.elong > (o.snakeElong ?? 3.2) && sh.len > w * (o.snakeLen ?? 0.3)) hint = 'snake';
        else if (sh.n > n * (o.whaleArea ?? 0.3)) hint = 'whale';             // a huge cloud
        else if (nSmall >= (o.sheepCount ?? 4) && sh.n < n * (o.sheepMax ?? 0.015)) hint = 'sheep';
        // wings: a lying cloud, thick in the middle and thin toward both ends; raised ends -> a bat, else a bird
        else if (sh.flat && sh.elong > 1.8 && sh.midThick > (o.wingThick ?? 1.9) && sh.endsUp > (o.batLift ?? 0.12)) hint = 'bat';
        else if (sh.flat && sh.elong > 1.8 && sh.midThick > (o.wingThick ?? 1.9)) hint = 'bird';
        else if (sh.elong < (o.bearElong ?? 1.6)) hint = 'bear';
        else if (sh.elong > (o.dolphinElong ?? 2.4) && sh.bend > (o.dolphinBend ?? 0.12)) hint = 'dolphin';   // long and arched
        else if (sh.elong > (o.fishElong ?? 2.7)) hint = 'fish';   // a long straight oval: a fish, its tail at the far end
        else hint = 'cat';
      } else if (shapeOn && det.cover > (o.whaleCover ?? 0.45)) hint = 'whale';   // overcast: one big whale
      const solo = (k) => k === 'snake' || k === 'whale' || k === 'peek';
      if (out.some((f) => f.lab === c.lab && f.lab >= 0 && (solo(f.hint) || solo(hint)))) continue;
      if (hint === 'whale' && out.some((f) => f.hint === 'whale')) continue;
      if (hint === 'snake') {
        const L = sh.line, m = Math.max(2, Math.floor(L.length * 0.25));
        const avg = (a) => a.reduce((t, p) => t + p.half, 0) / a.length;
        const fwd = avg(L.slice(0, m)) >= avg(L.slice(-m)), pts = fwd ? L : [...L].reverse();
        const hr = Math.min(maxR, Math.max(minFace, Math.max(...pts.slice(0, m * 2).map((p) => p.half)) * 0.85));
        // the head: the first centre-line point that is hr in from the end
        let k = 0, run = 0;
        while (k < pts.length - 1 && run < hr * 0.9) { run += Math.hypot(pts[k + 1].x - pts[k].x, pts[k + 1].y - pts[k].y); k++; }
        const hp = pts[k], body = [];
        for (let j = k; j < pts.length; j += 2) body.push({ x: pts[j].x, y: pts[j].y, half: pts[j].half });
        if (body.length >= 4) {
          out.push({ cx: c.x, cy: c.y, cr: c.r, r: hr, fx: hp.x, fy: hp.y, ears: null, paws: null, arms: null, body, hint, lab: c.lab });
          continue;
        }
        hint = 'cat';
      }
      let r = Math.min(c.r, maxR);
      // a face bigger than its cloud must not cover buildings / trees: shrink it while much of the disc is
      // neither cloud nor sky
      if (det.open) {
        const ground = (rr) => { let bad = 0, tot = 0;
          for (let k = 0; k < 24; k++) for (const q of [0.55, 0.95]) {
            const X = Math.round(c.x + Math.cos(k * Math.PI / 12) * rr * q), Y = Math.round(c.y + Math.sin(k * Math.PI / 12) * rr * q);
            if (X < 0 || Y < 0 || X >= w || Y >= h) continue;
            tot++; if (!det.open[Y * w + X]) bad++;
          }
          return tot ? bad / tot : 0; };
        while (r > Math.max(minFace, c.rin) && ground(r) > (o.groundMax ?? 0.15)) r *= 0.9;
      }
      // slide the face up toward the top of the bulge while there is still room for it
      let fx = c.x, fy = c.y;
      const step = Math.max(0.5, r * 0.05);
      for (let k = 0; k < 60 && o.faceUp > 0; k++) {
        const ny = fy - step, top = march(fx, ny, 0, -1, r * 4).t;
        if (distAt(fx, ny) < r * (1 - 0.35 * o.faceUp) || top < r * (1.25 - 0.2 * o.faceUp)) break;
        fy = ny;
      }
      const fr = r * o.faceScale;
      // keep the whole face (and the ears) in the picture
      if (o.clampFace ?? true) {
        fx = Math.min(w - fr, Math.max(fr, fx));
        fy = Math.max(fr * 1.3, fy);
      }
      // ears: two rays up and outward; on the outline when it is near, else on the head line
      let ears = [];
      for (const s of [-1, 1]) {
        const a = 30 * Math.PI / 180, dx = s * Math.sin(a), dy = -Math.cos(a);
        const m = march(fx, fy, dx, dy, fr * 1.25);
        if (m.edge && m.t < fr * 0.7) { ears = null; break; }
        // a face bigger than its cloud keeps the ears on its own head line
        const onEdge = !m.far && !m.edge && m.t >= fr * 0.8, tt = Math.max(fr * 0.8, Math.min(m.t, fr * 1.0));
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
      // arms: where the cloud reaches out sideways from the face, a hand at the end of that lobe
      let arms = null;
      if (o.arms ?? true) {
        arms = [];
        for (const s of [-1, 1]) {
          let best = null;
          for (let deg = 50; deg <= 150; deg += 10) {
            const a = deg * Math.PI / 180, dx = s * Math.sin(a), dy = -Math.cos(a);
            const m = march(fx, fy, dx, dy, fr * 3.5);
            if (m.edge || m.far) continue;
            if (!best || m.t > best.t) best = { t: m.t, dx, dy };
          }
          if (best && best.t > fr * (o.armMin ?? 1.7)) {
            const t1 = best.t - fr * 0.22;
            arms.push({ sx: fx + best.dx * fr * 0.8, sy: fy + best.dy * fr * 0.8, x: fx + best.dx * t1, y: fy + best.dy * t1, side: s });
          }
        }
        if (!arms.length) arms = null;
      }
      let tail = null, wings = null;
      if (hint === 'bird' || hint === 'bat') {
        ears = null; arms = null; if (hint === 'bat') paws = null;
        const L = sh.line, A = L[Math.min(L.length - 1, 1)], Z = L[Math.max(0, L.length - 2)];
        wings = (A.x < Z.x ? [A, Z] : [Z, A]).map((p) => ({ x: p.x, y: p.y }));
        if (Math.min(...wings.map((p) => Math.hypot(p.x - fx, p.y - fy))) < fr * 1.2) { hint = 'cat'; wings = null; }
      }
      if (hint === 'fish' || hint === 'dolphin') {
        ears = null; paws = null; arms = null;
        const L = sh.line, A = L[0], B = L[L.length - 1];
        const far = Math.hypot(A.x - fx, A.y - fy) > Math.hypot(B.x - fx, B.y - fy) ? 0 : L.length - 1;
        const e = L[far], q = L[far === 0 ? Math.min(L.length - 1, 3) : Math.max(0, L.length - 4)];
        let dx = e.x - q.x, dy = e.y - q.y; const l = Math.hypot(dx, dy) || 1;
        // the fin sits a little inside the end of the cloud
        const back = Math.min(fr * 0.35, Math.hypot(e.x - fx, e.y - fy) * 0.15);
        tail = { x: e.x - dx / l * back, y: e.y - dy / l * back, dx: dx / l, dy: dy / l };
        if (Math.hypot(e.x - fx, e.y - fy) < fr * 1.6) { hint = 'cat'; tail = null; }
      }
      if (hint === 'whale') { ears = null; paws = null; arms = null; }
      if (hint === 'sheep') arms = null;
      out.push({ cx: c.x, cy: c.y, cr: c.r, r, fx, fy, ears, paws: arms ? null : paws, arms, tail, wings, hint, lab: c.lab });
    }
    // drop the small ones next to a big one (relMin x the biggest face)
    const big = out.reduce((m, f) => Math.max(m, f.r), 0), rel = o.relMin ?? 0.45;
    // (the peeking animals stay, small as they are, and so does a flock of sheep when there is nothing bigger)
    const flock = out.length && out.every((f) => f.hint === 'sheep' || f.hint === 'peek');   // only small clouds: a flock
    return det.label ? out.filter((f) => f.r >= rel * big || f.hint === 'peek' || flock) : out;
  }
  // outlines of the cloud mask (detector px), for tracing the clouds with a pen line:
  // marching squares on a slightly blurred mask, linked into polylines, then smoothed (Chaikin).
  // Lines end at the picture's border instead of running along it.
  //   o.outlineBlur (1), o.outlineMin (shortest line kept, px; 8), o.outlineSmooth (Chaikin passes; 2)
  function outlines(det, o = {}) {
    const { w, h } = det, f = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) f[i] = det.mask[i];
    boxBlur(f, w, h, o.outlineBlur ?? 1);
    const iso = 0.5, segs = [], pt = new Map();
    // edge keys: horizontal edge (x,y)-(x+1,y) = 2*(y*w+x), vertical edge (x,y)-(x,y+1) = 2*(y*w+x)+1
    const P = (k) => {
      let p = pt.get(k); if (p) return k;
      const c = k >> 1, x = c % w, y = (c - x) / w, v0 = f[c], v1 = k & 1 ? f[c + w] : f[c + 1];
      const t = Math.abs(v1 - v0) < 1e-6 ? 0.5 : (iso - v0) / (v1 - v0);
      pt.set(k, k & 1 ? [x, y + t] : [x + t, y]);
      return k;
    };
    for (let y = 0; y < h - 1; y++) for (let x = 0; x < w - 1; x++) {
      const i = y * w + x, a = f[i], b = f[i + 1], c = f[i + w + 1], d = f[i + w];
      const code = (a > iso ? 1 : 0) | (b > iso ? 2 : 0) | (c > iso ? 4 : 0) | (d > iso ? 8 : 0);
      if (code === 0 || code === 15) continue;
      const T = 2 * i, B = 2 * (i + w), L = 2 * i + 1, R = 2 * (i + 1) + 1;
      const add = (e0, e1) => segs.push([P(e0), P(e1)]);
      const mid = (a + b + c + d) / 4 > iso;
      switch (code) {
        case 1: case 14: add(L, T); break;
        case 2: case 13: add(T, R); break;
        case 3: case 12: add(L, R); break;
        case 4: case 11: add(R, B); break;
        case 6: case 9: add(T, B); break;
        case 7: case 8: add(L, B); break;
        case 5: if (mid) { add(T, R); add(L, B); } else { add(L, T); add(R, B); } break;
        case 10: if (mid) { add(L, T); add(R, B); } else { add(T, R); add(L, B); } break;
      }
    }
    const at = new Map();
    segs.forEach((s, j) => { for (const k of s) { const l = at.get(k); if (l) l.push(j); else at.set(k, [j]); } });
    const used = new Uint8Array(segs.length), lines = [];
    const walk = (k, from) => {   // follow the chain from edge key k, away from segment `from`
      const ks = [];
      for (;;) {
        const nx = (at.get(k) || []).find((j) => j !== from && !used[j]);
        if (nx === undefined) break;
        used[nx] = 1; from = nx; k = segs[nx][0] === k ? segs[nx][1] : segs[nx][0]; ks.push(k);
      }
      return ks;
    };
    for (let j = 0; j < segs.length; j++) {
      if (used[j]) continue;
      used[j] = 1;
      const fw = walk(segs[j][1], j), bw = walk(segs[j][0], j);
      const ks = [...bw.reverse(), segs[j][0], segs[j][1], ...fw];
      const closed = ks.length > 3 && ks[0] === ks[ks.length - 1];
      let pts = ks.map((k) => pt.get(k));
      if (closed) pts.pop();
      // drop the parts that run along the picture's border
      const onB = ([x, y]) => x < 0.6 || y < 0.6 || x > w - 1.6 || y > h - 1.6;
      const runs = [];
      let cur = [];
      for (const q of pts) { if (onB(q)) { if (cur.length) runs.push(cur); cur = []; } else cur.push(q); }
      if (cur.length) runs.push(cur);
      const whole = runs.length === 1 && runs[0].length === pts.length;
      if (closed && !whole && runs.length > 1 && !onB(pts[0]) && !onB(pts[pts.length - 1])) {
        runs[0] = runs.pop().concat(runs[0]);   // the run that wraps around the start
      }
      for (const r of runs) {
        let q = r, len = 0;
        for (let m = 1; m < q.length; m++) len += Math.hypot(q[m][0] - q[m - 1][0], q[m][1] - q[m - 1][1]);
        if (len < (o.outlineMin ?? 8)) continue;
        const cl = closed && whole;
        for (let it = 0; it < (o.outlineSmooth ?? 2); it++) {
          const out = [], n = q.length;
          if (!cl) out.push(q[0]);
          for (let m = 0; m < (cl ? n : n - 1); m++) {
            const p0 = q[m], p1 = q[(m + 1) % n];
            out.push([0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]], [0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]]);
          }
          if (!cl) out.push(q[n - 1]);
          q = out;
        }
        lines.push({ pts: q, closed: cl, len });
      }
    }
    return lines;
  }

  return { analyse, find, outlines, otsu, boxBlur };
})();
