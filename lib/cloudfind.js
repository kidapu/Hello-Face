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
          && (c.tex < 0 || (c.tex <= (o.texMax ?? 0.2) && c.tex >= (thrMode === 'tint' ? (o.texMinTint ?? 0.012) : (o.texMin ?? 0.03))));   // too smooth = haze / glare
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
        let lim = cc.area > w * h * (o.bigCloud ?? 0.1) ? 2 * pc : pc;
        // a long thin bank along the horizon: one more for every ~30% of the picture's width it spans
        if (cc.wid) lim = Math.max(lim, Math.min(3 * pc, pc + Math.floor(cc.wid / (w * (o.longCloud ?? 0.3)))));
        if (out.filter((f) => f.lab === c.lab).length >= lim) continue;
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
      out.push({ cx: c.x, cy: c.y, cr: c.r, r, fx, fy, ears, paws, lab: c.lab });
    }
    // drop the small ones next to a big one (relMin x the biggest face)
    const big = out.reduce((m, f) => Math.max(m, f.r), 0), rel = o.relMin ?? 0.45;
    return det.label ? out.filter((f) => f.r >= rel * big) : out;
  }
  return { analyse, find, otsu, boxBlur };
})();
