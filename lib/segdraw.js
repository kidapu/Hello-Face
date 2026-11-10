// SegDraw: the segmenter's result (segnet.js) as coloured pen outlines with names, shared by seg.html and the
// camera's AI view.
//   const v = SegDraw.build(seg, o)    seg: { W, H, label, group, names }   (at the model's own output size)
//   SegDraw.draw(ctx, v, map, o)       map: { ox, oy, kx, ky } grid -> screen (screen px per grid cell)
// o: { lines: true, by: 'class'|'group', blur, smooth, min, grow (screen px), big, lineW, labels, labelMin, dpr, nameOf(k) }
//
// Each class (or group) region: its edges by marching squares + Chaikin (CloudFind.outlines), drawn as curves
// through the midpoints, pushed outward so they go round the things instead of cutting into them.
//  - a background (a region over o.big of the picture): its edge along the picture's border is traced too
//    (so the frame shows which background it is), its lines move into it, and the holes the things make in it
//    are not drawn (the things draw their own line there: no double lines)
//  - the name sits at the point of the region farthest from its edges (inside a ring-shaped background, not in
//    the hole in the middle)
window.SegDraw = (() => {
  const GCOL = [[90, 150, 240], [255, 255, 255], [160, 110, 60], [60, 170, 60], [130, 130, 140], [230, 60, 60], [170, 90, 200]];
  const hue = (k) => { const h = (k * 137.508) % 360, f = (n) => { const q = (n + h / 30) % 12; return Math.round(255 * (0.55 - 0.45 * Math.max(-1, Math.min(q - 3, 9 - q, 1)))); }; return [f(0), f(8), f(4)]; };

  // distance (in cells) to the nearest cell outside the region (the picture's border counts as outside)
  function inside(mask, W, H) {
    const D = new Float32Array(W * H), q = Math.SQRT2, B = 1e9;
    for (let i = 0; i < W * H; i++) D[i] = mask[i] ? B : 0;
    const at = (x, y) => (x < 0 || y < 0 || x >= W || y >= H ? 0 : D[y * W + x]);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const i = y * W + x; if (D[i]) D[i] = Math.min(D[i], at(x - 1, y) + 1, at(x, y - 1) + 1, at(x - 1, y - 1) + q, at(x + 1, y - 1) + q); }
    for (let y = H - 1; y >= 0; y--) for (let x = W - 1; x >= 0; x--) { const i = y * W + x; if (D[i]) D[i] = Math.min(D[i], at(x + 1, y) + 1, at(x, y + 1) + 1, at(x + 1, y + 1) + q, at(x - 1, y + 1) + q); }
    return D;
  }

  function build(seg, o) {
    const W = seg.W, H = seg.H, n = W * H, byG = o.by === 'group', key = byG ? seg.group : seg.label, nk = byG ? 7 : seg.names.length;
    const cnt = new Int32Array(nk); for (let i = 0; i < n; i++) cnt[key[i]]++;
    const lines = [], labels = [];
    for (let k = 0; k < nk; k++) {
      if (!cnt[k]) continue;
      const col = byG ? GCOL[k] : hue(k), big = cnt[k] > n * (o.big ?? 0.07);
      const mask = new Uint8Array(n); for (let i = 0; i < n; i++) mask[i] = key[i] === k ? 1 : 0;
      // the name: at the cell deepest inside the region
      if (o.labels !== false && cnt[k] >= n * (o.labelMin ?? 0.015)) {
        const D = inside(mask, W, H); let bi = 0; for (let i = 1; i < n; i++) if (D[i] > D[bi]) bi = i;
        labels.push({ k, x: bi % W, y: Math.floor(bi / W), name: o.nameOf ? o.nameOf(k) : byG ? String(k) : seg.names[k] });
      }
      if (o.lines === false || cnt[k] < n * (o.min ?? 0.005)) continue;
      // a background is traced along the picture's border too: the mask in a frame of empty cells
      const pad = big ? (o.blur ?? 3) + 2 : 0, Wp = W + 2 * pad, Hp = H + 2 * pad;   // wider than the blur, or it fills the frame
      let mk = mask;
      if (pad) { mk = new Uint8Array(Wp * Hp); for (let y = 0; y < H; y++) mk.set(mask.subarray(y * W, y * W + W), (y + pad) * Wp + pad); }
      for (const l of CloudFind.outlines({ w: Wp, h: Hp, mask: mk }, { outlineBlur: o.blur ?? 3, outlineSmooth: o.smooth ?? 4, outlineMin: 3, outlineKeepBorder: !!pad })) {
        const q = l.pts.map(([x, y]) => [x - pad, y - pad]), m = q.length, nrm = [];
        for (let i = 0; i < m; i++) {
          const a = q[l.closed ? (i - 1 + m) % m : Math.max(0, i - 1)], b = q[l.closed ? (i + 1) % m : Math.min(m - 1, i + 1)];
          const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1; nrm.push([dy / len, -dx / len]);
        }
        // which side is the region: a vote over the line, sampling the mask a little to one side
        let vote = 0;
        for (let i = 0; i < m; i++) {
          const x = Math.round(q[i][0] + nrm[i][0] * 0.8), y = Math.round(q[i][1] + nrm[i][1] * 0.8);
          vote += x >= 0 && y >= 0 && x < W && y < H ? (mask[y * W + x] ? -1 : 1) : 1;
        }
        const away = vote >= 0 ? 1 : -1;   // nrm * away points away from the region
        if (big && l.closed) {
          // a hole a thing makes in the background (the region lies outside the loop): the thing draws it
          let cx = 0, cy = 0; for (const p of q) { cx += p[0]; cy += p[1]; } cx /= m; cy /= m;
          let out = 0; for (let i = 0; i < m; i++) out += (nrm[i][0] * (q[i][0] - cx) + nrm[i][1] * (q[i][1] - cy)) * away;
          if (out < 0) continue;
        }
        const s0 = away * (big ? -1 : 1);   // things' lines move away from them, a background's into it
        lines.push({ k, col, pts: q, nrm: nrm.map(([x, y]) => [x * s0, y * s0]), closed: l.closed });
      }
    }
    return { lines, labels };
  }

  function curve(ctx, q, closed) {
    ctx.beginPath();
    if (closed) {
      const m0 = [(q[q.length - 1][0] + q[0][0]) / 2, (q[q.length - 1][1] + q[0][1]) / 2]; ctx.moveTo(m0[0], m0[1]);
      for (let i = 0; i < q.length; i++) { const a = q[i], b = q[(i + 1) % q.length]; ctx.quadraticCurveTo(a[0], a[1], (a[0] + b[0]) / 2, (a[1] + b[1]) / 2); }
      ctx.closePath();
    } else {
      ctx.moveTo(q[0][0], q[0][1]);
      for (let i = 1; i < q.length - 1; i++) { const a = q[i], b = q[i + 1]; ctx.quadraticCurveTo(a[0], a[1], (a[0] + b[0]) / 2, (a[1] + b[1]) / 2); }
      ctx.lineTo(q[q.length - 1][0], q[q.length - 1][1]);
    }
  }
  function draw(ctx, v, map, o) {
    const dpr = o.dpr || 1, gw = (o.grow ?? 4) * dpr;
    ctx.save(); ctx.lineWidth = (o.lineW ?? 2.5) * dpr; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    for (const l of v.lines) {
      if (l.pts.length < 2) continue;
      const q = l.pts.map((p, i) => [map.ox + (p[0] + 0.5) * map.kx + l.nrm[i][0] * gw, map.oy + (p[1] + 0.5) * map.ky + l.nrm[i][1] * gw]);
      ctx.strokeStyle = `rgb(${l.col})`; curve(ctx, q, l.closed); ctx.stroke();
    }
    if (o.labels !== false && v.labels.length) {
      const fs = Math.round(11 * dpr);
      ctx.font = `800 ${fs}px "M PLUS Rounded 1c", system-ui, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (const t of v.labels) {
        const X = map.ox + (t.x + 0.5) * map.kx, Y = map.oy + (t.y + 0.5) * map.ky, tw = ctx.measureText(t.name).width + fs * 0.8;
        ctx.fillStyle = '#000'; ctx.beginPath(); ctx.roundRect(X - tw / 2, Y - fs * 0.8, tw, fs * 1.6, fs * 0.5); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.fillText(t.name, X, Y + fs * 0.05);
      }
    }
    ctx.restore();
  }
  return { build, draw, hue, GCOL };
})();
