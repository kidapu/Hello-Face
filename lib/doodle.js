// Doodle: pen-line cute faces (cat / bear / plain) drawn with Canvas 2D, shared by
// cloud_doodle.html (still photos) and fluid_cats.html (the ghost fluid).
//
//   Doodle.draw(ctx, d, o)  d = one face in canvas px:
//     { x, y, U, rot, alpha, kind, expr, seed, pupil, gx, gy, blinkT,
//       ears: [{x, y, nx, ny}, {..}] | null   (base points on the outline + outward normal)
//       paws: [{x, y}, {..}] | null }         (points on the bottom outline)
//   o = style: { ink, pupilInk, lineW, boil, boilFrame, eyeSize, eyeGap, earSize,
//                ears, whiskers, paws, blush, dpr }
//   Doodle.life(d, dt, o) advances gaze and blinking ({ gaze, blink }).
window.Doodle = (() => {
  const KINDS = ['cat', 'bear', 'plain'];
  const EXPRS = ['pokan', 'sleepy', 'smile', 'happy', 'smug'];
  function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

  let ctx, O, LW = 2;
  // a wobbly pen stroke; the wobble is smooth along the stroke and changes boilFps times a
  // second (o.boilFrame), like hand-drawn animation
  function pen(pts, { closed = false, seed = 0, fill = null, width = 1, noStroke = false, color = null } = {}) {
    if (pts.length < 2) return;
    const r = mulberry(seed * 31 + O.boilFrame * 7919 + 3), a1 = r() * 6.28, a2 = r() * 6.28, a3 = r() * 6.28, a4 = r() * 6.28;
    const amp = O.boil * LW * 0.45, n = pts.length;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const u = i / (n - 1) * 6.28;
      const x = pts[i][0] + amp * (0.7 * Math.sin(u * 0.8 + a1) + 0.3 * Math.sin(u * 2.1 + a2));
      const y = pts[i][1] + amp * (0.7 * Math.sin(u * 0.9 + a3) + 0.3 * Math.sin(u * 1.7 + a4));
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    }
    if (closed) ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (!noStroke) { ctx.lineWidth = LW * width * (0.9 + 0.2 * r()); ctx.strokeStyle = color || O.ink; ctx.stroke(); }
  }
  function ellPts(cx, cy, rx, ry, a0 = 0, a1 = Math.PI * 2, n = 28) {
    const p = []; for (let i = 0; i <= n; i++) { const a = a0 + (a1 - a0) * i / n; p.push([cx + Math.cos(a) * rx, cy + Math.sin(a) * ry]); } return p;
  }
  function quad(p0, c, p1, n = 12) {
    const p = []; for (let i = 0; i <= n; i++) { const t = i / n, s = 1 - t;
      p.push([s * s * p0[0] + 2 * s * t * c[0] + t * t * p1[0], s * s * p0[1] + 2 * s * t * c[1] + t * t * p1[1]]); } return p;
  }

  function draw(context, d, o) {
    ctx = context; O = o;
    const U = d.U, kind = d.kind, expr = d.expr, sd = d.seed;
    if (U < 4 || !(d.alpha > 0.01)) return;
    LW = Math.max(1, o.lineW * (o.dpr || 1) * Math.min(1.8, Math.max(0.6, Math.sqrt(U / 70))));
    ctx.save();
    ctx.globalAlpha = Math.min(1, d.alpha);
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';

    // ears (standing on the outline), in canvas px
    if (o.ears && d.ears && kind !== 'plain') d.ears.forEach((e, k) => {
      const bx = e.x, by = e.y, nx = e.nx, ny = e.ny, tx = -ny, ty = nx, es = U * o.earSize;
      if (kind === 'cat') {
        const hw = 0.30 * es, hh = 0.55 * es, sink = 0.10 * es, lean = hw * 0.15 * (k ? 1 : -1);
        const b0 = [bx - nx * sink - tx * hw, by - ny * sink - ty * hw], b1 = [bx - nx * sink + tx * hw, by - ny * sink + ty * hw];
        const tip = [bx + nx * hh + tx * lean, by + ny * hh + ty * lean];
        pen([...quad(b0, [(b0[0] + tip[0]) / 2 - tx * hw * 0.12, (b0[1] + tip[1]) / 2 - ty * hw * 0.12], tip, 8),
             ...quad(tip, [(b1[0] + tip[0]) / 2 + tx * hw * 0.12, (b1[1] + tip[1]) / 2 + ty * hw * 0.12], b1, 8).slice(1)], { seed: sd + 11 + k });
        const i0 = [bx - tx * hw * 0.5 + nx * hh * 0.08, by - ty * hw * 0.5 + ny * hh * 0.08];
        const it = [bx + nx * hh * 0.68, by + ny * hh * 0.68], i1 = [bx + tx * hw * 0.5 + nx * hh * 0.08, by + ty * hw * 0.5 + ny * hh * 0.08];
        pen([i0, it, i1], { seed: sd + 21 + k, width: 0.7 });
      } else {
        const rr = 0.26 * es, c = [bx + nx * rr * 0.35, by + ny * rr * 0.35], a = Math.atan2(ny, nx);
        pen(ellPts(c[0], c[1], rr, rr, a - 2.3, a + 2.3, 22), { seed: sd + 11 + k });
        pen(ellPts(c[0] - nx * rr * 0.1, c[1] - ny * rr * 0.1, rr * 0.5, rr * 0.5, a - 1.9, a + 1.9, 14), { seed: sd + 21 + k, width: 0.7 });
      }
    });
    // paws (bear): mittens hanging on the bottom outline
    if (o.paws && d.paws && kind === 'bear') d.paws.forEach((p, k) => {
      const s = U * 0.24, dir = k ? 1 : -1, c = [p.x - dir * s * 0.3, p.y - s * 0.7];
      pen(ellPts(c[0], c[1], s, s * 0.7, -0.15, Math.PI + 0.15, 18), { seed: sd + 41 + k });
      for (let j = -1; j <= 1; j += 2) pen([[c[0] + j * s * 0.33, c[1] + s * 0.68], [c[0] + j * s * 0.3, c[1] + s * 0.38]], { seed: sd + 51 + k * 3 + j, width: 0.7 });
    });

    // the face itself, in face units around (x, y), turned by rot
    ctx.translate(d.x, d.y); ctx.rotate(d.rot || 0);
    const L = (x, y) => [x * U, y * U];
    const eR = 0.155 * o.eyeSize * (kind === 'cat' ? 1.08 : 1), eX = 0.36 * o.eyeGap, eY = 0.0;
    const pupilInk = o.pupilInk || o.ink;
    let lidU = 0, lidD = 0;
    if (expr === 'sleepy') lidU = 0.55;
    if (expr === 'smug') { lidU = 0.45; lidD = 0.15; }
    if (d.blinkT >= 0) { const b = Math.sin(Math.min(1, d.blinkT / 0.16) * Math.PI); lidU += (1 - lidU) * b; lidD += (1 - lidD) * b; }
    for (const s of [-1, 1]) {
      const [ex, ey] = L(s * eX, eY), er = eR * U;
      if (expr === 'happy') {
        pen(ellPts(ex, ey + er * 0.35, er * 0.85, er * 0.75, Math.PI * 1.1, Math.PI * 1.9, 14), { seed: sd + 61 + s, width: 1.2 });
        continue;
      }
      const up = -er * (1 - lidU), dn = er * (1 - lidD);
      if (up >= dn - 0.5) { pen([[ex - er, ey + (up + dn) / 2], [ex + er, ey + (up + dn) / 2 + er * 0.05]], { seed: sd + 71 + s, width: 1.1 }); continue; }
      const poly = ellPts(ex, ey, er, er * 1.04, 0, Math.PI * 2, 30).map(([x, y]) => [x, Math.min(ey + dn, Math.max(ey + up, y))]);
      pen(poly, { closed: true, fill: '#ffffff', noStroke: true, seed: sd + 71 + s });
      ctx.save(); ctx.beginPath(); poly.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.clip();
      const pr = er * d.pupil, gx = d.gx * er * 0.4, gy = d.gy * er * 0.3 + (expr === 'sleepy' ? er * 0.25 : 0);
      ctx.fillStyle = pupilInk; ctx.beginPath(); ctx.ellipse(ex + gx, ey + gy, pr, pr * 1.06, 0, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(ex + gx - pr * 0.35, ey + gy - pr * 0.38, pr * 0.32, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(ex + gx + pr * 0.3, ey + gy + pr * 0.35, pr * 0.13, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      pen(poly, { closed: true, seed: sd + 81 + s });
      if (lidU > 0.2) pen([[ex - er * 1.08, ey + up - er * 0.02], [ex + er * 1.08, ey + up + er * 0.02]], { seed: sd + 91 + s, width: 1.3 });
    }
    if (o.blush > 0) {
      ctx.fillStyle = `rgba(255,120,150,${o.blush * 0.5})`;
      for (const s of [-1, 1]) { const [bx, by] = L(s * (eX + 0.1), 0.24); ctx.beginPath(); ctx.ellipse(bx, by, U * 0.11, U * 0.06, 0, 0, Math.PI * 2); ctx.fill(); }
    }
    const nY = 0.2;
    if (kind === 'cat') {
      const [nx, ny] = L(0, nY), w = 0.075 * U;
      pen([[nx - w, ny - w * 0.5], [nx + w, ny - w * 0.5], [nx, ny + w * 0.6]], { closed: true, fill: o.ink, seed: sd + 101 });
      mouthW(L, U, nY + 0.05, 0.11, expr, sd);
    } else if (kind === 'bear') {
      const [nx, ny] = L(0, nY - 0.02);
      pen(ellPts(nx, ny, 0.12 * U, 0.08 * U, 0, Math.PI * 2, 20), { closed: true, fill: o.ink, seed: sd + 101 });
      ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.ellipse(nx - 0.04 * U, ny - 0.03 * U, 0.03 * U, 0.016 * U, -0.2, 0, Math.PI * 2); ctx.fill();
      pen([L(0, nY + 0.06), L(0, nY + 0.13)], { seed: sd + 103, width: 0.9 });
      mouthW(L, U, nY + 0.13, 0.1, expr, sd);
    } else {
      mouthPlain(L, U, nY + 0.04, expr, sd);
    }
    if (kind === 'cat' && o.whiskers) for (const s of [-1, 1]) for (let j = 0; j < 3; j++) {
      const y0 = nY + 0.02 + j * 0.06, x0 = s * (eX + 0.05);
      pen([L(x0, y0), L(x0 + s * 0.42, y0 - 0.1 + j * 0.1)], { seed: sd + 121 + j * 2 + s, width: 0.65 });
    }
    ctx.restore();
  }
  // ω mouth (cat & bear) and its variants
  function mouthW(L, U, y, w, expr, sd) {
    if (expr === 'pokan') {
      const [mx, my] = L(0, y + 0.07);
      pen(ellPts(mx, my, 0.05 * U, 0.06 * U, 0, Math.PI * 2, 16), { closed: true, fill: '#3a2030', seed: sd + 111 });
      return;
    }
    const lift = expr === 'smug' ? 0.04 : 0;
    pen(quad(L(0, y), L(-w * 0.5, y + 0.09), L(-w, y + 0.01), 10), { seed: sd + 113 });
    pen(quad(L(0, y), L(w * 0.5, y + 0.09 - lift), L(w + lift * 0.5, y + 0.01 - lift * 1.5), 10), { seed: sd + 115 });
    if (expr === 'smile' || expr === 'happy')
      pen(quad(L(-w * 0.7, y + 0.06), L(0, y + 0.22), L(w * 0.7, y + 0.06), 10), { closed: true, fill: '#e8798f', seed: sd + 117, width: 0.9 });
  }
  function mouthPlain(L, U, y, expr, sd) {
    if (expr === 'pokan') { const [mx, my] = L(0, y + 0.04);
      pen(ellPts(mx, my, 0.07 * U, 0.08 * U, 0, Math.PI * 2, 16), { closed: true, fill: '#3a2030', seed: sd + 111 }); return; }
    if (expr === 'smug') { pen(quad(L(-0.12, y + 0.02), L(0.02, y + 0.08), L(0.14, y - 0.02), 10), { seed: sd + 113 }); return; }
    if (expr === 'smile' || expr === 'happy') {
      pen(quad(L(-0.16, y), L(0, y + 0.22), L(0.16, y), 12), { closed: true, fill: '#e8798f', seed: sd + 113 }); return; }
    pen(quad(L(-0.1, y + 0.02), L(0, y + 0.09), L(0.1, y + 0.02), 10), { seed: sd + 113 });
  }

  // gaze & blink; the state lives on d (gx, gy, tgx, tgy, nextGaze, nextBlink, blinkT)
  function life(d, dt, o) {
    if (d.nextGaze === undefined) { const r = mulberry(d.seed); d.gx = d.gy = d.tgx = d.tgy = 0; d.nextGaze = r() * 2; d.nextBlink = 1 + r() * 4; d.blinkT = -1; }
    const r = mulberry(d.seed + Math.floor(performance.now() / 997));
    if (o.gaze && (d.nextGaze -= dt) < 0) {
      d.nextGaze = 1.2 + r() * 3;
      const k = r(); d.tgx = k < 0.35 ? 0 : (r() * 2 - 1); d.tgy = k < 0.35 ? 0 : (r() * 2 - 1) * 0.7;
    }
    if (!o.gaze) d.tgx = d.tgy = 0;
    d.gx += (d.tgx - d.gx) * Math.min(1, dt * 10); d.gy += (d.tgy - d.gy) * Math.min(1, dt * 10);
    if (d.blinkT >= 0) { d.blinkT += dt; if (d.blinkT > 0.16) d.blinkT = -1; }
    else if (o.blink && (d.nextBlink -= dt) < 0) { d.blinkT = 0; d.nextBlink = 1.5 + r() * 4.5; }
  }
  // a pen line through pts (canvas px) in the same hand-drawn style; U sets the line weight
  function stroke(context, pts, o, { U = 70, alpha = 1, width = 1, seed = 0, closed = false } = {}) {
    ctx = context; O = o;
    LW = Math.max(1, o.lineW * (o.dpr || 1) * Math.min(1.8, Math.max(0.6, Math.sqrt(U / 70))));
    ctx.save(); ctx.globalAlpha = Math.min(1, alpha); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    pen(pts, { seed, width, closed });
    ctx.restore();
  }
  return { KINDS, EXPRS, mulberry, draw, life, stroke };
})();
