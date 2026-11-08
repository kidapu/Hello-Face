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
  const KINDS = ['cat', 'bear', 'snake', 'fish', 'dolphin', 'bird', 'bat', 'sheep', 'whale', 'peek', 'plain'];
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

    // snake: the body along the cloud's centre line (d.body: [{x, y, half}] in canvas px, from the head to the tail)
    if (kind === 'snake' && o.snakeBody && d.body && d.body.length > 3) snakeBody(d.body, d.x, d.y, U, sd);
    // arms (d.arms: [{sx, sy, x, y}] canvas px): a short tube from the face to a round hand at the end of a lobe
    if (o.arms !== false && d.arms && (kind === 'cat' || kind === 'bear')) d.arms.forEach((a, k) => {
      let dx = a.x - a.sx, dy = a.y - a.sy; const l = Math.hypot(dx, dy) || 1; dx /= l; dy /= l;
      const px = -dy, py = dx, aw = U * 0.085, hr = U * 0.15, ex = a.x - dx * hr * 0.8, ey = a.y - dy * hr * 0.8;
      pen([[a.sx + px * aw, a.sy + py * aw], [ex + px * aw * 0.85, ey + py * aw * 0.85]], { seed: sd + 131 + k });
      pen([[a.sx - px * aw, a.sy - py * aw], [ex - px * aw * 0.85, ey - py * aw * 0.85]], { seed: sd + 133 + k });
      const ang = Math.atan2(dy, dx);
      pen(ellPts(a.x, a.y, hr, hr * 0.9, ang - 2.5, ang + 2.5, 18), { seed: sd + 135 + k });
      for (const j of [-1, 1]) {   // two little finger lines
        const fa = ang + j * 0.45;
        pen([[a.x + Math.cos(fa) * hr * 0.95, a.y + Math.sin(fa) * hr * 0.95], [a.x + Math.cos(fa) * hr * 0.55, a.y + Math.sin(fa) * hr * 0.55]], { seed: sd + 137 + k * 3 + j, width: 0.7 });
      }
    });
    // ears (standing on the outline), in canvas px
    if (o.ears && d.ears && (kind === 'cat' || kind === 'bear')) d.ears.forEach((e, k) => {
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
    if (o.paws && d.paws && kind === 'bear' && !d.arms) d.paws.forEach((p, k) => {
      const s = U * 0.24, dir = k ? 1 : -1, c = [p.x - dir * s * 0.3, p.y - s * 0.7];
      pen(ellPts(c[0], c[1], s, s * 0.7, -0.15, Math.PI + 0.15, 18), { seed: sd + 41 + k });
      for (let j = -1; j <= 1; j += 2) pen([[c[0] + j * s * 0.33, c[1] + s * 0.68], [c[0] + j * s * 0.3, c[1] + s * 0.38]], { seed: sd + 51 + k * 3 + j, width: 0.7 });
    });

    // fish / dolphin: a tail at the far end of the cloud (d.tail: {x, y, dx, dy} canvas px, pointing outward)
    if ((kind === 'fish' || kind === 'dolphin') && d.tail) {
      const t = d.tail, px = -t.dy, py = t.dx, s = U * (kind === 'fish' ? 0.62 : 0.7), bx = t.x, by = t.y;
      const P = (a, b) => [bx + t.dx * s * a + px * s * b, by + t.dy * s * a + py * s * b];
      if (kind === 'fish') {   // a forked fin with a few rays
        pen([...quad(P(0, 0.12), P(0.45, 0.45), P(0.95, 0.72), 8), ...quad(P(0.95, 0.72), P(0.62, 0.1), P(0.58, 0), 6).slice(1),
             ...quad(P(0.58, 0), P(0.62, -0.1), P(0.95, -0.72), 6).slice(1), ...quad(P(0.95, -0.72), P(0.45, -0.45), P(0, -0.12), 8).slice(1)],
          { closed: true, fill: 'rgba(255,255,255,0.6)', seed: sd + 221 });
        for (const j of [-0.4, 0, 0.4]) pen([P(0.15, j * 0.25), P(0.62, j * 1.1)], { seed: sd + 225 + Math.round(j * 10), width: 0.55 });
      } else {                 // dolphin flukes: two lobes sideways
        pen([...quad(P(0, 0.08), P(0.25, 0.25), P(0.55, 0.85), 8), ...quad(P(0.55, 0.85), P(0.62, 0.35), P(0.42, 0), 8).slice(1),
             ...quad(P(0.42, 0), P(0.62, -0.35), P(0.55, -0.85), 8).slice(1), ...quad(P(0.55, -0.85), P(0.25, -0.25), P(0, -0.08), 8).slice(1)],
          { closed: true, fill: 'rgba(255,255,255,0.6)', seed: sd + 221 });
      }
    }
    // bat: wings out to the two ends of the cloud (the bird has none: the cloud outline shows it) (d.wings: [{x, y}] canvas px, left then right)
    if (kind === 'bat' && d.wings) d.wings.forEach((w, k) => {
      const s = k ? 1 : -1, sh = [d.x + s * U * 0.72, d.y - U * 0.05], low = [d.x + s * U * 0.62, d.y + U * 0.42];
      let tx = w.x, ty = w.y; const reach = Math.hypot(tx - d.x, ty - d.y);
      if (reach > U * 3.2) { const f = U * 3.2 / reach; tx = d.x + (tx - d.x) * f; ty = d.y + (ty - d.y) * f; }
      const tip = [tx, ty - U * (kind === 'bat' ? 0.35 : 0.2)];
      const top = quad(sh, [(sh[0] + tip[0]) / 2, Math.min(sh[1], tip[1]) - U * 0.45], tip, 12);
      // the lower edge back to the body: feathers (bird) or scallops between the fingers (bat)
      const back = [], N = 3;
      let p0 = tip;
      for (let j = 1; j <= N; j++) {
        const a = j / N, p1 = [tip[0] + (low[0] - tip[0]) * a, tip[1] + (low[1] - tip[1]) * a + (kind === 'bat' ? U * 0.12 : U * 0.05)];
        const mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2], bulge = kind === 'bat' ? -U * 0.22 : U * 0.2;
        back.push(...quad(p0, [mid[0], mid[1] + bulge], p1, 6).slice(1));
        p0 = p1;
      }
      pen([...top, ...back], { closed: true, fill: kind === 'bat' ? 'rgba(70,60,95,0.28)' : 'rgba(255,255,255,0.55)', seed: sd + 241 + k });
      if (kind === 'bat') for (let j = 1; j < N; j++) {   // the finger bones
        const a = j / N; pen([sh, [tip[0] + (low[0] - tip[0]) * a, tip[1] + (low[1] - tip[1]) * a + U * 0.12]], { seed: sd + 245 + k * 3 + j, width: 0.6 });
      }
    });
    // bird: little orange feet on the bottom outline
    if (kind === 'bird' && d.paws) d.paws.forEach((p, k) => {
      const s = U * 0.16, c = o.beak || '#f0a030';
      pen([[p.x, p.y - s * 0.2], [p.x, p.y + s * 1.2]], { seed: sd + 251 + k, color: c, width: 1.1 });
      for (const j of [-1, 0, 1]) pen([[p.x, p.y + s * 1.2], [p.x + j * s * 0.6, p.y + s * 1.7]], { seed: sd + 253 + k * 3 + j, color: c, width: 1.0 });
    });
    // sheep: four thin legs under the wool (two pairs on the bottom outline)
    if (kind === 'sheep' && d.paws) d.paws.forEach((p, k) => {
      const s = U * 0.3;
      for (const j of [-1, 1]) {
        const x = p.x + j * s * 0.22, y0 = p.y - s * 0.15, y1 = p.y + s * 0.75;
        pen([[x, y0], [x + j * s * 0.04, y1]], { seed: sd + 171 + k * 3 + j, width: 1.1 });
        pen(ellPts(x + j * s * 0.04, y1, s * 0.13, s * 0.08, 0, Math.PI * 2, 10), { closed: true, fill: o.ink, seed: sd + 175 + k * 3 + j });
      }
    });

    // the face itself, in face units around (x, y), turned by rot
    ctx.translate(d.x, d.y); ctx.rotate(d.rot || 0);
    const L = (x, y) => [x * U, y * U];
    if (kind === 'sheep') {
      // floppy ears, then the face (cream, so it shows on the white wool)
      for (const s of [-1, 1]) {
        const ex = s * 0.5, ey = -0.12;
        pen(ellPts(ex * U, ey * U, 0.24 * U, 0.1 * U, 0, Math.PI * 2, 16).map(([x, y]) => {
          const a = s * 0.45, cx = ex * U, cy = ey * U, dx = x - cx, dy = y - cy;
          return [cx + dx * Math.cos(a) - dy * Math.sin(a), cy + dx * Math.sin(a) + dy * Math.cos(a)]; }),
          { closed: true, fill: '#f3e6d3', seed: sd + 181 + s });
      }
      pen(ellPts(0, 0.08 * U, 0.4 * U, 0.52 * U, 0, Math.PI * 2, 30), { closed: true, fill: '#f6ecdd', seed: sd + 185 });
    } else if (kind === 'whale') {
      // a round whale seen from the front-ish: body, tail flukes up on the right, a spout on top, a belly line
      pen(ellPts(0, 0.12 * U, 1.7 * U, 1.1 * U, 0, Math.PI * 2, 40), { closed: true, seed: sd + 191 });
      pen([...quad(L(1.5, -0.25), L(1.75, -0.75), L(2.2, -0.85), 10), ...quad(L(2.2, -0.85), L(1.95, -0.45), L(2.35, -0.2), 8).slice(1),
           ...quad(L(2.35, -0.2), L(1.95, -0.1), L(1.62, 0.2), 8).slice(1)], { seed: sd + 193 });
      pen(quad(L(-1.25, 0.6), L(0, 1.05), L(1.25, 0.6), 16), { seed: sd + 195, width: 0.7 });
      pen(quad(L(0, -0.98), L(0.05, -1.25), L(0, -1.5), 8), { seed: sd + 197, color: o.spout || '#5aa0e0' });
      for (const s of [-1, 1]) pen(quad(L(0, -1.45), L(s * 0.25, -1.65), L(s * 0.42, -1.4), 8), { seed: sd + 199 + s, color: o.spout || '#5aa0e0' });
    }
    if (kind === 'bat') for (const s of [-1, 1]) {   // tall pointed ears
      pen([L(s * 0.2, -0.55), L(s * 0.5, -1.25), L(s * 0.62, -0.42)], { seed: sd + 261 + s });
      pen([L(s * 0.32, -0.6), L(s * 0.49, -1.0), L(s * 0.53, -0.55)], { seed: sd + 263 + s, width: 0.6 });
    }
    if (kind === 'bird') for (const j of [-1, 0, 1]) pen(quad(L(j * 0.08, -0.62), L(j * 0.2 + 0.05, -0.95), L(j * 0.3 + 0.12, -0.85), 8), { seed: sd + 265 + j, width: 0.8 });   // a tuft
    const eSc = { sheep: 0.72, peek: 1.25, whale: 0.85, cat: 1.08, fish: 1.12, bat: 1.05, snake: 1.35 }[kind] || 1;
    const eR = 0.155 * o.eyeSize * eSc, eX = (kind === 'sheep' ? 0.2 : kind === 'peek' ? 0.32 : 0.36) * o.eyeGap, eY = kind === 'sheep' ? 0.02 : 0.0;
    const pupilInk = o.pupilInk || o.ink;
    let lidU = 0, lidD = 0;
    if (expr === 'sleepy') lidU = 0.55;
    if (expr === 'smug') { lidU = 0.45; lidD = 0.15; }
    if (d.blinkT >= 0) { const b = Math.sin(Math.min(1, d.blinkT / 0.16) * Math.PI); lidU += (1 - lidU) * b; lidD += (1 - lidD) * b; }
    // fish: one eye (seen from the side, away from the tail); peek: one big fierce eye; snake: two sharp ones
    const ts0 = d.tail ? (d.tail.x >= d.x ? 1 : -1) : 1;
    const eyes = kind === 'fish' ? [-ts0 * 0.32] : kind === 'peek' ? [0] : [-1, 1];
    for (const s of eyes) {
      const ex0 = kind === 'fish' ? s : s * eX, er0 = (kind === 'peek' ? 0.42 : kind === 'fish' ? eR * 1.15 : eR) * U;
      const [ex, ey] = L(ex0, kind === 'peek' ? -0.02 : eY), er = er0;
      if (kind === 'snake' || kind === 'peek') { sharpEye(ex, ey, er, kind === 'snake' ? (s || 1) : 0, kind, d, sd + 61 + s * 7); continue; }
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
      if (kind !== 'peek' && kind !== 'fish') for (const s of [-1, 1]) { const [bx, by] = L(s * (eX + 0.1), 0.24); ctx.beginPath(); ctx.ellipse(bx, by, U * 0.11, U * 0.06, 0, 0, Math.PI * 2); ctx.fill(); }
    }
    const nY = 0.2;
    if (kind === 'sheep') {
      mouthW(L, U, 0.3, 0.07, expr, sd);
    } else if (kind === 'whale') {
      pen(quad(L(-0.45, 0.32), L(0, 0.58), L(0.45, 0.32), 14), { seed: sd + 201 });
    } else if (kind === 'fish' || kind === 'dolphin') {
      // which side the tail is on (the gill and scales go on that side, the snout on the other)
      const ts = d.tail ? (d.tail.x >= d.x ? 1 : -1) : 1;
      if (kind === 'fish') {
        const [mx, my] = L(-ts * 0.62, 0.28);
        pen(ellPts(mx, my, 0.07 * U, 0.06 * U, 0, Math.PI * 2, 14), { closed: true, fill: '#e8798f', seed: sd + 231 });
        pen(ellPts(ts * 0.15 * U, 0.1 * U, 0.2 * U, 0.4 * U, ts > 0 ? -Math.PI * 0.4 : Math.PI * 0.6, ts > 0 ? Math.PI * 0.4 : Math.PI * 1.4, 12), { seed: sd + 233, width: 0.85 });   // gill
        [[0.55, -0.15], [0.55, 0.25], [0.8, 0.05]].forEach(([x, y], k) =>
          pen(ellPts(ts * x * U, y * U, 0.12 * U, 0.1 * U, 0.15 * Math.PI, 0.85 * Math.PI, 8), { seed: sd + 235 + k, width: 0.6 }));   // scales
        pen([...quad(L(-0.05, -0.55), L(0.15 * ts, -0.95), L(0.45 * ts, -0.75), 8), ...quad(L(0.45 * ts, -0.75), L(0.3 * ts, -0.62), L(0.3 * ts, -0.5), 6).slice(1)], { seed: sd + 239, width: 0.85 });   // top fin
      } else {
        // dolphin: a snout pointing away from the tail, a curved fin on top, a flipper, a blowhole
        const sn = -ts;
        pen([...quad(L(sn * 0.55, 0.12), L(sn * 1.0, 0.12), L(sn * 1.12, 0.32), 10), ...quad(L(sn * 1.12, 0.32), L(sn * 0.95, 0.5), L(sn * 0.55, 0.46), 10).slice(1)], { seed: sd + 271 });
        pen(quad(L(sn * 0.2, 0.38), L(sn * 0.7, 0.42), L(sn * 1.08, 0.33), 10), { seed: sd + 273, width: 0.8 });
        pen([...quad(L(-0.15 * ts, -0.62), L(0.05 * ts, -1.15), L(0.45 * ts, -1.2), 10), ...quad(L(0.45 * ts, -1.2), L(0.25 * ts, -0.9), L(0.3 * ts, -0.6), 8).slice(1)], { seed: sd + 275 });
        pen([...quad(L(0.2 * ts, 0.55), L(0.35 * ts, 0.85), L(0.55 * ts, 0.9), 8), ...quad(L(0.55 * ts, 0.9), L(0.45 * ts, 0.65), L(0.42 * ts, 0.5), 6).slice(1)], { seed: sd + 277, width: 0.85 });
        const [bx, by] = L(0, -0.5); ctx.fillStyle = o.ink; ctx.beginPath(); ctx.ellipse(bx, by, U * 0.05, U * 0.025, 0, 0, Math.PI * 2); ctx.fill();
      }
    } else if (kind === 'bird') {
      const [bx, by] = L(0, 0.2), w = 0.11 * U, c = o.beak || '#f0a030';
      pen([[bx - w, by - w * 0.3], [bx + w, by - w * 0.3], [bx, by + w * 1.1]], { closed: true, fill: c, seed: sd + 281 });
      pen([[bx - w * 0.9, by + w * 0.1], [bx + w * 0.9, by + w * 0.1]], { seed: sd + 283, width: 0.6 });
    } else if (kind === 'bat') {
      const [nx, ny] = L(0, 0.18), w = 0.05 * U;
      pen([[nx - w, ny], [nx, ny + w], [nx + w, ny]], { seed: sd + 291, width: 0.9 });
      pen(quad(L(-0.2, 0.32), L(0, 0.46), L(0.2, 0.32), 10), { seed: sd + 293 });
      for (const s of [-1, 1]) pen([L(s * 0.1, 0.37), L(s * 0.075, 0.47), L(s * 0.05, 0.385)], { closed: true, fill: '#ffffff', seed: sd + 295 + s, width: 0.7 });   // fangs
    } else if (kind === 'peek') {
      // just the one big eye
    } else if (kind === 'snake') {
      // two nostrils, a smile and a forked tongue (the cloud outline is the head)
      for (const s of [-1, 1]) { const [nx, ny] = L(s * 0.09, nY); ctx.fillStyle = o.ink; ctx.beginPath(); ctx.arc(nx, ny, U * 0.025, 0, Math.PI * 2); ctx.fill(); }
      mouthPlain(L, U, nY + 0.1, expr, sd);
      const tc = o.tongue || '#d9435e';
      pen([L(0, nY + 0.33), L(0, nY + 0.5)], { seed: sd + 143, color: tc, width: 1.1 });
      pen([L(0, nY + 0.5), L(-0.06, nY + 0.6)], { seed: sd + 145, color: tc, width: 1.1 });
      pen([L(0, nY + 0.5), L(0.06, nY + 0.6)], { seed: sd + 147, color: tc, width: 1.1 });
    } else if (kind === 'cat') {
      const [nx, ny] = L(0, nY), w = 0.075 * U;
      pen([[nx - w, ny - w * 0.5], [nx + w, ny - w * 0.5], [nx, ny + w * 0.6]], { closed: true, fill: o.ink, seed: sd + 101 });
      mouthW(L, U, nY + 0.05, 0.11, expr, sd);
    } else if (kind === 'bear') {
      pen(ellPts(0, (nY + 0.09) * U, 0.27 * U, 0.19 * U, 0, Math.PI * 2, 24), { closed: true, fill: 'rgba(255,248,236,0.85)', seed: sd + 99, width: 0.8 });   // muzzle
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
  // the snake's body: two pen lines along the centre line, thinner toward the tail, with a few belly stripes;
  // it starts just outside the head
  function snakeBody(body, hx, hy, U, sd) {
    let i0 = 0; while (i0 < body.length - 1 && Math.hypot(body[i0].x - hx, body[i0].y - hy) < U * 0.75) i0++;
    const b = body.slice(Math.max(0, i0 - 1));
    if (b.length < 3) return;
    const n = b.length, L = [], R = [];
    for (let i = 0; i < n; i++) {
      const p = b[Math.max(0, i - 1)], q = b[Math.min(n - 1, i + 1)];
      let tx = q.x - p.x, ty = q.y - p.y; const l = Math.hypot(tx, ty) || 1; tx /= l; ty /= l;
      const t = i / (n - 1), taper = t < 0.6 ? 1 : Math.max(0.05, 1 - (t - 0.6) / 0.4);
      const hw = Math.max(1, Math.min(b[i].half * 0.7, U * 0.5) * taper);
      L.push([b[i].x - ty * hw, b[i].y + tx * hw]); R.push([b[i].x + ty * hw, b[i].y - tx * hw]);
    }
    const tip = [b[n - 1].x, b[n - 1].y];
    pen([...L, tip, ...R.reverse()], { seed: sd + 151 });
    R.reverse();
    // stripes across the body
    let run = 0, next = U * 0.9;
    for (let i = 1; i < n * 0.85; i++) {
      run += Math.hypot(b[i].x - b[i - 1].x, b[i].y - b[i - 1].y);
      if (run < next) continue;
      next = run + U * 0.9;
      const m = [(L[i][0] + R[i][0]) / 2, (L[i][1] + R[i][1]) / 2];
      pen(quad(L[i], [m[0] + (b[i].x - b[i - 1].x) * 0.8, m[1] + (b[i].y - b[i - 1].y) * 0.8], R[i], 8), { seed: sd + 161 + i, width: 0.6 });
    }
  }
  // an almond eye: the snake's (narrow, tilted up at the outer corner, a slit pupil) and the peeking one's
  // (big, a heavy frowning brow, an amber iris that looks around)
  function sharpEye(ex, ey, er, side, kind, d, seed) {
    const snake = kind === 'snake', tilt = snake ? -side * 0.32 : 0, w = er * (snake ? 1.25 : 1.3), hgt = er * (snake ? 0.62 : 0.78);
    let open = 1;
    if (d.blinkT >= 0) open = 1 - Math.sin(Math.min(1, d.blinkT / 0.16) * Math.PI);
    if (d.expr === 'sleepy') open *= 0.55;
    const ca = Math.cos(tilt), sa = Math.sin(tilt), R = (x, y) => [ex + x * ca - y * sa, ey + x * sa + y * ca];
    const poly = [];
    for (let i = 0; i <= 16; i++) { const t = i / 16; poly.push(R((t * 2 - 1) * w, -Math.sin(t * Math.PI) * hgt * open * (snake ? 1 : 1.05))); }
    for (let i = 15; i >= 1; i--) { const t = i / 16; poly.push(R((t * 2 - 1) * w, Math.sin(t * Math.PI) * hgt * open * (snake ? 1 : 0.85))); }
    if (open < 0.15) { pen([R(-w, 0), R(w, 0)], { seed, width: 1.2 }); }
    else {
      pen(poly, { closed: true, fill: snake ? '#fff4b8' : '#ffffff', noStroke: true, seed });
      ctx.save(); ctx.beginPath(); poly.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.clip();
      const gx = (d.gx || 0) * w * 0.35, gy = (d.gy || 0) * hgt * 0.3;
      if (snake) {
        ctx.fillStyle = O.ink; ctx.beginPath(); ctx.ellipse(ex + gx, ey + gy, er * 0.17, hgt * 1.0, tilt, 0, Math.PI * 2); ctx.fill();
      } else {
        const ir = hgt * 0.95;
        ctx.fillStyle = '#e09a28'; ctx.beginPath(); ctx.arc(ex + gx, ey + gy, ir, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = O.ink; ctx.beginPath(); ctx.arc(ex + gx, ey + gy, ir * 0.45, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = 'rgba(120,60,10,0.6)'; ctx.lineWidth = Math.max(1, LW * 0.5);
        for (let k = 0; k < 10; k++) { const a = k / 10 * Math.PI * 2; ctx.beginPath(); ctx.moveTo(ex + gx + Math.cos(a) * ir * 0.5, ey + gy + Math.sin(a) * ir * 0.5); ctx.lineTo(ex + gx + Math.cos(a) * ir * 0.92, ey + gy + Math.sin(a) * ir * 0.92); ctx.stroke(); }
      }
      ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(ex + gx - er * 0.18, ey + gy - hgt * 0.35, er * (snake ? 0.09 : 0.13), 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      pen(poly, { closed: true, seed: seed + 1, width: snake ? 1.1 : 1.4 });
    }
    if (!snake) {   // a heavy frowning brow, and a crease under the eye
      pen([[ex - w * 1.12, ey - hgt * 1.75], [ex - w * 0.2, ey - hgt * 1.25], [ex + w * 0.35, ey - hgt * 1.05], [ex + w * 1.1, ey - hgt * 1.4]], { seed: seed + 2, width: 2.2 });
      pen(quad([ex - w * 0.8, ey + hgt * 1.2], [ex, ey + hgt * 1.55], [ex + w * 0.8, ey + hgt * 1.2], 10), { seed: seed + 3, width: 0.7 });
    } else {
      pen([R(w * 0.95 * side, -hgt * 0.2), R(w * 1.3 * side, -hgt * 0.65)], { seed: seed + 2, width: 0.8 });   // a sharp outer corner
    }
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
