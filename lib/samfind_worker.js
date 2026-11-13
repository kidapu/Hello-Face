// samfind_worker.js: the Segment Anything side of samfind.js, in a module worker so the page keeps drawing
// (and answering taps) while the model loads and analyses a photo (seconds on a phone's CPU).
// Messages in: { id, type: 'load', lib, local, models, model, device, dtype } / { id, type: 'embed', data, width, height }
//              / { id, type: 'segment', u, v, w, h, level, maxFrac }
// Messages out: { id, ok, ... } or { id, error }; { type: 'status', text } while loading.
let T = null, model = null, processor = null, emb = null, inputs = null;
const status = (text) => postMessage({ type: 'status', text });

async function load(m) {
  T = await import(m.lib);
  if (m.local) {
    T.env.allowRemoteModels = false; T.env.allowLocalModels = true; T.env.localModelPath = m.models;
    T.env.backends.onnx.wasm.wasmPaths = m.wasmPaths;
  }
  const files = {};
  const progress_callback = (p) => {
    if (p.status !== 'progress' || !p.total) return;
    files[p.file] = [p.loaded, p.total];
    let a = 0, b = 0; for (const [l, t] of Object.values(files)) { a += l; b += t; }
    status(`AI: モデルをダウンロード中… ${(a / 1e6).toFixed(1)} / ${(b / 1e6).toFixed(1)} MB`);
  };
  status('AI: モデルを準備中…');
  model = await T.SamModel.from_pretrained(m.model, { device: m.device, dtype: m.dtype, progress_callback });
  processor = await T.AutoProcessor.from_pretrained(m.model);
  return {};
}

async function embed(m) {
  const raw = new T.RawImage(new Uint8ClampedArray(m.data), m.width, m.height, 4).rgb();
  inputs = await processor(raw);
  emb = await model.get_image_embeddings(inputs);
  return {};
}

async function segment({ u, v, w, h, level, maxFrac }) {
  if (!emb) return { r: null };
  const [rh, rw] = inputs.reshaped_input_sizes[0];
  const input_points = new T.Tensor('float32', [u * rw, v * rh], [1, 1, 1, 2]);
  const input_labels = new T.Tensor('int64', [1n], [1, 1, 1]);
  const out = await model({ ...emb, input_points, input_labels });
  const masks = await processor.post_process_masks(out.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes);
  const mt = masks[0], [, nm, H, W] = mt.dims, sc = out.iou_scores.data, md = mt.data;
  // SAM gives 3 masks (a part, a bigger part, the whole thing) in no fixed order: sample each on the grid
  const cand = [];
  for (let k = 0; k < nm; k++) {
    const off = k * H * W, mask = new Uint8Array(w * h);
    let area = 0;
    for (let y = 0; y < h; y++) {
      const Y = Math.min(H - 1, Math.floor((y + 0.5) * H / h));
      for (let x = 0; x < w; x++) {
        const X = Math.min(W - 1, Math.floor((x + 0.5) * W / w));
        if (md[off + Y * W + X]) { mask[y * w + x] = 1; area++; }
      }
    }
    area = keepPiece(mask, w, h, Math.min(w - 1, Math.floor(u * w)), Math.min(h - 1, Math.floor(v * h)), area);
    cand.push({ mask, area, score: sc[k] });
  }
  cand.sort((a, b) => a.area - b.area);
  // auto: the best score among the ones that are not (almost) the whole picture
  let pick;
  if (level === 'auto') {
    const ok = cand.filter((c) => c.area > 12 && c.area < w * h * maxFrac);
    pick = (ok.length ? ok : cand).reduce((m, c) => (c.score > m.score ? c : m));
  } else pick = cand[Math.min(nm - 1, Math.max(0, +level))];
  return { r: pick, transfer: [pick.mask.buffer] };
}

// SAM's mask is not one piece: it also takes other places that look alike (tap a dark shirt -> the dark hair too).
// Keep only the piece under the tap (8-connected; the biggest piece when the tap itself fell just outside).
function keepPiece(mask, w, h, tx, ty, area) {
  if (!area) return 0;
  const lab = new Int32Array(w * h), st = new Int32Array(w * h);
  let n = 0, best = 0, bestA = 0;
  for (let s = 0; s < w * h; s++) {
    if (!mask[s] || lab[s]) continue;
    lab[s] = ++n; let top = 0, a = 0; st[top++] = s;
    while (top) {
      const i = st[--top], x = i % w, y = (i - x) / w; a++;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const X = x + dx, Y = y + dy, j = Y * w + X;
        if (X >= 0 && Y >= 0 && X < w && Y < h && mask[j] && !lab[j]) { lab[j] = n; st[top++] = j; }
      }
    }
    if (a > bestA) { bestA = a; best = n; }
  }
  if (n <= 1) return area;
  const keep = lab[ty * w + tx] || best;
  let a = 0;
  for (let i = 0; i < w * h; i++) { if (lab[i] === keep) a++; else mask[i] = 0; }
  return a;
}

const DO = { load, embed, segment };
onmessage = async ({ data: m }) => {
  try {
    const { transfer, ...res } = await DO[m.type](m);
    postMessage({ id: m.id, ok: true, ...res }, transfer || []);
  } catch (e) {
    console.error(e);
    postMessage({ id: m.id, error: String((e && e.message) || e) });
  }
};
