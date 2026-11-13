// namefind_worker.js: the model side of namefind.js (MobileCLIP2-S0's image encoder + ~6400 Japanese words), in a worker.
//   in  { id, type: 'load', ort, dir }                     -> { id, ok, words }
//   in  { id, type: 'name', pixels: Float32Array(3*256*256) } -> { id, ok, top: [[ja, score], ...] }
let sess = null, W = null, T = null, HUB = null, dim = 0, scale = 1, hubScale = 1;
const BONUS = 0.02;   // the words for kids (park, town, home, toys: pri 1) win close calls

async function loadModel(m) {
  importScripts(m.ort);
  ort.env.wasm.wasmPaths = m.ort.replace(/[^/]+$/, '');
  ort.env.wasm.numThreads = 1;
  const get = async (f) => { const r = await fetch(m.dir + f); if (!r.ok) throw new Error(f + ': ' + r.status); return r; };
  const [wj, tb, hb] = await Promise.all([get('words.json').then((r) => r.json()), get('text.bin').then((r) => r.arrayBuffer()),
    get('hub.bin').then((r) => r.arrayBuffer())]);
  W = wj.words; dim = wj.dim; scale = wj.scale; hubScale = wj.hubScale; T = new Int8Array(tb); HUB = new Int8Array(hb);
  sess = await ort.InferenceSession.create(m.dir + 'image_fp16.onnx', { executionProviders: ['wasm'] });
  return { words: W.length };
}

async function nameIt(m) {   // (not "name": that is self.name in a worker, which the runtime reads)
  const t0 = performance.now();
  const r = await sess.run({ pixel: new ort.Tensor('float32', m.pixels, [1, 3, 256, 256]) });
  const e = r.embed.data, n = W.length, s = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    let d = 0; const o = k * dim;
    for (let j = 0; j < dim; j++) d += T[o + j] * e[j];
    s[k] = d * scale - HUB[k] * hubScale + BONUS * W[k][1];
  }
  const order = Array.from(s.keys()).sort((a, b) => s[b] - s[a]), top = [], seen = new Set();
  for (const k of order) { if (seen.has(W[k][0])) continue; seen.add(W[k][0]); top.push([W[k][0], +s[k].toFixed(3)]); if (top.length >= 5) break; }
  return { top, ms: Math.round(performance.now() - t0) };
}

const DO = { load: loadModel, name: nameIt };
onmessage = async ({ data: m }) => {
  try { postMessage({ id: m.id, ok: true, ...(await DO[m.type](m)) }); }
  catch (e) { console.error(e); postMessage({ id: m.id, error: String((e && e.message) || e) }); }
};
