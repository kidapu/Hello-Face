// SegNet: what is where in a picture, in the browser: our own COCO-Stuff 171-class segmenter (LR-ASPP
// MobileNetV3, trained in segtrain/; ~6 MB fp16). Runs with onnxruntime-web on WebGPU (else WASM).
// Nothing is uploaded; the runtime (CDN) and the model (models/ next to this script) are only fetched by load().
//   await SegNet.load(status)            -> 'webgpu' | 'wasm'
//   const r = await SegNet.run(canvas, W, H)
//       -> { label: Uint8Array(W*H) class 0..170, group: Uint8Array(W*H) (SegNet.G.*), names }
//   SegNet.G = { SKY, CLOUDS, GROUND, PLANT, BUILDING, PERSON, OTHER }
window.SegNet = (() => {
  const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.js';
  const BASE = new URL('.', (document.currentScript && document.currentScript.src) || location.href).href;
  const DIR = `${BASE}models/cocostuff-lraspp/`;
  const G = { SKY: 0, CLOUDS: 1, GROUND: 2, PLANT: 3, BUILDING: 4, PERSON: 5, OTHER: 6 };
  const GROUND = ['grass', 'dirt', 'sand', 'gravel', 'ground', 'mud', 'pavement', 'road', 'playingfield', 'railroad', 'snow', 'floor',
    'platform', 'rug', 'carpet', 'moss', 'hill', 'mountain', 'rock', 'stone', 'sea', 'river', 'water', 'leaves', 'waterdrops'];
  const PLANT = ['tree', 'bush', 'plant', 'flower', 'branch', 'straw', 'potted plant'];
  const BUILD = ['building', 'house', 'skyscraper', 'wall', 'roof', 'bridge', 'tent', 'fence', 'window', 'door', 'stairs', 'ceiling'];
  let sess = null, names = null, groupOf = null, loading = null, device = '';

  function script(src) {
    return new Promise((ok, bad) => { const s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = () => bad(new Error('runtime load failed')); document.head.appendChild(s); });
  }
  function load(status = () => {}) {
    if (loading) return loading;
    loading = (async () => {
      status('AI: ライブラリを読み込み中…');
      if (!window.ort) await script(ORT);
      let gpu = null;
      try { gpu = navigator.gpu ? await navigator.gpu.requestAdapter() : null; } catch (e) { gpu = null; }
      device = gpu ? 'webgpu' : 'wasm';
      const file = gpu && !gpu.features.has('shader-f16') ? 'model.onnx' : 'model_fp16.onnx';   // fp16 needs shader-f16 on the GPU
      status('AI: モデルを読み込み中…');
      names = await (await fetch(DIR + 'classes.json')).json();
      groupOf = new Uint8Array(names.length);
      names.forEach((n, i) => {
        const b = n.split('-')[0];
        groupOf[i] = n === 'sky-other' ? G.SKY : n === 'clouds' ? G.CLOUDS : GROUND.includes(b) ? G.GROUND : PLANT.includes(b) ? G.PLANT
          : BUILD.includes(b) ? G.BUILDING : n === 'person' ? G.PERSON : G.OTHER;
      });
      ort.env.wasm.numThreads = 1;
      try { sess = await ort.InferenceSession.create(DIR + file, { executionProviders: [device] }); }
      catch (e) { if (device !== 'webgpu') throw e; device = 'wasm'; sess = await ort.InferenceSession.create(DIR + 'model_fp16.onnx', { executionProviders: ['wasm'] }); }
      status(`AI: 準備完了 (${device === 'webgpu' ? 'GPU' : 'CPU'})`);
      return device;
    })();
    loading.catch(() => { loading = null; });
    return loading;
  }
  // the picture at long side 512 (multiples of 32), labels at 1/8, resampled (nearest) to the W x H grid
  async function run(src, W, H) {
    const sw = src.width, sh = src.height, k = 512 / Math.max(sw, sh);
    const w = Math.max(32, Math.round(sw * k / 32) * 32), h = Math.max(32, Math.round(sh * k / 32) * 32);
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(src, 0, 0, w, h);
    const d = g.getImageData(0, 0, w, h).data, x = new Float32Array(3 * w * h), n = w * h;
    for (let i = 0; i < n; i++) { x[i] = d[i * 4]; x[n + i] = d[i * 4 + 1]; x[2 * n + i] = d[i * 4 + 2]; }
    const t0 = performance.now();
    const r = await sess.run({ image: new ort.Tensor('float32', x, [1, 3, h, w]) });
    const L = r.label.data, h8 = r.label.dims[1], w8 = r.label.dims[2];
    const label = new Uint8Array(W * H), group = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) {
      const yy = Math.min(h8 - 1, Math.floor((y + 0.5) / H * h8));
      for (let xx = 0; xx < W; xx++) {
        const v = L[yy * w8 + Math.min(w8 - 1, Math.floor((xx + 0.5) / W * w8))];
        label[y * W + xx] = v; group[y * W + xx] = groupOf[v];
      }
    }
    return { label, group, names, ms: performance.now() - t0 };
  }
  return { load, run, G, get ready() { return !!sess; }, get device() { return device; }, get names() { return names; } };
})();
