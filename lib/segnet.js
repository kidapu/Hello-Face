// SegNet: what is where in a picture, in the browser: our own COCO-Stuff 171-class segmenter (LR-ASPP
// MobileNetV3, trained in segtrain/; ~6 MB fp16). Runs with onnxruntime-web on WebGPU (else WASM), in a worker
// thread when it can. Nothing is uploaded; the runtime (CDN) and the model (models/ next to this script) are only
// fetched by load().
//   await SegNet.load(status, opts)      opts: { worker: true, device: 'auto'|'webgpu'|'wasm', precision: 'auto'|'fp16'|'fp32',
//                                                up: 1|2|4|8 (labels at 1/8 .. 1/1 of the input: the scores are upsampled
//                                                smoothly before the class is picked, so the boundaries are round) }
//   const r = await SegNet.run(src, W, H, size)
//       -> { label: Uint8Array(W*H) class 0..170, group: Uint8Array(W*H) (SegNet.G.*), conf: Uint8Array(W*H) 0..255,
//            names, ms (inference), prep, post, w, h (model input), w8, h8 (model output) }
//   SegNet.reset()                       drop the loaded model (then load() again with other opts)
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
  let sess = null, names = null, groupOf = null, loading = null, device = '', precision = '', worker = null, wready = false, wid = 0, up = 1;
  const fileFor = (prec, u) => `model${u > 1 ? '_up' + u : ''}${prec === 'fp32' ? '' : '_fp16'}.onnx`;
  const pend = new Map();

  async function pickDevice(opts) {
    let gpu = null;
    if (opts.device !== 'wasm') { try { gpu = navigator.gpu ? await navigator.gpu.requestAdapter() : null; } catch (e) { gpu = null; } }
    const dev = gpu ? 'webgpu' : 'wasm';
    // fp16 on the GPU needs shader-f16; on the CPU fp16 works (converted inside)
    const prec = opts.precision && opts.precision !== 'auto' ? opts.precision : gpu && !gpu.features.has('shader-f16') ? 'fp32' : 'fp16';
    return { dev, prec, file: fileFor(prec, up) };
  }
  function startWorker(status, opts) {
    return new Promise((ok, bad) => {
      worker = new Worker(BASE + 'segnet_worker.js');
      const to = setTimeout(() => bad(new Error('worker timeout')), 90000);
      worker.onerror = (e) => { clearTimeout(to); bad(new Error(e.message || 'worker error')); };
      worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'ready') { clearTimeout(to); names = m.names; groupOf = m.groupOf; device = m.device; precision = m.precision; wready = true; status(`AI: 準備完了 (${device === 'webgpu' ? 'GPU' : 'CPU'}・別スレッド)`); ok(); }
        else if (m.type === 'error') { clearTimeout(to); bad(new Error(m.message)); }
        else if (m.type === 'result' || m.type === 'runError') { const p = pend.get(m.id); pend.delete(m.id); if (p) m.type === 'result' ? p.ok(m) : p.bad(new Error(m.message)); }
      };
      status('AI: モデルを読み込み中…');
      worker.postMessage({ type: 'init', dir: DIR, ort: ORT, device: opts.device || 'auto', precision: opts.precision || 'auto', up });
    });
  }
  function script(src) {
    return new Promise((ok, bad) => { const s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = () => bad(new Error('runtime load failed')); document.head.appendChild(s); });
  }
  function load(status = () => {}, opts = {}) {
    if (loading) return loading;
    up = [1, 2, 4, 8].includes(+opts.up) ? +opts.up : 1;
    loading = (async () => {
      status('AI: ライブラリを読み込み中…');
      // first choice: a worker thread (the page's own work then never holds the AI up); ?segworker=0 = main thread
      const useWorker = opts.worker !== false && new URLSearchParams(location.search).get('segworker') !== '0';
      if (useWorker && window.Worker && window.createImageBitmap) {
        try { await startWorker(status, opts); return device; } catch (e) { console.warn('SegNet worker failed, using the main thread:', e); if (worker) worker.terminate(); worker = null; }
      }
      if (!window.ort) await script(ORT);
      const pk = await pickDevice(opts);
      device = pk.dev; precision = pk.prec;
      status('AI: モデルを読み込み中…');
      names = await (await fetch(DIR + 'classes.json')).json();
      groupOf = new Uint8Array(names.length);
      names.forEach((n, i) => {
        const b = n.split('-')[0];
        groupOf[i] = n === 'sky-other' ? G.SKY : n === 'clouds' ? G.CLOUDS : GROUND.includes(b) ? G.GROUND : PLANT.includes(b) ? G.PLANT
          : BUILD.includes(b) ? G.BUILDING : n === 'person' ? G.PERSON : G.OTHER;
      });
      ort.env.wasm.numThreads = 1;
      try { sess = await ort.InferenceSession.create(DIR + pk.file, { executionProviders: [device] }); }
      catch (e) { if (device !== 'webgpu') throw e; device = 'wasm'; sess = await ort.InferenceSession.create(DIR + pk.file, { executionProviders: ['wasm'] }); }
      status(`AI: 準備完了 (${device === 'webgpu' ? 'GPU' : 'CPU'})`);
      return device;
    })();
    loading.catch(() => { loading = null; });
    return loading;
  }
  function reset() {
    for (const p of pend.values()) p.bad(new Error('reset'));
    pend.clear();
    if (worker) worker.terminate();
    if (sess && sess.release) sess.release().catch(() => {});
    worker = null; wready = false; sess = null; loading = null; device = ''; precision = '';
  }
  // the model input: the picture with its long side at `size` (multiples of 32); the labels come out at 1/8 and are
  // resampled (nearest) to the W x H grid (W, H = 0: the model's own output size)
  function inputSize(sw, sh, size) {
    const k = size / Math.max(sw, sh);
    return [Math.max(32, Math.round(sw * k / 32) * 32), Math.max(32, Math.round(sh * k / 32) * 32)];
  }
  async function run(src, W, H, size = 512) {
    const tA = performance.now();
    const sw = src.videoWidth || src.width, sh = src.videoHeight || src.height, [w, h] = inputSize(sw, sh, size);
    if (!W || !H) { W = w * up / 8; H = h * up / 8; }   // the model's own output size
    if (worker && wready) {
      const bitmap = await createImageBitmap(src, { resizeWidth: w, resizeHeight: h, resizeQuality: 'low' });
      const id = ++wid, tB = performance.now();
      const r = await new Promise((ok, bad) => { pend.set(id, { ok, bad }); worker.postMessage({ type: 'run', id, bitmap, W, H }, [bitmap]); });
      return { label: r.label, group: r.group, conf: r.conf, names, ms: r.ms, prep: (tB - tA) + r.prep, post: r.post, w, h, w8: w * up / 8, h8: h * up / 8, W, H };
    }
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(src, 0, 0, w, h);
    const d = g.getImageData(0, 0, w, h).data, x = new Float32Array(3 * w * h), n = w * h;
    for (let i = 0; i < n; i++) { x[i] = d[i * 4]; x[n + i] = d[i * 4 + 1]; x[2 * n + i] = d[i * 4 + 2]; }
    const t0 = performance.now();
    const r = await sess.run({ image: new ort.Tensor('float32', x, [1, 3, h, w]) });
    const t1 = performance.now();
    const L = r.label.data, Pr = r.prob.data, h8 = r.label.dims[1], w8 = r.label.dims[2];
    const label = new Uint8Array(W * H), group = new Uint8Array(W * H), conf = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) {
      const yy = Math.min(h8 - 1, Math.floor((y + 0.5) / H * h8));
      for (let xx = 0; xx < W; xx++) {
        const j = yy * w8 + Math.min(w8 - 1, Math.floor((xx + 0.5) / W * w8)), v = L[j];
        label[y * W + xx] = v; group[y * W + xx] = groupOf[v]; conf[y * W + xx] = Math.round(Pr[j] * 255);
      }
    }
    const t2 = performance.now();
    return { label, group, conf, names, ms: t1 - t0, prep: t0 - tA, post: t2 - t1, w, h, w8, h8, W, H };
  }
  return { load, run, reset, inputSize, G, get ready() { return !!sess || wready; }, get worker() { return wready; }, get device() { return device; },
    get precision() { return precision; }, get up() { return up; }, get names() { return names; }, get groupOf() { return groupOf; } };
})();
