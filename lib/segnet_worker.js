// SegNet worker: runs the segmenter off the page's main thread (so the camera, the drawing and the cloud detection
// don't hold it up). Messages:
//   in  { type: 'init', dir, ort, gpu }               -> out { type: 'ready', device, names, groupOf } | { type: 'error', message }
//   in  { type: 'run', id, bitmap, W, H }              -> out { type: 'result', id, label, group, ms, prep, post, w, h }
// `bitmap` is an ImageBitmap already at the model's input size (multiples of 32).
let sess = null, groupOf = null, device = '';
const G = { SKY: 0, CLOUDS: 1, GROUND: 2, PLANT: 3, BUILDING: 4, PERSON: 5, OTHER: 6 };
const GROUND = ['grass', 'dirt', 'sand', 'gravel', 'ground', 'mud', 'pavement', 'road', 'playingfield', 'railroad', 'snow', 'floor',
  'platform', 'rug', 'carpet', 'moss', 'hill', 'mountain', 'rock', 'stone', 'sea', 'river', 'water', 'leaves', 'waterdrops'];
const PLANT = ['tree', 'bush', 'plant', 'flower', 'branch', 'straw', 'potted plant'];
const BUILD = ['building', 'house', 'skyscraper', 'wall', 'roof', 'bridge', 'tent', 'fence', 'window', 'door', 'stairs', 'ceiling'];
let cv = null, cx = null;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      importScripts(m.ort);
      ort.env.wasm.wasmPaths = m.ort.replace(/[^/]+$/, '');   // in a worker the runtime can't tell where its own files are
      if (!self.navigator.gpu) throw new Error('no WebGPU in the worker');
      const gpu = await navigator.gpu.requestAdapter();
      if (!gpu) throw new Error('no GPU adapter in the worker');
      device = 'webgpu';
      const file = gpu.features.has('shader-f16') ? 'model_fp16.onnx' : 'model.onnx';
      const names = await (await fetch(m.dir + 'classes.json')).json();
      groupOf = new Uint8Array(names.length);
      names.forEach((n, i) => {
        const b = n.split('-')[0];
        groupOf[i] = n === 'sky-other' ? G.SKY : n === 'clouds' ? G.CLOUDS : GROUND.includes(b) ? G.GROUND : PLANT.includes(b) ? G.PLANT
          : BUILD.includes(b) ? G.BUILDING : n === 'person' ? G.PERSON : G.OTHER;
      });
      ort.env.wasm.numThreads = 1;
      sess = await ort.InferenceSession.create(m.dir + file, { executionProviders: ['webgpu'] });
      self.postMessage({ type: 'ready', device, names, groupOf });
    } else if (m.type === 'run') {
      const tA = performance.now(), { bitmap, W, H } = m, w = bitmap.width, h = bitmap.height, n = w * h;
      if (!cv || cv.width !== w || cv.height !== h) { cv = new OffscreenCanvas(w, h); cx = cv.getContext('2d', { willReadFrequently: true }); }
      cx.drawImage(bitmap, 0, 0); bitmap.close();
      const d = cx.getImageData(0, 0, w, h).data, x = new Float32Array(3 * n);
      for (let i = 0; i < n; i++) { x[i] = d[i * 4]; x[n + i] = d[i * 4 + 1]; x[2 * n + i] = d[i * 4 + 2]; }
      const t0 = performance.now();
      const r = await sess.run({ image: new ort.Tensor('float32', x, [1, 3, h, w]) });
      const t1 = performance.now();
      const L = r.label.data, h8 = r.label.dims[1], w8 = r.label.dims[2];
      const label = new Uint8Array(W * H), group = new Uint8Array(W * H);
      for (let y = 0; y < H; y++) {
        const yy = Math.min(h8 - 1, Math.floor((y + 0.5) / H * h8));
        for (let xx = 0; xx < W; xx++) {
          const v = L[yy * w8 + Math.min(w8 - 1, Math.floor((xx + 0.5) / W * w8))];
          label[y * W + xx] = v; group[y * W + xx] = groupOf[v];
        }
      }
      self.postMessage({ type: 'result', id: m.id, label, group, ms: t1 - t0, prep: t0 - tA, post: performance.now() - t1, w, h }, [label.buffer, group.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: m.type === 'init' ? 'error' : 'runError', id: m.id, message: String(err && err.message || err) });
  }
};
