// SamFind: "tap a thing -> its outline" with Segment Anything (SlimSAM), running in the browser
// (Transformers.js + ONNX Runtime Web: WebGPU when there is one, else WASM on the CPU), in a worker
// (samfind_worker.js) so the page keeps drawing while it works.
// Nothing is uploaded; the model files (~21 MB on a GPU with fp16, ~14 MB lite / CPU, plus ~3 MB runtime)
// are downloaded once and then cached by the browser.
//
//   await SamFind.load(status)            status(text) is called with progress messages
//   await SamFind.embed(image)            once per photo (the heavy part, ~0.3-3 s on a PC, longer on a phone)
//   const m = await SamFind.segment(u, v, w, h, level)
//       u, v: the tapped point (0..1 of the photo); w, h: the grid to return the mask on
//       level: 'auto' (best score, not the whole picture) | 0 (smallest of the 3) | 1 | 2 (biggest)
//       -> { mask: Uint8Array(w*h), area, score } | null
window.SamFind = (() => {
  const VER = '3.8.1', MODEL = 'Xenova/slimsam-77-uniform';
  const CDN = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${VER}`;
  // served from the same server when the files are there (models/, vendor/ next to this script), else from
  // the CDN / Hugging Face; ?samremote forces the remote files
  const SRC = (document.currentScript && document.currentScript.src) || location.href;
  const BASE = new URL('.', SRC).href, VQ = new URL(SRC).search;
  const LOCAL_LIB = `${BASE}vendor/transformers-${VER}/`, LOCAL_MODELS = `${BASE}models/`;
  // iPhone / iPad (also in an app's browser, e.g. Slack): the 8-bit model on the CPU (WebGPU there is new and the
  // memory is tight); ?samdevice=webgpu to try the GPU anyway
  const IOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  let local = false, worker = null, loading = null, device = '', ready = false, seq = 0, onStatus = () => {};
  const wait = new Map();

  function call(type, msg = {}, transfer = []) {
    return new Promise((res, rej) => {
      const id = ++seq; wait.set(id, { res, rej });
      worker.postMessage({ id, type, ...msg }, transfer);
    });
  }
  function start() {
    worker = new Worker(`${BASE}samfind_worker.js${VQ}`, { type: 'module' });
    worker.onmessage = ({ data: m }) => {
      if (m.type === 'status') { onStatus(m.text); return; }
      const w = wait.get(m.id); if (!w) return; wait.delete(m.id);
      m.error ? w.rej(new Error(m.error)) : w.res(m);
    };
    worker.onerror = (e) => {   // the worker died (out of memory, could not load ...): fail every waiting call
      const err = new Error(e.message || 'AI の処理が止まりました');
      for (const w of wait.values()) w.rej(err); wait.clear();
      worker = null; loading = null; ready = false;
    };
  }

  function load(status = () => {}) {
    onStatus = status;
    if (loading) return loading;
    loading = (async () => {
      status('AI: ライブラリを読み込み中…');
      const sp = new URLSearchParams(location.search);
      if (!sp.has('samremote')) {
        try { local = (await fetch(`${LOCAL_MODELS}${MODEL}/config.json`, { method: 'HEAD', cache: 'no-store' })).ok; } catch (e) { local = false; }
      }
      let gpu = null;
      try { gpu = navigator.gpu ? await navigator.gpu.requestAdapter() : null; } catch (e) { gpu = null; }
      // lite (?samlite, and on iPhone / iPad): 8-bit model on the CPU, ~14 MB instead of ~21 MB (fp16);
      // 8-bit weights give broken masks on WebGPU, so lite always runs on the CPU (a bit slower to analyse)
      const lite = sp.has('samlite') || (IOS && sp.get('samdevice') !== 'webgpu');
      if (lite || sp.get('samdevice') === 'wasm' || (gpu && !gpu.features.has('shader-f16'))) gpu = null;   // no fp16 -> lite
      device = gpu ? 'webgpu' : 'wasm';
      // ?samdtype=q8 / fp16 / fp32 / fp16,q8 (or {vision_encoder, prompt_encoder_mask_decoder} via window.SAM_DTYPE) to override
      let q = sp.get('samdtype');
      if (q && q.includes(',')) { const [e, d] = q.split(','); q = { vision_encoder: e, prompt_encoder_mask_decoder: d }; }
      const dtype = window.SAM_DTYPE || q || (gpu ? 'fp16' : 'q8');
      if (!worker) start();
      await call('load', { lib: local ? `${LOCAL_LIB}transformers.min.js` : CDN, local, models: LOCAL_MODELS, wasmPaths: LOCAL_LIB,
        model: MODEL, device, dtype });
      status(`AI: 準備完了 (${device === 'webgpu' ? 'GPU' : 'CPU'}, ${local ? 'このサーバー' : 'CDN'})`);
      return device;
    })();
    loading.catch(() => { loading = null; });
    return loading;
  }

  async function embed(image) {
    await load(onStatus);
    ready = false;
    // the model works at 1024 px on the long side; a smaller copy keeps the mask upsampling cheap
    const k = Math.min(1, 1024 / Math.max(image.width, image.height));
    const c = document.createElement('canvas');
    c.width = Math.round(image.width * k); c.height = Math.round(image.height * k);
    const x = c.getContext('2d'); x.drawImage(image, 0, 0, c.width, c.height);
    const d = x.getImageData(0, 0, c.width, c.height).data;
    await call('embed', { data: d.buffer, width: c.width, height: c.height }, [d.buffer]);
    ready = true;
  }

  async function segment(u, v, w, h, level = 'auto', maxFrac = 0.6) {
    if (!ready) return null;
    return (await call('segment', { u, v, w, h, level, maxFrac })).r;
  }

  return { load, embed, segment, get ready() { return ready; }, get device() { return device; }, get local() { return local; } };
})();
