// SamFind: "tap a thing -> its outline" with Segment Anything (SlimSAM), running in the browser
// (Transformers.js + ONNX Runtime Web: WebGPU when there is one, else WASM on the CPU).
// Nothing is uploaded; the model files (~21 MB on a GPU with fp16, ~14 MB lite / CPU, plus ~3 MB runtime)
// are downloaded once and then cached by the browser.
//
//   await SamFind.load(status)            status(text) is called with progress messages
//   await SamFind.embed(image)            once per photo (the heavy part, ~0.3-3 s)
//   const m = await SamFind.segment(u, v, w, h, level)
//       u, v: the tapped point (0..1 of the photo); w, h: the grid to return the mask on
//       level: 'auto' (best score, not the whole picture) | 0 (smallest of the 3) | 1 | 2 (biggest)
//       -> { mask: Uint8Array(w*h), area, score } | null
window.SamFind = (() => {
  const VER = '3.8.1', MODEL = 'Xenova/slimsam-77-uniform';
  const CDN = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${VER}`;
  // served from the same server when the files are there (models/, vendor/ next to this script), else from
  // the CDN / Hugging Face; ?samremote forces the remote files
  const BASE = new URL('.', (document.currentScript && document.currentScript.src) || location.href).href;
  const LOCAL_LIB = `${BASE}vendor/transformers-${VER}/`, LOCAL_MODELS = `${BASE}models/`;
  let local = false;
  let T = null, model = null, processor = null, loading = null, emb = null, inputs = null, device = '';

  function load(status = () => {}) {
    if (loading) return loading;
    loading = (async () => {
      status('AI: ライブラリを読み込み中…');
      if (!new URLSearchParams(location.search).has('samremote')) {
        try { local = (await fetch(`${LOCAL_MODELS}${MODEL}/config.json`, { method: 'HEAD', cache: 'no-store' })).ok; } catch (e) { local = false; }
      }
      T = await import(local ? `${LOCAL_LIB}transformers.min.js` : CDN);
      if (local) {
        T.env.allowRemoteModels = false; T.env.allowLocalModels = true; T.env.localModelPath = LOCAL_MODELS;
        T.env.backends.onnx.wasm.wasmPaths = LOCAL_LIB;
      }
      let gpu = null;
      try { gpu = navigator.gpu ? await navigator.gpu.requestAdapter() : null; } catch (e) { gpu = null; }
      // lite (?samlite): 8-bit model on the CPU, ~14 MB instead of ~21 MB (fp16) / ~40 MB (fp32);
      // 8-bit weights give broken masks on WebGPU, so lite always runs on the CPU (a bit slower to analyse)
      const sp = new URLSearchParams(location.search), lite = sp.has('samlite');
      if (lite || sp.get('samdevice') === 'wasm' || (gpu && !gpu.features.has('shader-f16'))) gpu = null;   // no fp16 -> lite
      device = gpu ? 'webgpu' : 'wasm';
      // ?samdtype=q8 / fp16 / fp32 / fp16,q8 (or {vision_encoder, prompt_encoder_mask_decoder} via window.SAM_DTYPE) to override
      let q = new URLSearchParams(location.search).get('samdtype');   // 'q8' or 'fp16,q8' (encoder, decoder)
      if (q && q.includes(',')) { const [e, d] = q.split(','); q = { vision_encoder: e, prompt_encoder_mask_decoder: d }; }
      const dtype = window.SAM_DTYPE || q || (gpu ? 'fp16' : 'q8');
      const files = {};
      const progress_callback = (p) => {
        if (p.status !== 'progress' || !p.total) return;
        files[p.file] = [p.loaded, p.total];
        let a = 0, b = 0; for (const [l, t] of Object.values(files)) { a += l; b += t; }
        status(`AI: モデルをダウンロード中… ${(a / 1e6).toFixed(1)} / ${(b / 1e6).toFixed(1)} MB`);
      };
      status('AI: モデルを準備中…');
      model = await T.SamModel.from_pretrained(MODEL, { device, dtype, progress_callback });
      processor = await T.AutoProcessor.from_pretrained(MODEL);
      status(`AI: 準備完了 (${device === 'webgpu' ? 'GPU' : 'CPU'}, ${local ? 'このサーバー' : 'CDN'})`);
      return device;
    })();
    loading.catch(() => { loading = null; });
    return loading;
  }

  async function embed(image) {
    await load();
    // the model works at 1024 px on the long side; a smaller copy keeps the mask upsampling cheap
    const k = Math.min(1, 1024 / Math.max(image.width, image.height));
    const c = document.createElement('canvas');
    c.width = Math.round(image.width * k); c.height = Math.round(image.height * k);
    c.getContext('2d').drawImage(image, 0, 0, c.width, c.height);
    const raw = T.RawImage.fromCanvas(c);
    inputs = await processor(raw);
    emb = await model.get_image_embeddings(inputs);
  }

  async function segment(u, v, w, h, level = 'auto', maxFrac = 0.6) {
    if (!emb) return null;
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
      cand.push({ mask, area, score: sc[k] });
    }
    cand.sort((a, b) => a.area - b.area);
    // auto: the best score among the ones that are not (almost) the whole picture
    let pickC;
    if (level === 'auto') {
      const ok = cand.filter((c) => c.area > 12 && c.area < w * h * maxFrac);
      pickC = (ok.length ? ok : cand).reduce((m, c) => (c.score > m.score ? c : m));
    } else pickC = cand[Math.min(nm - 1, Math.max(0, +level))];
    return pickC;
  }

  return { load, embed, segment, get ready() { return !!emb; }, get device() { return device; }, get local() { return local; } };
})();
