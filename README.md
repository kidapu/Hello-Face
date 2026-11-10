# Hello, Face!

Faces hiding in clouds, leaves, anything.

- **Cloud Camera** — https://kidapu.github.io/Hello-Face/cloud/
  Point your phone at the sky and animal doodles appear on the clouds. The cloud's shape picks the animal
  (cat, bear, snake, fish, dolphin, bird, bat, sheep, whale, or an eye peeking through a hole).
  Tap a face to change the animal, long-press to change the expression.
  The **AI** button (WebGPU browsers only) freezes the frame and segments it: sky, clouds, ground and other things.
- **Debug page** — https://kidapu.github.io/Hello-Face/cloud/debug.html
  For tuning. Open it in a browser:
  1. **Faces** — every animal in every expression, each with an ID such as `cat/smile`, plus sliders for the face style.
  2. **Rules** — the shape → animal rules in the order they are checked, with threshold sliders and what each feature measures.
  3. **Photo** — pick a cloud photo on your device; each face shows its measured features and which rules passed or failed.
     The photo stays on the device.
  4. **Export** — the settings you changed, as text to copy. Settings are kept in the browser.

- **Segmentation view** — https://kidapu.github.io/Hello-Face/cloud/seg.html
  The AI's segmentation alone (no faces), on the camera or a photo, coloured by group, by class or by confidence.
  Knobs for speed and resolution: input size, output resolution (1/8 blocky … 1/1 smooth edges), interval between runs,
  GPU / CPU, fp16 / fp32, worker thread, camera resolution; timings shown at the top left.

## Layout

- `lib/doodle.js` — draws the hand-drawn animal faces (Canvas 2D)
- `lib/cloudfind.js` — finds clouds and their shapes in an image; the shape → animal rules (`CloudFind.RULES`)
- `lib/segnet.js`, `lib/segnet_worker.js` — the AI segmenter (COCO-Stuff 171 classes, onnxruntime-web, in a worker thread)
- `lib/models/cocostuff-lraspp/` — its model (fp16 / fp32 ONNX; `_up2/4/8` = smooth edges at 1/4, 1/2, 1/1) and class names
- `cloud/` — camera web app (`index.html`), debug page (`debug.html`), segmentation view (`seg.html`)
