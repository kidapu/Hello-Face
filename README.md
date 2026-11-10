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

## Layout

- `lib/doodle.js` — draws the hand-drawn animal faces (Canvas 2D)
- `lib/cloudfind.js` — finds clouds and their shapes in an image; the shape → animal rules (`CloudFind.RULES`)
- `lib/segnet.js` — the AI segmenter (COCO-Stuff 171 classes, onnxruntime-web)
- `lib/models/cocostuff-lraspp/` — its model (fp16 / fp32 ONNX) and class names
- `cloud/` — camera web app (`index.html`) and debug page (`debug.html`)
