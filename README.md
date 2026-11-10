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
     A switch at the top picks how the animal is chosen: the **rules**, or **learned** — a small CNN that looks at the
     cloud's silhouette and says which animal it looks most like (trained on animal silhouettes: COCO cats, bears,
     sheep and birds, and drawn snakes, fish, dolphins, whales, bats and flying birds, blurred into cloud-like blobs).
     Each animal has a weight; when the model is unsure, the rules decide. The camera has the same switch (動物の決め方).
  3. **Photo** — pick a cloud photo on your device; each face shows its measured features and which rules passed or failed
     (learned: each animal's probability and the silhouette the model saw).
     The photo stays on the device.
  4. **Export** — the settings you changed, as text to copy. Settings are kept in the browser.

- **Segmentation view** — https://kidapu.github.io/Hello-Face/cloud/seg.html
  The AI's segmentation alone (no faces), on the camera or a photo, coloured by group, by class or by confidence.
  Knobs for speed and resolution: input size, output resolution (1/8 blocky … 1/1 smooth edges), interval between runs,
  GPU / CPU, fp16 / fp32, worker thread, camera resolution; timings shown at the top left.

- **Talking segmentation** — https://kidapu.github.io/Hello-Face/cloud/seg.html?talk
  Tap a region and Zundamon says its name in Japanese, in a few phrases (voices made in advance with VOICEVOX);
  recordings carry the voice too.
  Voice: VOICEVOX:ずんだもん

## Layout

- `lib/doodle.js` — draws the hand-drawn animal faces (Canvas 2D)
- `lib/cloudfind.js` — finds clouds and their shapes in an image; the shape → animal rules (`CloudFind.RULES`)
- `lib/shapenet.js`, `lib/shapenet.json` — the learned shape → animal classifier (plain JS, no WebGPU needed)
- `lib/segnet.js`, `lib/segnet_worker.js` — the AI segmenter (COCO-Stuff 171 classes, onnxruntime-web, in a worker thread)
- `lib/models/cocostuff-lraspp/` — its model (fp16 / fp32 ONNX; `_up2/4/8` = smooth edges at 1/4, 1/2, 1/1) and class names
- `lib/voice/` — the spoken names for `seg.html?talk` (mp3, VOICEVOX:ずんだもん; `v<n><phrase>.mp3`, `voice.json` = phrases + word → n)
- `cloud/` — camera web app (`index.html`), debug page (`debug.html`), segmentation view (`seg.html`)
