/* Facial Emotion Recognition — fully in-browser (ONNX Runtime Web).
   Pipeline mirrors the Python app: YuNet face detection -> eye-level alignment + tight crop (0.95x box, -4% shift)
   -> 112x112 ImageNet-normalised tensor -> DDAMFN -> softmax(logits / T). */
'use strict';

const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.wasm.wasmPaths = ORT_DIST;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

const CLASSES = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral'];
const EMOJI = { Surprise: '😮', Fear: '😨', Disgust: '🤢', Happiness: '😄', Sadness: '😢', Anger: '😠', Neutral: '😐' };
const SENTIMENT = { Surprise: 'Positive', Happiness: 'Positive', Neutral: 'Neutral', Fear: 'Negative', Disgust: 'Negative', Sadness: 'Negative', Anger: 'Negative' };
const COLOR = { Happiness: '#22c55e', Surprise: '#06b6d4', Neutral: '#94a3b8', Sadness: '#3b82f6', Fear: '#a855f7', Disgust: '#eab308', Anger: '#ef4444' };
const MODELS = {
  generalist: { url: 'models/emotion_generalist.onnx', T: 1.0, mb: 16.8 },
  specialist: { url: 'models/emotion_specialist.onnx', T: 1.552, mb: 16.8 },   // T fitted on RAF-DB validation
};
const REGIONS = ['forehead & brows', 'eyes', 'nose bridge', 'mouth & cheeks', 'jaw & chin'];
const HINTS = {
  Happiness: 'the cheeks lift and the corners of the mouth pull upward',
  Sadness: 'the inner eyebrows rise and the corners of the lips turn down',
  Anger: 'the brows are pulled down and together while the lips press firmly',
  Fear: 'the eyes open wide and the lips stretch back',
  Disgust: 'the nose wrinkles and the upper lip rises',
  Surprise: 'the brows arch high and the jaw drops open',
  Neutral: 'the face is relaxed with no strong muscle activity',
};
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const S = 112, DET = 640;

const $ = (id) => document.getElementById(id);
let modelKey = 'generalist';

/* ------------------------------------------------------------------ loading (with Cache API) */
async function fetchModel(url, label, mb) {
  try {
    const cache = await caches.open('fer-models-v1');
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
    showLoader(`Downloading ${label} (${mb} MB, one time)…`, 0);
    const res = await fetch(url);
    const total = +res.headers.get('content-length') || mb * 1e6;
    const reader = res.body.getReader(); const chunks = []; let got = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; showLoader(null, got / total); }
    const buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
    try { await cache.put(url, new Response(buf.slice())); } catch (e) { /* storage may be unavailable */ }
    return buf;
  } catch (e) {   // Cache API unavailable (e.g. some private modes): plain fetch
    showLoader(`Downloading ${label}…`, 0.3);
    return new Uint8Array(await (await fetch(url)).arrayBuffer());
  }
}
function showLoader(text, frac) {
  $('loader').classList.add('show');
  if (text) $('loaderText').textContent = text;
  if (frac != null) $('loaderBar').style.width = `${Math.round(frac * 100)}%`;
}
const hideLoader = () => $('loader').classList.remove('show');

// ONNX Runtime Web cannot run two inferences on one session at the same time ("Session already started"),
// so every run goes through a single promise queue.
let ortQueue = Promise.resolve();
function runSerial(sess, feeds) { const p = ortQueue.then(() => sess.run(feeds)); ortQueue = p.catch(() => {}); return p; }

const sessions = {};
function session(key) {
  if (!sessions[key]) {
    sessions[key] = (async () => {
      const buf = key === 'yunet' ? await fetchModel('models/yunet.onnx', 'face detector', 0.2)
                                  : await fetchModel(MODELS[key].url, `${key} model`, MODELS[key].mb);
      showLoader('Starting the model…', 1);
      const s = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      hideLoader(); return s;
    })().catch((e) => { delete sessions[key]; hideLoader(); alert('Could not load the model: ' + e.message); throw e; });
  }
  return sessions[key];
}

/* ------------------------------------------------------------------ face detection (YuNet) */
const detCanvas = document.createElement('canvas'); detCanvas.width = detCanvas.height = DET;
const detCtx = detCanvas.getContext('2d', { willReadFrequently: true });
const detInput = new Float32Array(3 * DET * DET);

async function detectFaces(src, W, H, thr = 0.6) {
  const det = await session('yunet');
  const s = Math.min(DET / W, DET / H), nw = Math.round(W * s), nh = Math.round(H * s);
  detCtx.fillStyle = '#000'; detCtx.fillRect(0, 0, DET, DET);
  detCtx.drawImage(src, 0, 0, nw, nh);
  const px = detCtx.getImageData(0, 0, DET, DET).data, N = DET * DET;
  for (let i = 0; i < N; i++) {           // BGR planes, 0..255, no normalisation (as OpenCV's FaceDetectorYN)
    detInput[i] = px[i * 4 + 2]; detInput[N + i] = px[i * 4 + 1]; detInput[2 * N + i] = px[i * 4];
  }
  const out = await runSerial(det, { input: new ort.Tensor('float32', detInput.slice(), [1, 3, DET, DET]) });
  const cand = [];
  for (const st of [8, 16, 32]) {
    const cols = DET / st, cls = out[`cls_${st}`].data, obj = out[`obj_${st}`].data, bb = out[`bbox_${st}`].data, kp = out[`kps_${st}`].data;
    for (let i = 0; i < cls.length; i++) {
      const score = Math.sqrt(Math.min(Math.max(cls[i], 0), 1) * Math.min(Math.max(obj[i], 0), 1));
      if (score < thr) continue;
      const r = Math.floor(i / cols), c = i % cols;
      const cx = (c + bb[i * 4]) * st, cy = (r + bb[i * 4 + 1]) * st, w = Math.exp(bb[i * 4 + 2]) * st, h = Math.exp(bb[i * 4 + 3]) * st;
      const lm = []; for (let n = 0; n < 5; n++) lm.push([(kp[i * 10 + 2 * n] + c) * st / s, (kp[i * 10 + 2 * n + 1] + r) * st / s]);
      cand.push({ box: [(cx - w / 2) / s, (cy - h / 2) / s, w / s, h / s], lm, score });
    }
  }
  cand.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const f of cand) if (keep.every((k) => iou(k.box, f.box) <= 0.3)) keep.push(f);
  return keep.sort((a, b) => b.box[2] * b.box[3] - a.box[2] * a.box[3]);
}
function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1); return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9);
}

/* ------------------------------------------------------------------ alignment + crop + tensor */
const midCanvas = document.createElement('canvas'), midCtx = midCanvas.getContext('2d');
function alignCrop(src, face) {           // -> 112x112 canvas, same geometry as the Python align_crop()
  const [x, y, w, h] = face.box;
  const cx = x + w / 2, cy = y + h / 2 - 0.04 * h, side = 0.95 * Math.max(w, h);
  let ang = 0;
  if (face.lm) { const [re, le] = face.lm; ang = Math.atan2(le[1] - re[1], le[0] - re[0]); if (Math.abs(ang) > Math.PI / 4) ang = 0; }
  const m = Math.max(S, Math.min(448, Math.round(side)));      // rotate at moderate resolution, then downscale (antialiased)
  midCanvas.width = midCanvas.height = m;
  midCtx.setTransform(1, 0, 0, 1, 0, 0); midCtx.fillStyle = '#000'; midCtx.fillRect(0, 0, m, m);
  midCtx.imageSmoothingQuality = 'high';
  midCtx.translate(m / 2, m / 2); midCtx.scale(m / side, m / side); midCtx.rotate(-ang); midCtx.translate(-cx, -cy);
  midCtx.drawImage(src, 0, 0);
  const out = document.createElement('canvas'); out.width = out.height = S;
  const octx = out.getContext('2d', { willReadFrequently: true }); octx.imageSmoothingQuality = 'high';
  octx.drawImage(midCanvas, 0, 0, S, S);
  return out;
}
function toTensorData(crop) {
  const px = crop.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, S, S).data, N = S * S, t = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) for (let c = 0; c < 3; c++) t[c * N + i] = (px[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  return t;
}
async function classifyData(data, key = modelKey) {
  const sess = await session(key);
  const out = await runSerial(sess, { input: new ort.Tensor('float32', data, [1, 3, S, S]) });
  const z = Array.from(out.logits.data, (v) => v / MODELS[key].T), mx = Math.max(...z);
  const e = z.map((v) => Math.exp(v - mx)), sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}
const argmax = (p) => p.indexOf(Math.max(...p));

/* ------------------------------------------------------------------ rendering helpers */
function renderBars(el, probs) {
  const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  el.innerHTML = order.map(([p, i]) => `<div class="bar"><span>${EMOJI[CLASSES[i]]} ${CLASSES[i]}</span>
    <div class="track"><div class="fill" style="width:${(p * 100).toFixed(1)}%;background:${COLOR[CLASSES[i]]}"></div></div>
    <span class="v">${(p * 100).toFixed(0)}%</span></div>`).join('');
}
function renderVerdict(prefix, probs) {
  const i = argmax(probs), emo = CLASSES[i], sent = SENTIMENT[emo];
  $(prefix + 'Emoji').textContent = EMOJI[emo]; $(prefix + 'Name').textContent = `${emo} · ${(probs[i] * 100).toFixed(0)}%`;
  const pill = $(prefix + 'Pill'); pill.className = `pill ${sent}`; pill.textContent = `${sent} sentiment`;
}
function lowConfNote(probs) {
  const o = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  return o[0][0] >= 0.55 ? '' : `<div class="note">⚠️ Low confidence (${(o[0][0] * 100).toFixed(0)}%). This could also be <b>${CLASSES[o[1][1]]}</b> (${(o[1][0] * 100).toFixed(0)}%). Expressions are often genuinely ambiguous.</div>`;
}
function drawBox(ctx, face, probs, label, scale, mirrorW) {
  const i = argmax(probs), emo = CLASSES[i], col = COLOR[emo];
  let [x, y, w, h] = face.box.map((v) => v * scale);
  if (mirrorW) x = mirrorW - x - w;
  const lw = Math.max(2, Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 320));
  ctx.lineWidth = lw; ctx.strokeStyle = col; ctx.beginPath(); ctx.roundRect(x, y, w, h, Math.min(12, w / 8)); ctx.stroke();
  const fs = Math.round(Math.max(ctx.canvas.width / 64, Math.min(30, w / 5))); ctx.font = `700 ${fs}px Inter, sans-serif`;
  const text = `${label}${EMOJI[emo]} ${emo} ${(probs[i] * 100).toFixed(0)}%`, tw = ctx.measureText(text).width;
  const ty = y - fs - 10 > 0 ? y - fs - 10 : y + h + 4;
  ctx.fillStyle = col; ctx.beginPath(); ctx.roundRect(x - lw / 2, ty, tw + 14, fs + 8, 7); ctx.fill();
  ctx.fillStyle = '#fff'; ctx.fillText(text, x + 7 - lw / 2, ty + fs);
}
function fitCanvas(canvas, W, H, maxSide = 1280) {
  const k = Math.min(1, maxSide / Math.max(W, H)); canvas.width = Math.round(W * k); canvas.height = Math.round(H * k); return k;
}

/* ------------------------------------------------------------------ LIVE */
let stream = null, running = false, ema = null, lastT = 0, frames = 0;
const video = document.createElement('video'); video.playsInline = true; video.muted = true;
const frame = document.createElement('canvas');     // unmirrored copy of the current video frame
async function startCam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false });
  } catch (e) { alert('Camera unavailable: ' + e.message); return; }
  video.srcObject = stream; await video.play();
  $('livePh').hidden = true; $('fps').hidden = false; $('camBtn').textContent = '■ Stop camera';
  await Promise.all([session('yunet'), session(modelKey)]);
  running = true; ema = null; loop();
}
function stopCam() {
  running = false; if (stream) stream.getTracks().forEach((t) => t.stop()); stream = null;
  $('camBtn').textContent = '▶ Start camera'; $('fps').hidden = true;
}
async function loop() {
  if (!running) return;
  const W = video.videoWidth, H = video.videoHeight, canvas = $('liveCanvas');
  if (W && H) {
    canvas.width = W; canvas.height = H; const ctx = canvas.getContext('2d');
    ctx.save(); ctx.translate(W, 0); ctx.scale(-1, 1); ctx.drawImage(video, 0, 0, W, H); ctx.restore();   // mirror like a selfie
    frame.width = W; frame.height = H; frame.getContext('2d').drawImage(video, 0, 0);
    const faces = (await detectFaces(frame, W, H)).slice(0, 4);
    const probsList = [];
    for (const f of faces) probsList.push(await classifyData(toTensorData(alignCrop(frame, f))));
    if (faces.length) {
      ema = ema ? ema.map((v, i) => 0.55 * v + 0.45 * probsList[0][i]) : probsList[0];
      probsList[0] = ema;
      faces.forEach((f, k) => drawBox(ctx, f, probsList[k], '', 1, W));
      renderVerdict('live', ema); renderBars($('liveBars'), ema);
      $('liveNote').innerHTML = `${faces.length} face${faces.length > 1 ? 's' : ''} in view.` + lowConfNote(ema).replace('<div class="note">', ' ').replace('</div>', '');
    } else { $('liveName').textContent = 'No face detected'; $('livePill').className = 'pill Neutral'; $('livePill').textContent = 'look at the camera'; }
    frames++; const now = performance.now();
    if (now - lastT > 1000) { $('fps').textContent = `${(frames * 1000 / (now - lastT)).toFixed(1)} fps · on-device`; lastT = now; frames = 0; }
  }
  requestAnimationFrame(loop);
}
$('camBtn').onclick = () => (running ? stopCam() : startCam());

/* ------------------------------------------------------------------ PHOTO */
let photoState = null, lastPhoto = null, lastGroup = null;
async function analysePhoto(img) {
  lastPhoto = img;
  const W = img.naturalWidth, H = img.naturalHeight, canvas = $('photoCanvas');
  const k = fitCanvas(canvas, W, H); const ctx = canvas.getContext('2d'); ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  $('photoPh').hidden = true; $('explainBox').hidden = true; $('explainStatus').textContent = '';
  const faces = await detectFaces(img, W, H);
  const face = faces[0] || { box: [0, 0, W, H], lm: null };          // no face found: treat image as a face crop
  const crop = alignCrop(img, face), data = toTensorData(crop), probs = await classifyData(data);
  if (faces.length) drawBox(ctx, face, probs, '', k);
  renderVerdict('photo', probs); renderBars($('photoBars'), probs);
  $('photoLowConf').innerHTML = (faces.length ? '' : '<div class="note">No face detected, so the whole image was treated as a face crop.</div>') + lowConfNote(probs);
  photoState = { crop, data, probs }; $('explainBtn').disabled = false;
}
async function explain() {
  if (!photoState) return;
  const { crop, data, probs } = photoState, cls = argmax(probs), base = probs[cls];
  $('explainBtn').disabled = true;
  const G = 7, P = S / G, heat = new Float32Array(G * G), N = S * S;
  for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {     // occlusion sensitivity: hide one patch, measure the drop
    const d = data.slice();
    for (let y = Math.round(gy * P); y < Math.round((gy + 1) * P); y++) for (let x = Math.round(gx * P); x < Math.round((gx + 1) * P); x++)
      for (let c = 0; c < 3; c++) d[c * N + y * S + x] = 0;   // 0 = dataset-mean colour after normalisation
    heat[gy * G + gx] = Math.max(0, base - (await classifyData(d))[cls]);
    $('explainStatus').textContent = `analysing… ${Math.round(((gy * G + gx + 1) / (G * G)) * 100)}%`;
  }
  // 5 horizontal bands -> facial regions
  const bands = [];
  for (let b = 0; b < 5; b++) {
    const d = data.slice(), y0 = Math.round(b * S / 5), y1 = Math.round((b + 1) * S / 5);
    for (let y = y0; y < y1; y++) for (let x = 0; x < S; x++) for (let c = 0; c < 3; c++) d[c * N + y * S + x] = 0;
    bands.push(Math.max(0, base - (await classifyData(d))[cls]));
  }
  const tot = bands.reduce((a, b) => a + b, 0) || 1, share = bands.map((v) => v / tot);
  drawHeat(crop, heat, G);
  const order = share.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
  $('regionBars').innerHTML = order.map(([v, i]) => `<div class="bar"><span>${REGIONS[i]}</span><div class="track"><div class="fill" style="width:${(v * 100).toFixed(0)}%;background:var(--accent)"></div></div><span class="v">${(v * 100).toFixed(0)}%</span></div>`).join('');
  const emo = CLASSES[cls];
  $('explainText').innerHTML = `Hiding the <b>${REGIONS[order[0][1]]}</b> lowers the model's confidence in <b>${emo}</b> the most, followed by the <b>${REGIONS[order[1][1]]}</b>. That is consistent with ${emo.toLowerCase()} expressions, where ${HINTS[emo]}.`;
  $('explainBox').hidden = false; $('explainStatus').textContent = 'warmer = more important'; $('explainBtn').disabled = false;
}
function drawHeat(crop, heat, G) {
  const c = $('heatCanvas'), ctx = c.getContext('2d'), mx = Math.max(...heat) || 1;
  ctx.imageSmoothingEnabled = true; ctx.drawImage(crop, 0, 0, c.width, c.height);
  const hm = document.createElement('canvas'); hm.width = hm.height = G; const hctx = hm.getContext('2d'), id = hctx.createImageData(G, G);
  for (let i = 0; i < G * G; i++) { const v = heat[i] / mx; const [r, g, b] = inferno(v); id.data.set([r, g, b, Math.round(40 + 175 * v)], i * 4); }
  hctx.putImageData(id, 0, 0); ctx.globalAlpha = 0.6; ctx.drawImage(hm, 0, 0, c.width, c.height); ctx.globalAlpha = 1;
}
function inferno(t) {   // compact warm colormap: dark purple -> red -> yellow
  const stops = [[20, 11, 52], [120, 28, 109], [207, 68, 70], [251, 155, 6], [252, 255, 164]];
  const x = Math.min(0.999, Math.max(0, t)) * (stops.length - 1), i = Math.floor(x), f = x - i;
  return stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
}
$('explainBtn').onclick = explain;

/* ------------------------------------------------------------------ GROUP */
async function analyseGroup(img) {
  lastGroup = img;
  const W = img.naturalWidth || img.width, H = img.naturalHeight || img.height, canvas = $('groupCanvas');
  const k = fitCanvas(canvas, W, H); const ctx = canvas.getContext('2d'); ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  $('groupPh').hidden = true;
  const faces = (await detectFaces(img, W, H, 0.6)).slice(0, 60);
  if (!faces.length) { $('groupName').textContent = 'No faces found'; $('groupCount').textContent = 'Try a clearer, front-facing photo'; return; }
  const crops = [], probsList = [];
  for (const f of faces) { const cr = alignCrop(img, f); crops.push(cr); probsList.push(await classifyData(toTensorData(cr))); }
  faces.forEach((f, i) => drawBox(ctx, f, probsList[i], `#${i + 1} `, k));
  const pos = [0, 3], neg = [1, 2, 4, 5];
  const idx = probsList.reduce((a, p) => a + pos.reduce((s, i) => s + p[i], 0) - neg.reduce((s, i) => s + p[i], 0), 0) / faces.length;
  const sents = probsList.map((p) => SENTIMENT[CLASSES[argmax(p)]]);
  const [emoji, verdict] = idx > 0.35 ? ['😊', 'Positive room'] : idx < -0.2 ? ['😟', 'Negative room'] : ['😐', 'Mixed / neutral room'];
  $('groupEmoji').textContent = emoji; $('groupName').textContent = verdict;
  $('groupCount').textContent = `${faces.length} face${faces.length > 1 ? 's' : ''} analysed`;
  $('needle').style.left = `${((idx + 1) / 2) * 100}%`; $('moodIdx').textContent = `mood index ${idx >= 0 ? '+' : ''}${idx.toFixed(2)}`;
  $('groupBars').innerHTML = ['Positive', 'Neutral', 'Negative'].map((s) => {
    const v = sents.filter((x) => x === s).length / faces.length, col = s === 'Positive' ? 'var(--pos)' : s === 'Negative' ? 'var(--neg)' : 'var(--neu)';
    return `<div class="bar"><span>${s}</span><div class="track"><div class="fill" style="width:${(v * 100).toFixed(0)}%;background:${col}"></div></div><span class="v">${(v * 100).toFixed(0)}%</span></div>`;
  }).join('');
  const list = $('faceList'); list.innerHTML = '';
  crops.forEach((cr, i) => {
    const d = document.createElement('div'), p = probsList[i], e = CLASSES[argmax(p)]; d.className = 'face';
    const cv = document.createElement('canvas'); cv.width = cv.height = S; cv.getContext('2d').drawImage(cr, 0, 0);
    d.appendChild(cv); d.insertAdjacentHTML('beforeend', `<b>#${i + 1}</b> ${EMOJI[e]} ${e}<br><span class="muted">${(Math.max(...p) * 100).toFixed(0)}%</span>`);
    list.appendChild(d);
  });
}
$('groupCamBtn').onclick = async () => {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } } });
    const v = document.createElement('video'); v.srcObject = s; v.playsInline = true; v.muted = true; await v.play();
    await new Promise((r) => setTimeout(r, 600));
    const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight; c.getContext('2d').drawImage(v, 0, 0);
    s.getTracks().forEach((t) => t.stop());
    await analyseGroup(c);
  } catch (e) { alert('Camera unavailable: ' + e.message); }
};

/* ------------------------------------------------------------------ inputs: drop, pick, examples */
function loadImage(src) { return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; }); }
function wireDrop(stageId, fileId, handler) {
  const stage = $(stageId), file = $(fileId);
  stage.onclick = () => file.click();
  file.onchange = async () => { if (file.files[0]) handler(await loadImage(URL.createObjectURL(file.files[0]))); file.value = ''; };
  stage.ondragover = (e) => { e.preventDefault(); stage.classList.add('over'); };
  stage.ondragleave = () => stage.classList.remove('over');
  stage.ondrop = async (e) => { e.preventDefault(); stage.classList.remove('over'); const f = e.dataTransfer.files[0]; if (f) handler(await loadImage(URL.createObjectURL(f))); };
}
wireDrop('photoStage', 'photoFile', analysePhoto);
wireDrop('groupStage', 'groupFile', analyseGroup);
document.querySelectorAll('.examples').forEach((box) => box.querySelectorAll('img').forEach((im) => {
  im.onclick = async () => { const img = await loadImage(im.src); (box.dataset.target === 'photo' ? analysePhoto : analyseGroup)(img); };
}));
document.addEventListener('paste', async (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/')); if (!item) return;
  const img = await loadImage(URL.createObjectURL(item.getAsFile()));
  (document.querySelector('.tab[aria-selected="true"]').dataset.tab === 'group' ? analyseGroup : analysePhoto)(img);
});

/* ------------------------------------------------------------------ tabs, model switch, theme */
document.querySelectorAll('.tab').forEach((t) => t.onclick = () => {
  document.querySelectorAll('.tab').forEach((x) => x.setAttribute('aria-selected', x === t));
  document.querySelectorAll('.panel').forEach((p) => p.classList.toggle('active', p.id === 'panel-' + t.dataset.tab));
  if (t.dataset.tab !== 'live' && running) stopCam();
});
$('modelSel').onchange = async (e) => {   // switch model -> re-run whatever is on screen with the new one
  modelKey = e.target.value; ema = null;
  await session(modelKey);
  if (lastPhoto) await analysePhoto(lastPhoto);
  if (lastGroup) await analyseGroup(lastGroup);
};
$('themeBtn').onclick = () => {
  const cur = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.dataset.theme = cur === 'dark' ? 'light' : 'dark';
};
renderBars($('liveBars'), CLASSES.map(() => 0)); renderBars($('photoBars'), CLASSES.map(() => 0));
// warm up: fetch the detector + default model in the background so the first click is instant
session('yunet').then(() => session(modelKey)).catch(() => {});

// deep links for demos: #group / #photo open that tab and run its first example (#photo also explains it)
(async () => {
  const h = location.hash.replace('#', '');
  if (h !== 'group' && h !== 'photo') return;
  document.querySelector(`.tab[data-tab="${h}"]`).click();
  const img = await loadImage(document.querySelector(`.examples[data-target="${h}"] img`).src);
  if (h === 'group') await analyseGroup(img); else { await analysePhoto(img); await explain(); }
  document.body.dataset.demoDone = '1';
})();

// expose for automated checks
window.__fer = { detectFaces, alignCrop, toTensorData, classifyData, analysePhoto, analyseGroup, loadImage, setModel: (k) => { modelKey = k; } };
