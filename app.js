// xraypoint 웹앱 — 브라우저 내 추론 (onnxruntime-web, GPU 불필요)
// 파이프라인: 좌우 분할(왼쪽 미러) → stage1(256×512) → 비율 ROI → hip/knee/ankle
// → DARK 서브픽셀 디코딩 → 원본 좌표 복원 → 각도 계산 (파이썬 구현과 동일 로직)
"use strict";

ort.env.wasm.wasmPaths = new URL("lib/", location.href).href;
ort.env.wasm.numThreads = 1; // cross-origin isolation 없이 동작

const MODELS = {};   // name -> ort.InferenceSession
let META = null;     // models/meta.json

// ---------- 상태 ----------
const state = {
  images: [],   // {name, bitmap, W, H, legs:{right:{points,angles,edited:Set}|null, left:...}}
  cur: -1,
  showPts: true, showAxes: true, showNames: false,
  angOn: { mHKA: true, MPTA: true, LDFA: true, JLCA: true, WBL: true },
  sideOn: { right: true, left: true }, // 환자 기준 좌우
  jointOn: { hip: true, knee: true, ankle: true }, // 관절 확대창

  zoom: { k: 1, tx: 0, ty: 0 },
};

const KNEE_NAMES = ["p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p15"];
const ANGLE_DEFS = { mHKA: [["p1","p4"],["p9","p14"]], MPTA: [["p9","p14"],["p5","p6"]],
                     LDFA: [["p4","p1"],["p3","p2"]], JLCA: [["p2","p3"],["p5","p6"]] };
const MEASURES = ["mHKA","MPTA","LDFA","JLCA","WBL"];
const MEASURE_LINES = { // 측정 항목별 축선 (true = 점선)
  mHKA: [["p1","p4",false],["p9","p14",false]],
  MPTA: [["p9","p14",false],["p5","p6",false]],
  LDFA: [["p4","p1",false],["p3","p2",false]],
  JLCA: [["p2","p3",false],["p5","p6",false]],
  WBL:  [["p1","p14",true]],
};
const PT_ORDER = ["p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15"];

// ---------- 수학/영상 유틸 (파이썬 파이프라인 이식) ----------
function toGray(imgData) { // ImageData -> Float32Array [0,1]
  const { data, width, height } = imgData;
  const g = new Float32Array(width * height);
  for (let i = 0, j = 0; i < g.length; i++, j += 4)
    g[i] = (0.299 * data[j] + 0.587 * data[j+1] + 0.114 * data[j+2]) / 255.0;
  return g;
}

// cv2.GaussianBlur(ch,(5,5),0) 등가: 이항 커널 [1,4,6,4,1]/16, BORDER_REFLECT_101
function blur5(src, H, W) {
  const k = [1/16, 4/16, 6/16, 4/16, 1/16];
  const tmp = new Float32Array(H * W), out = new Float32Array(H * W);
  const r101 = (i, n) => i < 0 ? -i : (i >= n ? 2*n - 2 - i : i);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let d = -2; d <= 2; d++) s += k[d+2] * src[y*W + r101(x+d, W)];
      tmp[y*W + x] = s;
    }
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let d = -2; d <= 2; d++) s += k[d+2] * tmp[r101(y+d, H)*W + x];
      out[y*W + x] = s;
    }
  return out;
}

// DARK 서브픽셀 디코딩 (heatmap.py _dark_refine 이식)
function decodeHeatmaps(data, C, H, W) {
  const coords = [];
  for (let c = 0; c < C; c++) {
    const ch = data.subarray(c*H*W, (c+1)*H*W);
    let best = -Infinity, bi = 0;
    for (let i = 0; i < ch.length; i++) if (ch[i] > best) { best = ch[i]; bi = i; }
    let x = bi % W, y = (bi / W) | 0, fx = x, fy = y;
    if (x >= 1 && x < W-1 && y >= 1 && y < H-1) {
      const sm = blur5(ch, H, W);
      const L = (yy, xx) => Math.log(Math.max(sm[yy*W + xx], 1e-10));
      const dx = 0.5 * (L(y, x+1) - L(y, x-1));
      const dy = 0.5 * (L(y+1, x) - L(y-1, x));
      const dxx = L(y, x+1) - 2*L(y, x) + L(y, x-1);
      const dyy = L(y+1, x) - 2*L(y, x) + L(y-1, x);
      const dxy = 0.25 * (L(y+1, x+1) - L(y+1, x-1) - L(y-1, x+1) + L(y-1, x-1));
      const det = dxx*dyy - dxy*dxy;
      if (Math.abs(det) >= 1e-12) {
        const ox = -(dyy*dx - dxy*dy) / det, oy = -(-dxy*dx + dxx*dy) / det;
        if (Math.abs(ox) <= 1 && Math.abs(oy) <= 1) { fx = x + ox; fy = y + oy; }
      }
    }
    coords.push([fx, fy]);
  }
  return coords;
}

function calcAngles(pts, side) {
  const out = {};
  for (const [name, [[a,b],[c,d]]] of Object.entries(ANGLE_DEFS)) {
    if (!(pts[a] && pts[b] && pts[c] && pts[d])) continue;
    const v1 = [pts[b][0]-pts[a][0], pts[b][1]-pts[a][1]];
    const v2 = [pts[d][0]-pts[c][0], pts[d][1]-pts[c][1]];
    let ang = Math.atan2(v1[0]*v2[1]-v1[1]*v2[0], v1[0]*v2[0]+v1[1]*v2[1]) * 180 / Math.PI;
    if (side === "right") ang = -ang;
    if (ang > 180) ang -= 360; else if (ang < -180) ang += 360;
    out[name] = ang;
  }
  if (pts.p1 && pts.p14 && pts.p6 && pts.p5) {
    const [k1, k2] = [pts.p1, pts.p14];
    const den = Math.hypot(k2[1]-k1[1], k2[0]-k1[0]);
    if (den > 1e-6) {
      const dl = p => Math.abs((k2[1]-k1[1])*p[0]-(k2[0]-k1[0])*p[1]+k2[0]*k1[1]-k2[1]*k1[0]) / den;
      out.WBL = dl(pts.p6) / (dl(pts.p6) + dl(pts.p5)) * 100;
    }
  }
  return out;
}

// ---------- 모델 로딩 ----------
async function loadModels() {
  META = await (await fetch("models/meta.json")).json();
  const names = ["stage1", "hip", "knee", "ankle"];
  for (let i = 0; i < names.length; i++) {
    setStatus(`모델 로딩 ${i+1}/4 (${names[i]})…`);
    MODELS[names[i]] = await ort.InferenceSession.create(`models/${names[i]}.onnx`,
      { executionProviders: ["wasm"] });
  }
  setStatus("준비 완료 — X-ray를 열어주세요");
}

// ---------- 추론 파이프라인 ----------
function canvasGrayTensor(cv) { // canvas -> ort.Tensor [1,1,H,W]
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  const g = toGray(ctx.getImageData(0, 0, cv.width, cv.height));
  return { tensor: new ort.Tensor("float32", g, [1, 1, cv.height, cv.width]), gray: g };
}

async function runModel(name, cv) {
  const { tensor } = canvasGrayTensor(cv);
  const out = await MODELS[name].run({ image: tensor });
  const hm = out.heatmaps;
  const [, C, H, W] = hm.dims;
  return decodeHeatmaps(hm.data, C, H, W);
}

function makeCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "medium";
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, w, h);
  return c;
}

async function predictLeg(img, side) { // img: {bitmap, W, H} -> {points, angles} | null
  const hw = Math.floor(img.W / 2);
  // 반다리 캔버스 (왼쪽은 수평 반전해 오른쪽 방향으로 통일 — split_leg 등가)
  const half = makeCanvas(hw, img.H);
  const hctx = half.getContext("2d");
  if (side === "right") hctx.drawImage(img.bitmap, 0, 0, hw, img.H, 0, 0, hw, img.H);
  else { hctx.save(); hctx.translate(hw, 0); hctx.scale(-1, 1);
         hctx.drawImage(img.bitmap, img.W - hw, 0, hw, img.H, 0, 0, hw, img.H); hctx.restore(); }

  // [1] stage1: letterbox 256×512
  const s1meta = META.stage1;
  const [tw, th] = [s1meta.input[3], s1meta.input[2]];
  const scale = Math.min(tw / hw, th / img.H);
  const nw = Math.round(hw * scale), nh = Math.round(img.H * scale);
  const ox = Math.floor((tw - nw) / 2), oy = Math.floor((th - nh) / 2);
  const lb = makeCanvas(tw, th);
  lb.getContext("2d").drawImage(half, 0, 0, hw, img.H, ox, oy, nw, nh);
  const s1coords = await runModel("stage1", lb);
  const centers = {};
  s1meta.landmarks.forEach((n, i) =>
    centers[n] = [(s1coords[i][0] - ox) / scale, (s1coords[i][1] - oy) / scale]);

  const L = Math.hypot(centers.hip_center[0] - centers.ankle_center[0],
                       centers.hip_center[1] - centers.ankle_center[1]);
  if (!isFinite(L) || L <= 1) return null;

  // [2] 관절별 ROI → 정밀 예측기 → 역변환 (crop_roi/AffineMap 등가)
  const points = {};
  for (const joint of ["hip", "knee", "ankle"]) {
    const m = META[joint];
    const out = m.input[2];
    const sideLen = m.roi.ratio * L;
    const ck = { hip: "hip_center", knee: "knee_center", ankle: "ankle_center" }[joint];
    const [cx, cy] = centers[ck];
    const x0 = cx - sideLen / 2, y0 = cy - sideLen / 2;
    const roi = makeCanvas(out, out);
    roi.getContext("2d").drawImage(half, x0, y0, sideLen, sideLen, 0, 0, out, out);
    const coords = await runModel(joint, roi);
    const s = out / sideLen;
    m.landmarks.forEach((n, i) => {
      const hx = coords[i][0] / s + x0, hy = coords[i][1] / s + y0; // 반다리 좌표
      // points_to_full 등가: 왼쪽이면 반전 복원
      const fx = side === "right" ? hx : (img.W - hw) + (hw - 1) - hx;
      points[n] = [fx, hy];
    });
  }
  return { points, angles: calcAngles(points, side), edited: new Set() };
}

async function processImage(entry) {
  entry.status = "분석 중…"; renderList();
  try {
    entry.legs = {};
    for (const side of ["right", "left"]) {
      entry.legs[side] = await predictLeg(entry, side);
      entry.status = side === "right" ? "분석 중… (오른쪽 완료)" : "완료";
      renderList();
      await new Promise(r => setTimeout(r, 0)); // UI 갱신 양보
    }
  } catch (e) {
    console.error(e); entry.status = "실패: " + e.message;
  }
  renderList(); renderAll();
  updateCsvButtons();
}

// ---------- 파일 입력 ----------
async function addFiles(files) {
  for (const f of files) {
    const bitmap = await createImageBitmap(f);
    const entry = { name: f.name, bitmap, W: bitmap.width, H: bitmap.height,
                    legs: null, status: "대기" };
    state.images.push(entry);
    if (state.cur < 0) selectImage(state.images.length - 1);
  }
  renderList();
  for (const entry of state.images) if (!entry.legs && !entry.status.startsWith("실패"))
    await processImage(entry);
}

function selectImage(i) {
  state.cur = i;
  state.zoom = { k: 1, tx: 0, ty: 0 };
  renderList(); renderAll(); updateCsvButtons();
}

// ---------- UI: 목록/표 ----------
const $ = id => document.getElementById(id);
function setStatus(t) { $("status").textContent = t; }

function renderList() {
  const el = $("filelist");
  el.innerHTML = "";
  state.images.forEach((im, i) => {
    const d = document.createElement("div");
    d.className = "item" + (i === state.cur ? " sel" : "");
    d.innerHTML = `${im.name}<div class="st">${im.status}</div>`;
    d.onclick = () => selectImage(i);
    el.appendChild(d);
  });
}

function fmtAng(v, n) {
  return v === undefined ? "—" : v.toFixed(1) + (n === "WBL" ? "%" : "°");
}

function renderAngles() {
  const el = $("angles");
  const im = state.images[state.cur];
  if (!im || !im.legs) { el.innerHTML = '<div class="sec">분석 대기 중…</div>'; return; }
  const dimS = s => state.sideOn[s] ? "" : ' style="opacity:.35"';
  let h = `<table><tr><th></th><th${dimS("right")}>오른쪽</th><th${dimS("left")}>왼쪽</th></tr>`;
  MEASURES.forEach((n, i) => {
    const r = im.legs.right ? im.legs.right.angles[n] : undefined;
    const l = im.legs.left ? im.legs.left.angles[n] : undefined;
    const dim = state.angOn[n] ? "" : ' style="opacity:.35"';
    h += `<tr${dim}><td><kbd>${i+1}</kbd> ${n}</td><td${dimS("right")}>${fmtAng(r, n)}</td>` +
         `<td${dimS("left")}>${fmtAng(l, n)}</td></tr>`;
  });
  h += "</table>";
  const ed = ["right","left"].reduce((s, k) => s + (im.legs[k] ? im.legs[k].edited.size : 0), 0);
  if (ed) h += `<div class="sec">수정된 점 ${ed}개 반영됨</div>`;
  el.innerHTML = h;
}

// ---------- UI: 캔버스 ----------
const cv = $("cv");
const cctx = cv.getContext("2d");

function viewTransform(im) { // 기본 fit + 사용자 줌 합성
  const W = cv.clientWidth, H = cv.clientHeight;
  const s0 = Math.min(W / im.W, H / im.H);
  const bx = (W - im.W * s0) / 2, by = (H - im.H * s0) / 2;
  const z = state.zoom;
  return { S: s0 * z.k, OX: bx * z.k + z.tx, OY: by * z.k + z.ty };
}

function renderCanvas() {
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  cctx.clearRect(0, 0, W, H);
  const im = state.images[state.cur];
  if (!im) return;
  $("cvtitle").textContent = `${im.name} (${im.W}×${im.H})`;
  const { S, OX, OY } = viewTransform(im);
  cctx.imageSmoothingEnabled = S < 3;
  cctx.drawImage(im.bitmap, OX, OY, im.W * S, im.H * S);
  if (!im.legs) return;
  for (const side of ["right", "left"]) {
    const leg = im.legs[side];
    if (!leg || !state.sideOn[side]) continue;
    const pts = leg.points;
    if (state.showAxes) {
      cctx.lineWidth = 1.5; cctx.strokeStyle = "#4FB3C6"; cctx.globalAlpha = 0.8;
      for (const [a, b, dashed] of activeSegs()) {
        if (!pts[a] || !pts[b]) continue;
        cctx.setLineDash(dashed ? [7, 6] : []);
        cctx.beginPath();
        cctx.moveTo(pts[a][0]*S + OX, pts[a][1]*S + OY);
        cctx.lineTo(pts[b][0]*S + OX, pts[b][1]*S + OY);
        cctx.stroke();
      }
      cctx.setLineDash([]); cctx.globalAlpha = 1;
    }
    if (state.showPts) {
      for (const n of PT_ORDER) {
        if (!pts[n]) continue;
        const x = pts[n][0]*S + OX, y = pts[n][1]*S + OY;
        cctx.beginPath(); cctx.arc(x, y, 2.5, 0, 7);
        cctx.fillStyle = leg.edited.has(n) ? "#FFD966" : "#38BDF8";
        cctx.fill();
        if (state.showNames) {
          cctx.font = "bold 12px sans-serif";
          cctx.lineWidth = 3; cctx.strokeStyle = "rgba(0,0,0,.85)";
          cctx.strokeText(n, x + 7, y - 7);
          cctx.fillStyle = "#FFD966"; cctx.fillText(n, x + 7, y - 7);
        }
      }
    }
  }
}

function activeSegs() { // 켜진 측정 항목의 축선 합집합 (공유 선분 중복 제거)
  const seen = new Set(), segs = [];
  for (const m of MEASURES) {
    if (!state.angOn[m]) continue;
    for (const [a, b, dashed] of MEASURE_LINES[m]) {
      const key = a < b ? a + "|" + b : b + "|" + a;
      if (seen.has(key)) continue;
      seen.add(key); segs.push([a, b, dashed]);
    }
  }
  return segs;
}

// ---------- 관절 확대창 ----------
const JOINT_KO = { hip: "골반", knee: "무릎", ankle: "발목" };

function drawInset(cnv, im, side, joint, u = 1) { // u: 표기 스케일 (내보내기용 확대)
  const leg = im.legs[side];
  const ps = META[joint].landmarks.map(n => leg.points[n]).filter(Boolean);
  if (!ps.length) return;
  const xs = ps.map(p => p[0]), ys = ps.map(p => p[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const size = Math.max(Math.max(...xs) - Math.min(...xs),
                        Math.max(...ys) - Math.min(...ys), im.H * 0.04) * 1.8;
  // 셀 종횡비에 맞춰 크롭 영역 확장 (여백 없이 꽉 채움)
  const cw = cnv.width, ch = cnv.height, ar = cw / ch;
  const regW = size * Math.max(ar, 1), regH = size * Math.max(1 / ar, 1);
  const x0 = cx - regW / 2, y0 = cy - regH / 2;
  const s = cw / regW;
  const ctx = cnv.getContext("2d");
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cw, ch);
  ctx.drawImage(im.bitmap, x0, y0, regW, regH, 0, 0, cw, ch);
  const pts = leg.points;
  if (state.showAxes) {
    ctx.lineWidth = 1.5 * u; ctx.strokeStyle = "#4FB3C6"; ctx.globalAlpha = 0.85;
    for (const [a, b, dashed] of activeSegs()) {
      if (!pts[a] || !pts[b]) continue;
      ctx.setLineDash(dashed ? [7 * u, 6 * u] : []);
      ctx.beginPath();
      ctx.moveTo((pts[a][0] - x0) * s, (pts[a][1] - y0) * s);
      ctx.lineTo((pts[b][0] - x0) * s, (pts[b][1] - y0) * s);
      ctx.stroke();
    }
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  }
  if (state.showPts) {
    for (const n of PT_ORDER) {
      if (!pts[n]) continue;
      const x = (pts[n][0] - x0) * s, y = (pts[n][1] - y0) * s;
      if (x < -10 * u || x > cw + 10 * u || y < -10 * u || y > ch + 10 * u) continue;
      ctx.beginPath(); ctx.arc(x, y, 2.2 * u, 0, 7);
      ctx.fillStyle = leg.edited.has(n) ? "#FFD966" : "#38BDF8";
      ctx.fill();
      if (state.showNames) {
        ctx.font = `bold ${11 * u}px sans-serif`;
        ctx.lineWidth = 3 * u; ctx.strokeStyle = "rgba(0,0,0,.85)";
        ctx.strokeText(n, x + 6 * u, y - 6 * u);
        ctx.fillStyle = "#FFD966"; ctx.fillText(n, x + 6 * u, y - 6 * u);
      }
    }
  }
  const label = (side === "right" ? "오른쪽 " : "왼쪽 ") + JOINT_KO[joint];
  ctx.font = `bold ${11 * u}px sans-serif`;
  ctx.fillStyle = "rgba(16,23,26,.75)";
  ctx.fillRect(0, 0, ctx.measureText(label).width + 12 * u, 18 * u);
  ctx.fillStyle = "#E4ECEF"; ctx.fillText(label, 6 * u, 13 * u);
}

function renderInsets() { // 메인 창 분할: 오른쪽에 관절 확대창 그리드
  const box = $("jointviews");
  box.innerHTML = "";
  const im = state.images[state.cur];
  const views = [];
  if (im && im.legs && META)
    for (const joint of ["hip", "knee", "ankle"]) {
      if (!state.jointOn[joint]) continue;
      for (const side of ["right", "left"])
        if (state.sideOn[side] && im.legs[side]) views.push([side, joint]);
    }
  if (!views.length) { box.style.display = "none"; return; }
  box.style.display = "grid";
  box.style.width = views.length <= 3 ? "26%" : "42%";
  box.style.gridTemplateColumns = views.length <= 3 ? "1fr" : "1fr 1fr";
  const cells = views.map(([side, joint]) => {
    const c = document.createElement("canvas");
    c.className = "jv";
    box.appendChild(c);
    return [c, side, joint];
  });
  for (const [c, side, joint] of cells) { // 레이아웃 확정 후 실크기로 드로잉
    c.width = c.clientWidth; c.height = c.clientHeight;
    drawInset(c, im, side, joint);
  }
}

function renderAll() { renderInsets(); renderCanvas(); renderAngles(); }

// 줌/팬/점 드래그
let drag = null; // {mode:"pan"|"point", ...}
cv.addEventListener("wheel", e => {
  e.preventDefault();
  const z = state.zoom, f = e.deltaY < 0 ? 1.25 : 0.8;
  const r = cv.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  z.tx = mx - (mx - z.tx) * f;
  z.ty = my - (my - z.ty) * f;
  z.k = Math.min(Math.max(z.k * f, 0.5), 60);
  renderCanvas();
}, { passive: false });

function hitPoint(mx, my) {
  const im = state.images[state.cur];
  if (!im || !im.legs) return null;
  const { S, OX, OY } = viewTransform(im);
  let best = null, bd = 12; // 12px 이내
  for (const side of ["right", "left"]) {
    const leg = im.legs[side];
    if (!leg || !state.sideOn[side]) continue;
    for (const n of PT_ORDER) {
      if (!leg.points[n]) continue;
      const d = Math.hypot(leg.points[n][0]*S + OX - mx, leg.points[n][1]*S + OY - my);
      if (d < bd) { bd = d; best = { side, n }; }
    }
  }
  return best;
}

cv.addEventListener("pointerdown", e => {
  const r = cv.getBoundingClientRect();
  const hit = state.showPts ? hitPoint(e.clientX - r.left, e.clientY - r.top) : null;
  drag = hit ? { mode: "point", ...hit } : { mode: "pan", x: e.clientX, y: e.clientY };
  cv.setPointerCapture(e.pointerId);
});
cv.addEventListener("pointermove", e => {
  if (!drag) return;
  if (drag.mode === "pan") {
    state.zoom.tx += e.clientX - drag.x; state.zoom.ty += e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    renderCanvas();
  } else {
    const im = state.images[state.cur];
    const { S, OX, OY } = viewTransform(im);
    const r = cv.getBoundingClientRect();
    const leg = im.legs[drag.side];
    leg.points[drag.n] = [(e.clientX - r.left - OX) / S, (e.clientY - r.top - OY) / S];
    leg.edited.add(drag.n);
    leg.angles = calcAngles(leg.points, drag.side);
    renderAll();
  }
});
cv.addEventListener("pointerup", () => drag = null);
cv.addEventListener("dblclick", () => { state.zoom = { k: 1, tx: 0, ty: 0 }; renderCanvas(); });

// ---------- CSV ----------
function csvRows(im) {
  const rows = [];
  for (const side of ["right", "left"]) {
    const leg = im.legs && im.legs[side];
    if (!leg) continue;
    const row = [im.name, side];
    for (const m of MEASURES) row.push(leg.angles[m] !== undefined ? leg.angles[m].toFixed(2) : "");
    for (const n of PT_ORDER) {
      const p = leg.points[n];
      row.push(p ? p[0].toFixed(1) : "", p ? p[1].toFixed(1) : "");
    }
    row.push(leg.edited.size ? [...leg.edited].join(";") : "");
    rows.push(row);
  }
  return rows;
}

function downloadCsv(rows, fname) {
  const head = ["image", "side", ...MEASURES,
    ...PT_ORDER.flatMap(n => [n + "_x", n + "_y"]), "edited_points"];
  const txt = "﻿" + [head, ...rows].map(r => r.join(",")).join("\r\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([txt], { type: "text/csv" }));
  a.download = fname; a.click();
  URL.revokeObjectURL(a.href);
}

function updateCsvButtons() {
  const cur = state.images[state.cur];
  $("btn-csv").disabled = !(cur && cur.legs);
  $("btn-png").disabled = !(cur && cur.legs);
  const any = !state.images.some(im => im.legs);
  $("btn-csv-all").disabled = any;
  $("btn-csv-prev").disabled = any;
}

$("btn-csv").onclick = () => {
  const im = state.images[state.cur];
  downloadCsv(csvRows(im), im.name.replace(/\.[^.]+$/, "") + "_angles.csv");
};
$("btn-csv-all").onclick = () => {
  const rows = state.images.filter(im => im.legs).flatMap(csvRows);
  downloadCsv(rows, "xraypoint_angles.csv");
};

// ---------- 주석 입힌 이미지 저장 (메인 + 관절 확대창 합성) ----------
function exportAnnotated(im) { // 원본 해상도 PNG (현재 토글 상태 반영)
  const u = Math.max(im.W, im.H) / 1600; // 해상도 비례 표기 단위 (점이 뼈를 덮지 않게 소형)
  const main = makeCanvas(im.W, im.H);
  const ctx = main.getContext("2d");
  ctx.drawImage(im.bitmap, 0, 0);
  for (const side of ["right", "left"]) {
    const leg = im.legs[side];
    if (!leg || !state.sideOn[side]) continue;
    const pts = leg.points;
    if (state.showAxes) {
      ctx.lineWidth = 1.3 * u; ctx.strokeStyle = "#4FB3C6"; ctx.globalAlpha = 0.85;
      for (const [a, b, dashed] of activeSegs()) {
        if (!pts[a] || !pts[b]) continue;
        ctx.setLineDash(dashed ? [7 * u, 6 * u] : []);
        ctx.beginPath();
        ctx.moveTo(pts[a][0], pts[a][1]);
        ctx.lineTo(pts[b][0], pts[b][1]);
        ctx.stroke();
      }
      ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    if (state.showPts) {
      for (const n of PT_ORDER) {
        if (!pts[n]) continue;
        ctx.beginPath(); ctx.arc(pts[n][0], pts[n][1], 1.6 * u, 0, 7);
        ctx.fillStyle = leg.edited.has(n) ? "#FFD966" : "#38BDF8";
        ctx.fill();
        if (state.showNames) {
          ctx.font = `bold ${9 * u}px sans-serif`;
          ctx.lineWidth = 2.5 * u; ctx.strokeStyle = "rgba(0,0,0,.85)";
          ctx.strokeText(n, pts[n][0] + 5 * u, pts[n][1] - 5 * u);
          ctx.fillStyle = "#FFD966"; ctx.fillText(n, pts[n][0] + 5 * u, pts[n][1] - 5 * u);
        }
      }
    }
    // 각도 텍스트 블록 — 다리와 겹치지 않게 하단 모서리에 소형으로
    const lines = MEASURES.filter(m => state.angOn[m] && leg.angles[m] !== undefined)
      .map(m => `${m} ${fmtAng(leg.angles[m], m)}`);
    if (lines.length) {
      const fs = 11 * u, pad = 6 * u, lh = fs * 1.3;
      ctx.font = `bold ${fs}px sans-serif`;
      const bw = Math.max(...lines.map(t => ctx.measureText(t).width)) + pad * 2;
      const bh = (lines.length + 1) * lh + pad * 2;
      const bx = side === "right" ? pad : im.W - bw - pad;
      const by = im.H - bh - pad;
      ctx.fillStyle = "rgba(16,23,26,.72)";
      ctx.fillRect(bx, by, bw, bh);
      ctx.fillStyle = "#9FB2B9";
      ctx.fillText(side === "right" ? "오른쪽" : "왼쪽", bx + pad, by + pad + fs);
      ctx.fillStyle = "#E4ECEF";
      lines.forEach((t, i) =>
        ctx.fillText(t, bx + pad, by + pad + fs + (i + 1) * lh));
    }
  }

  // 화면과 같은 구성으로 관절 확대창을 오른쪽에 합성
  const views = [];
  for (const joint of ["hip", "knee", "ankle"]) {
    if (!state.jointOn[joint]) continue;
    for (const side of ["right", "left"])
      if (state.sideOn[side] && im.legs[side]) views.push([side, joint]);
  }
  let out = main;
  if (views.length) {
    const cols = views.length <= 3 ? 1 : 2;
    const rows = Math.ceil(views.length / cols);
    const g = Math.round(im.H * 0.008);
    const tile = Math.floor((im.H - g * (rows + 1)) / rows);
    const panelW = g + cols * (tile + g);
    out = makeCanvas(im.W + panelW, im.H);
    const octx = out.getContext("2d");
    octx.fillStyle = "#10171A"; octx.fillRect(0, 0, out.width, out.height);
    octx.drawImage(main, 0, 0);
    views.forEach(([side, joint], i) => {
      const c = makeCanvas(tile, tile);
      drawInset(c, im, side, joint, tile / 260);
      octx.drawImage(c, im.W + g + (i % cols) * (tile + g),
                     g + Math.floor(i / cols) * (tile + g));
    });
  }

  out.toBlob(blob => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = im.name.replace(/\.[^.]+$/, "") + "_annotated.png";
    a.click();
    URL.revokeObjectURL(a.href);
  }, "image/png");
}

$("btn-png").onclick = () => exportAnnotated(state.images[state.cur]);

// ---------- CSV 미리보기 ----------
const CSV_HEAD = ["image", "side", ...MEASURES,
  ...PT_ORDER.flatMap(n => [n + "_x", n + "_y"]), "edited_points"];

function openCsvPreview() {
  const rows = state.images.filter(im => im.legs).flatMap(csvRows);
  let h = "<table><tr>" + CSV_HEAD.map(c => `<th>${c}</th>`).join("") + "</tr>";
  for (const r of rows)
    h += "<tr>" + r.map(v => `<td>${v === "" ? "—" : v}</td>`).join("") + "</tr>";
  h += "</table>";
  $("csvtable").innerHTML = h;
  $("csvcount").textContent = `${rows.length}행 (다리 단위) · ${state.images.filter(im => im.legs).length}장`;
  $("csvmodal").style.display = "flex";
}
$("btn-csv-prev").onclick = openCsvPreview;
$("csv-close").onclick = () => $("csvmodal").style.display = "none";
$("csvmodal").addEventListener("click", e => {
  if (e.target === $("csvmodal")) $("csvmodal").style.display = "none";
});
$("csv-dl").onclick = () => $("btn-csv-all").click();

// ---------- 토글/입력 바인딩 ----------
function toggleBtn(id, key) {
  $(id).classList.toggle("on", state[key]);
}
$("btn-pts").onclick = () => { state.showPts = !state.showPts; toggleBtn("btn-pts","showPts"); renderAll(); };
$("btn-axes").onclick = () => { state.showAxes = !state.showAxes; toggleBtn("btn-axes","showAxes"); renderAll(); };
$("btn-names").onclick = () => { state.showNames = !state.showNames; toggleBtn("btn-names","showNames"); renderAll(); };
for (const joint of ["hip", "knee", "ankle"]) {
  $("btn-joint-" + joint).onclick = () => {
    state.jointOn[joint] = !state.jointOn[joint];
    $("btn-joint-" + joint).classList.toggle("on", state.jointOn[joint]);
    renderAll(); // 분할 폭이 바뀌므로 메인 캔버스도 다시 그림
  };
}
MEASURES.forEach(m => {
  $("btn-ang-" + m).onclick = () => {
    state.angOn[m] = !state.angOn[m];
    $("btn-ang-" + m).classList.toggle("on", state.angOn[m]);
    renderAll();
  };
});
for (const side of ["right", "left"]) {
  $("btn-side-" + side).onclick = () => {
    state.sideOn[side] = !state.sideOn[side];
    $("btn-side-" + side).classList.toggle("on", state.sideOn[side]);
    renderAll();
  };
}
document.addEventListener("keydown", e => {
  if (e.key === "Escape") { $("csvmodal").style.display = "none"; return; }
  if (e.key >= "1" && e.key <= "5") { $("btn-ang-" + MEASURES[e.key - 1]).click(); return; }
  if (e.key === "6") { $("btn-side-right").click(); return; }
  if (e.key === "7") { $("btn-side-left").click(); return; }
  if (e.key === "8") { $("btn-joint-hip").click(); return; }
  if (e.key === "9") { $("btn-joint-knee").click(); return; }
  if (e.key === "0") { $("btn-joint-ankle").click(); return; }
  if (e.key.toLowerCase() === "p") $("btn-pts").click();
  if (e.key.toLowerCase() === "a") $("btn-axes").click();
  if (e.key.toLowerCase() === "l") $("btn-names").click();
});
$("file").addEventListener("change", e => addFiles([...e.target.files]));
window.addEventListener("dragover", e => { e.preventDefault(); $("drop").style.display = "flex"; });
window.addEventListener("dragleave", e => { if (!e.relatedTarget) $("drop").style.display = "none"; });
window.addEventListener("drop", e => {
  e.preventDefault(); $("drop").style.display = "none";
  addFiles([...e.dataTransfer.files].filter(f => f.type.startsWith("image/")));
});
window.addEventListener("resize", renderAll);

loadModels().catch(e => setStatus("모델 로딩 실패: " + e.message));
