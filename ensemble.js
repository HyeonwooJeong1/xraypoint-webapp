// Shared by the browser and the numerical/regression tests. No image upload.
(function (root) {
  "use strict";
  function same(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
  const JOINTS = ["hip", "knee", "ankle"];
  function compatibilityReason(meta, names) {
    if (!names.length) return "모델을 하나 이상 선택하세요";
    const base = meta[names[0]];
    for (const name of names) {
      const m = meta[name];
      if (!m || !base) return "모델 정보가 없습니다";
      if (!same(m.input, base.input)) return "입력 크기가 달라 함께 선택할 수 없습니다";
      if (!same(m.landmarks, base.landmarks) || m.channels !== base.channels)
        return "랜드마크 정의가 달라 함께 선택할 수 없습니다";
      if (!same(m.sigma, base.sigma)) return "히트맵 σ가 달라 함께 선택할 수 없습니다";
      if (m.preprocessing !== base.preprocessing || m.roi?.ratio !== base.roi?.ratio)
        return "전처리·ROI 조건이 달라 함께 선택할 수 없습니다";
    }
    return "";
  }
  function selectedModels(meta, choice) {
    if (typeof choice === "string") {
      const parts = choice.split("|");
      if (parts.length !== JOINTS.length) throw new Error("잘못된 모델 조합입니다");
      choice = Object.fromEntries(JOINTS.map((j,i) => [j,parts[i].split("+")]));
    }
    const out = {};
    for (const joint of JOINTS) {
      const names = choice?.[joint];
      if (!Array.isArray(names) || !names.length || new Set(names).size !== names.length ||
          names.some(n => !meta.catalog.models.includes(n) || meta[n]?.joint !== joint))
        throw new Error("각 관절에서 사용 가능한 모델을 하나 이상 선택하세요");
      out[joint] = meta.catalog.models.filter(n => names.includes(n));
      const reason = compatibilityReason(meta,out[joint]);
      if (reason) throw new Error(reason);
    }
    return out;
  }
  function selectionKey(meta, choice) {
    const selected = selectedModels(meta, choice);
    return JOINTS.map(j=>selected[j].join("+")).join("|");
  }
  function modelAssets(m) {
    return [{path:m.file, sha256:m.sha256, bytes:m.bytes}, ...(m.external_data || [])];
  }
  function validAssets(m) {
    const assets = modelAssets(m);
    return new Set(assets.map(a=>a.path)).size === assets.length && assets.every(a =>
      typeof a.path === "string" && /^[a-zA-Z0-9_.-]+$/.test(a.path) && a.path !== "." && a.path !== ".." &&
      /^[a-f0-9]{64}$/.test(a.sha256) && Number.isSafeInteger(a.bytes) && a.bytes > 0) &&
      assets.reduce((n,a)=>n+a.bytes,0) === m.download_bytes;
  }

  function validateMetadata(meta) {
      const ids = meta?.catalog?.models;
      if (!Array.isArray(ids) || new Set(ids).size !== ids.length || !ids.length)
        throw new Error("모델 목록 구성이 올바르지 않습니다");
      for (const id of ids) {
        const m = meta[id];
        if (!m || !JOINTS.includes(m.joint) || !same(m.training_split,[.9,.1,0]) ||
            m.data_generation !== "fixed-labels-90-10" || m.channels !== m.landmarks?.length ||
            m.preprocessing !== "grayscale_0_1" || m.input?.length !== 4 ||
            m.input[0] !== 1 || m.input[1] !== 1 || m.input[2] !== m.input[3] ||
            !m.input.every(n=>Number.isSafeInteger(n) && n>0) ||
            !["efficientnet-b0","tu-hrnet_w32"].includes(m.encoder) || m.source_checkpoint !== id ||
            !Array.isArray(m.landmarks) || new Set(m.landmarks).size !== m.landmarks.length ||
            !Number.isSafeInteger(m.parameter_count) || m.parameter_count <= 0 ||
            !Number.isFinite(m.roi?.ratio) || m.roi.ratio <= 0 ||
            !validAssets(m)) {
          throw new Error(`잘못된 모델 목록 항목: ${id}`);
        }
      }
      const defaults = selectedModels(meta, meta.catalog.defaults);
      if (!same(meta.stage1?.training_split,[.9,.1,0]) || meta.stage1?.source_checkpoint !== "stage1_90" ||
          meta.stage1.data_generation !== "fixed-labels-90-10" ||
          !same(meta.stage1.input,[1,1,512,256]) || meta.stage1.channels !== 3 ||
          !same(meta.stage1.landmarks,["hip_center","knee_center","ankle_center"]) ||
          !["training","ready"].includes(meta.stage1.status) ||
          (meta.stage1.status === "ready" && !validAssets(meta.stage1)))
        throw new Error("최신 stage1 모델 정보가 필요합니다");
      return ["stage1",...JOINTS.flatMap(j=>defaults[j])];
  }

  // Return only a complete benchmark tied to the exact deployed checkpoints.
  function modelErrorCases(meta, report, joint) {
    if (report?.schema !== 1 || report.metric !== "gt_p5_p6_normalized_error_pct" ||
        report.model_release !== meta.release ||
        report.stage1_checkpoint_sha256 !== meta.stage1.checkpoint_sha256 ||
        !Number.isInteger(report.dataset?.legs) || report.dataset.legs <= 0) return null;
    const names = meta.catalog.models.filter(n => meta[n].joint === joint);
    if (names.length !== 2 || names.some(n => report.model_checkpoint_sha256?.[n] !== meta[n].checkpoint_sha256)) return null;
    const wanted = [[names[0]], [names[1]], names], cases = report.groups?.[joint];
    if (!Array.isArray(cases) || cases.length !== wanted.length) return null;
    const points = meta[names[0]].landmarks;
    for (let i=0; i<wanted.length; i++) {
      const c=cases[i];
      if (!same(c?.models,wanted[i]) || c.n_legs !== report.dataset.legs ||
          !Number.isFinite(c.mean_pct) || c.mean_pct < 0 ||
          !same(Object.keys(c.points || {}),points) ||
          points.some(p => !Number.isFinite(c.points[p].mean_pct) || c.points[p].mean_pct < 0 ||
                           c.points[p].n !== report.dataset.legs)) return null;
    }
    return cases;
  }

  // Only model assets enter Cache Storage. Images and predictions stay in memory.
  function createAssetStore({baseUrl, cacheStorage = root.caches, fetchImpl = root.fetch.bind(root),
                              cryptoImpl = root.crypto, onStorageError = () => {}}) {
    let cachePromise;
    const cache = () => cachePromise ||= (async () => {
      try { return await cacheStorage?.open("xraypoint-model-assets-v1") || null; }
      catch (e) { onStorageError(e); return null; }
    })();
    const url = a => new URL(`models/${a.path}?sha256=${a.sha256}`, baseUrl).href;
    async function valid(bytes, a) {
      if (bytes.byteLength !== a.bytes) return false;
      const hash = await cryptoImpl.subtle.digest("SHA-256", bytes);
      return [...new Uint8Array(hash)].map(b => b.toString(16).padStart(2,"0")).join("") === a.sha256;
    }
    async function read(a, progress = () => {}) {
      const c = await cache(), key = url(a);
      let cached;
      try { cached = await c?.match(key); } catch (e) { onStorageError(e); }
      if (cached) {
        const bytes = new Uint8Array(await cached.arrayBuffer());
        if (await valid(bytes,a)) { progress(bytes.length); return {bytes,persistent:true}; }
        try { await c.delete(key); } catch(e) { onStorageError(e); }
      }
      const response = await fetchImpl(key, {cache:"no-store"});
      if (!response.ok) throw new Error(`모델 다운로드 실패 (HTTP ${response.status})`);
      const bytes = new Uint8Array(a.bytes), reader = response.body.getReader();
      let offset = 0;
      try {
        for (;;) {
          const {done,value} = await reader.read();
          if (done) break;
          if (offset+value.length > bytes.length) throw new Error("모델 파일 크기 불일치");
          bytes.set(value,offset); offset += value.length; progress(offset);
        }
      } catch (e) { await reader.cancel().catch(()=>{}); throw e; }
      finally { reader.releaseLock(); }
      if (offset !== a.bytes || !await valid(bytes,a)) throw new Error("모델 파일 무결성 검사 실패 — 다시 다운로드하세요");
      let persistent = false;
      if (c) {
        try {
          await c.put(key,new Response(bytes,{headers:{"X-Xray-SHA256":a.sha256,"Content-Length":String(a.bytes)}}));
          persistent = true;
        } catch (e) { onStorageError(e); }
      }
      return {bytes,persistent};
    }
    async function has(m) {
      const c = await cache();
      if (!c) return false;
      try {
        for (const a of modelAssets(m)) {
          const r = await c.match(url(a));
          if (!r || r.headers.get("X-Xray-SHA256") !== a.sha256 ||
              Number(r.headers.get("Content-Length")) !== a.bytes) return false;
        }
      } catch(e) { onStorageError(e); return false; }
      return true;
    }
    async function download(m, progress) {
      let complete = 0, persisted = true;
      for (const a of modelAssets(m)) {
        const result = await read(a, n => progress?.(complete+n));
        complete += a.bytes; persisted &&= result.persistent;
      }
      if (!persisted) throw new Error("브라우저에 저장할 수 없습니다. 저장 공간·사이트 저장 권한을 확인하고 재시도하세요.");
    }
    async function prune(models) {
      const c = await cache(); if (!c) return;
      const keep = new Set(models.flatMap(modelAssets).map(url));
      const prefix = new URL('models/',baseUrl).href;
      try {
        for (const request of await c.keys()) {
          if (request.url.startsWith(prefix) && !keep.has(request.url)) await c.delete(request);
        }
      } catch(e) { onStorageError(e); }
    }
    return {read,has,download,prune};
  }

  // Run sequentially on the SAME input tensor. Average raw heatmaps before DARK.
  // Failure of any member fails the measurement; never silently use a single model.
  async function predictHeatmaps(sessions, names, tensor, expectedShape) {
    let sum = null;
    try {
      if (!names.length) throw new Error("추론 모델이 없습니다");
      for (const name of names) {
        if (!sessions[name]) throw new Error(`모델이 준비되지 않았습니다: ${name}`);
        let outputs;
        try {
          outputs = await sessions[name].run({ image: tensor });
          const hm = outputs.heatmaps;
          if (!hm || hm.type !== "float32" || !same(hm.dims, expectedShape) ||
              hm.data.length !== expectedShape.reduce((a, b) => a * b, 1)) {
            throw new Error(`히트맵 크기 불일치: ${name}`);
          }
          if (sum === null) sum = new Float32Array(hm.data.length);
          for (let i = 0; i < sum.length; i++) {
            if (!Number.isFinite(hm.data[i])) throw new Error(`유효하지 않은 히트맵: ${name}`);
            sum[i] += hm.data[i];
          }
        } finally {
          if (outputs) for (const t of new Set(Object.values(outputs))) t.dispose?.();
        }
      }
      for (let i = 0; i < sum.length; i++) {
        sum[i] /= names.length;
        if (!Number.isFinite(sum[i])) throw new Error("유효하지 않은 앙상블 결과");
      }
      return { data: sum, dims: expectedShape };
    } finally {
      tensor.dispose?.();
    }
  }

  const api = { modelErrorCases, validateMetadata, predictHeatmaps, JOINTS, selectedModels,
    compatibilityReason, selectionKey, modelAssets, createAssetStore };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.XrayEnsemble = api;
})(globalThis);
