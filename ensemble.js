// Shared by the browser and the numerical/regression tests. No image upload.
(function (root) {
  "use strict";
  function same(a, b) { return JSON.stringify(a) === JSON.stringify(b); }

  function validateMetadata(meta) {
    const base = meta?.knee;
    const spec = base?.ensemble;
    if (spec?.method !== "heatmap_mean" || !Array.isArray(spec.members) ||
        spec.members.length !== 2 || new Set(spec.members).size !== 2 ||
        !spec.members.includes("knee")) {
      throw new Error("무릎 앙상블 모델 구성이 올바르지 않습니다");
    }
    for (const name of spec.members) {
      const m = meta[name];
      if (!m || !same(m.input, base.input) || !same(m.landmarks, base.landmarks) ||
          m.channels !== base.channels || m.channels !== m.landmarks.length ||
          m.sigma !== base.sigma || m.roi?.ratio !== base.roi?.ratio ||
          m.preprocessing !== "grayscale_0_1" || m.input.length !== 4 ||
          m.input[0] !== 1 || m.input[1] !== 1 || m.input[2] !== m.input[3]) {
        throw new Error(`무릎 앙상블 입력·랜드마크 불일치: ${name}`);
      }
    }
    return [...new Set(["stage1", "hip", ...spec.members, "ankle"])];
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

  const api = { validateMetadata, predictHeatmaps };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.XrayEnsemble = api;
})(globalThis);
