(() => {
  "use strict";

  const MODEL_URL = "./models/tfjs_model/model.json";
  const LABELS_URL = "./models/tflite/label.txt";
  const META_URL = "./models/model_meta.json";
  const FALLBACK_LABELS = [
    "Baked Potato",
    "Burger",
    "Crispy Chicken",
    "Donut",
    "Fries",
    "Pizza",
  ];
  const INPUT_SIZE = 224;
  const MAX_FILE_BYTES = 15 * 1024 * 1024;
  const ALLOWED_TYPES = new Set([
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
  ]);

  const els = {
    modelStatus: document.getElementById("model-status"),
    modelStatusText: document.getElementById("model-status-text"),
    dropzone: document.getElementById("dropzone"),
    fileInput: document.getElementById("file-input"),
    dropzoneEmpty: document.getElementById("dropzone-empty"),
    previewWrapper: document.getElementById("preview-wrapper"),
    previewImage: document.getElementById("preview-image"),
    fileControls: document.getElementById("file-controls"),
    fileInfo: document.getElementById("file-info"),
    changeBtn: document.getElementById("change-btn"),
    clearBtn: document.getElementById("clear-btn"),
    inputError: document.getElementById("input-error"),
    inferenceTime: document.getElementById("inference-time"),
    resultEmpty: document.getElementById("result-empty"),
    resultLoading: document.getElementById("result-loading"),
    loadingTitle: document.getElementById("loading-title"),
    loadingDesc: document.getElementById("loading-desc"),
    resultError: document.getElementById("result-error"),
    errorTitle: document.getElementById("error-title"),
    errorDesc: document.getElementById("error-desc"),
    retryModelBtn: document.getElementById("retry-model-btn"),
    resultContent: document.getElementById("result-content"),
    topPredictionBox: document.getElementById("top-prediction-box"),
    topLabel: document.getElementById("top-label"),
    topClass: document.getElementById("top-class"),
    topConfidence: document.getElementById("top-confidence"),
    confidenceNote: document.getElementById("confidence-note"),
    distributionTitle: document.getElementById("distribution-title"),
    probabilityList: document.getElementById("probability-list"),
    footerArch: document.getElementById("footer-arch"),
  };

  let model = null;
  let dualModel = null;
  let labels = FALLBACK_LABELS.slice();
  let modelMeta = {
    version: "v1",
    preprocessing: "rescale_0_1",
    crop: "center_square",
    other_class: null,
    ood: {
      enabled: true,
      embedding_layer: null,
      min_cosine_similarity: null,
      min_confidence: 0.65,
      max_entropy: 1.05,
      class_centroids: {},
    },
  };
  let modelState = "loading";
  let currentObjectUrl = null;
  let pendingInference = false;

  function setModelStatus(state, message) {
    modelState = state;
    els.modelStatus.classList.remove("status-loading", "status-ready", "status-error");
    els.modelStatus.classList.add(`status-${state}`);
    els.modelStatusText.textContent = message;
  }

  function showResultPanel(panelName) {
    els.resultEmpty.hidden = panelName !== "empty";
    els.resultLoading.hidden = panelName !== "loading";
    els.resultError.hidden = panelName !== "error";
    els.resultContent.hidden = panelName !== "content";
  }

  function setInputError(message) {
    if (!message) {
      els.inputError.hidden = true;
      els.inputError.textContent = "";
      return;
    }
    els.inputError.textContent = message;
    els.inputError.hidden = false;
  }

  function showResultError(title, desc, showRetry = false) {
    els.errorTitle.textContent = title;
    els.errorDesc.textContent = desc;
    els.retryModelBtn.hidden = !showRetry;
    els.inferenceTime.textContent = "";
    showResultPanel("error");
  }

  async function fetchLabels() {
    try {
      const response = await fetch(LABELS_URL);
      if (!response.ok) return FALLBACK_LABELS.slice();
      const text = await response.text();
      const parsed = text
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);
      return parsed.length > 0 ? parsed : FALLBACK_LABELS.slice();
    } catch {
      return FALLBACK_LABELS.slice();
    }
  }

  async function fetchModelMeta() {
    try {
      const response = await fetch(META_URL);
      if (!response.ok) return null;
      return await response.json();
    } catch {
      return null;
    }
  }

  function computeShannonEntropy(scores) {
    let entropy = 0;
    for (let i = 0; i < scores.length; i += 1) {
      const p = Math.max(Number(scores[i] || 0), 1e-12);
      entropy -= p * Math.log(p);
    }
    return entropy;
  }

  function computeMaxCosineSimilarity(embedding) {
    const centroids = (modelMeta.ood && modelMeta.ood.class_centroids) || {};
    const names = Object.keys(centroids);
    if (!embedding || names.length === 0) return null;

    let normSq = 0;
    for (let i = 0; i < embedding.length; i += 1) {
      normSq += embedding[i] * embedding[i];
    }
    const norm = Math.sqrt(Math.max(normSq, 1e-12));

    let bestSim = -1;
    for (const name of names) {
      const centroid = centroids[name];
      if (!Array.isArray(centroid) || centroid.length !== embedding.length) continue;
      let dot = 0;
      for (let i = 0; i < embedding.length; i += 1) {
        dot += (embedding[i] / norm) * centroid[i];
      }
      if (dot > bestSim) bestSim = dot;
    }
    return bestSim > -1 ? bestSim : null;
  }

  function evaluateOod(bestItem, entropy, maxCosineSim) {
    const otherClass = modelMeta.other_class;
    if (otherClass && bestItem.label === otherClass) {
      return {
        isOod: true,
        headline: "Di Luar 6 Kategori Fast Food",
        note: `Model mengenali foto ini sebagai kelas '${otherClass}' (${(bestItem.probability * 100).toFixed(1)}%), bukan bagian dari 6 kategori fast food target.`,
      };
    }

    const oodCfg = modelMeta.ood || {};
    if (!oodCfg.enabled) {
      return { isOod: false, headline: bestItem.label, note: null };
    }

    if (
      typeof maxCosineSim === "number" &&
      typeof oodCfg.min_cosine_similarity === "number" &&
      maxCosineSim < oodCfg.min_cosine_similarity
    ) {
      return {
        isOod: true,
        headline: "Di Luar 6 Kategori Fast Food",
        note: `Kemiripan fitur visual (${(maxCosineSim * 100).toFixed(1)}%) berada di bawah ambang model (${(oodCfg.min_cosine_similarity * 100).toFixed(1)}%). Foto ini kemungkinan bukan salah satu dari 6 kategori fast food.`,
      };
    }

    if (typeof oodCfg.min_confidence === "number" && bestItem.probability < oodCfg.min_confidence) {
      return {
        isOod: true,
        headline: "Tidak Yakin / Di Luar Kategori",
        note: `Keyakinan tertinggi hanya ${(bestItem.probability * 100).toFixed(1)}% (di bawah ambang ${(oodCfg.min_confidence * 100).toFixed(0)}%). Pastikan foto menampilkan salah satu dari 6 kategori dengan jelas.`,
      };
    }

    if (typeof oodCfg.max_entropy === "number" && entropy > oodCfg.max_entropy) {
      return {
        isOod: true,
        headline: "Tidak Yakin / Di Luar Kategori",
        note: `Distribusi skor terlalu menyebar antar kelas (entropi ${entropy.toFixed(2)}). Foto ini kemungkinan berada di luar 6 kategori fast food.`,
      };
    }

    return { isOod: false, headline: bestItem.label, note: null };
  }

  function buildInputTensor(imgElement) {
    let pixels = tf.browser.fromPixels(imgElement, 3);
    if (modelMeta.crop === "center_square") {
      const [h, w] = pixels.shape;
      if (h !== w) {
        const side = Math.min(h, w);
        const top = Math.floor((h - side) / 2);
        const left = Math.floor((w - side) / 2);
        pixels = pixels.slice([top, left, 0], [side, side, 3]);
      }
    }
    const resized = tf.image.resizeBilinear(pixels, [INPUT_SIZE, INPUT_SIZE]).toFloat();
    const scaled =
      modelMeta.preprocessing === "mobilenet_v2"
        ? resized.div(127.5).sub(1.0)
        : resized.div(255.0);
    return scaled.expandDims(0);
  }

  async function loadModel() {
    if (window.location.protocol === "file:") {
      setModelStatus("error", "Perlu server HTTP lokal");
      showResultError(
        "Model tidak dapat dimuat lewat protokol file://",
        "Browser memblokir pembacaan berkas model lokal secara langsung. Jalankan perintah 'python -m http.server 8000' di folder proyek, lalu buka http://localhost:8000.",
        false
      );
      return;
    }

    if (typeof tf === "undefined") {
      setModelStatus("error", "TensorFlow.js gagal dimuat");
      showResultError(
        "Library TensorFlow.js tidak tersedia",
        "Periksa koneksi internet Anda karena skrip @tensorflow/tfjs dimuat dari CDN, lalu klik Muat Ulang Model.",
        true
      );
      return;
    }

    setModelStatus("loading", "Memuat model TensorFlow.js...");
    els.retryModelBtn.hidden = true;

    try {
      const [loadedLabels, fetchedMeta, loadedModel] = await Promise.all([
        fetchLabels(),
        fetchModelMeta(),
        tf.loadLayersModel(MODEL_URL),
      ]);

      if (fetchedMeta) {
        modelMeta = { ...modelMeta, ...fetchedMeta };
      }
      labels =
        Array.isArray(modelMeta.labels) && modelMeta.labels.length > 0
          ? modelMeta.labels
          : loadedLabels;
      model = loadedModel;
      dualModel = null;

      const embLayerName = modelMeta.ood && modelMeta.ood.embedding_layer;
      if (embLayerName) {
        try {
          const embLayer = model.getLayer(embLayerName);
          dualModel = tf.model({
            inputs: model.inputs,
            outputs: [embLayer.output, model.output],
          });
        } catch (err) {
          console.warn(`Embedding layer '${embLayerName}' unavailable for OOD cosine check:`, err);
        }
      }

      // Warm up shaders on first load so user inference feels immediate.
      tf.tidy(() => {
        const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
        if (dualModel) {
          dualModel.predict(dummy);
        } else {
          model.predict(dummy);
        }
      });

      const scaleLabel = modelMeta.preprocessing === "mobilenet_v2" ? "[-1, 1]" : "[0, 1]";
      const oodTag = dualModel ? " + OOD Embedding Gate" : " + OOD Threshold";
      if (els.footerArch) {
        els.footerArch.innerHTML = `<strong>Arsitektur (${modelMeta.version || "v1"}):</strong> MobileNetV2 (ImageNet) + Conv2D head, center-crop 224x224 RGB (skala ${scaleLabel}${oodTag}).`;
      }
      if (els.distributionTitle) {
        els.distributionTitle.textContent = `Distribusi Probabilitas (${labels.length} Kelas)`;
      }

      setModelStatus("ready", "Model siap digunakan");

      if (pendingInference && els.previewImage.complete && els.previewImage.naturalWidth > 0) {
        pendingInference = false;
        await runInference(els.previewImage);
      } else if (!els.previewWrapper.hidden) {
        await runInference(els.previewImage);
      } else {
        showResultPanel("empty");
      }
    } catch (error) {
      console.error("Failed to load TFJS model:", error);
      setModelStatus("error", "Gagal memuat model");
      showResultError(
        "Gagal memuat model klasifikasi",
        `Berkas ${MODEL_URL} tidak dapat dimuat (${error.message}). Pastikan folder models/tfjs_model/ lengkap dan diakses melalui server HTTP.`,
        true
      );
    }
  }

  function renderPredictions(scores, embedding, durationMs) {
    const ranked = labels
      .map((label, idx) => ({
        label,
        probability: Number(scores[idx] || 0),
      }))
      .sort((a, b) => b.probability - a.probability);

    const best = ranked[0];
    const entropy = computeShannonEntropy(scores);
    const maxCosineSim = computeMaxCosineSimilarity(embedding);
    const oodResult = evaluateOod(best, entropy, maxCosineSim);

    els.topPredictionBox.classList.toggle("is-ood", oodResult.isOod);
    els.topLabel.textContent = oodResult.isOod ? "Deteksi di Luar Kategori (OOD)" : "Prediksi Teratas";
    els.topClass.textContent = oodResult.headline;
    els.topConfidence.textContent = oodResult.isOod
      ? `Skor maks ${(best.probability * 100).toFixed(1)}%`
      : `${(best.probability * 100).toFixed(2)}%`;
    els.inferenceTime.textContent = `${Math.round(durationMs)} ms`;

    if (oodResult.note) {
      els.confidenceNote.textContent = oodResult.note;
      els.confidenceNote.hidden = false;
    } else {
      els.confidenceNote.hidden = true;
    }

    els.probabilityList.textContent = "";
    ranked.forEach((item, index) => {
      const pct = (item.probability * 100).toFixed(2);
      const li = document.createElement("li");
      li.className = !oodResult.isOod && index === 0 ? "prob-item is-top" : "prob-item";

      const row = document.createElement("div");
      row.className = "prob-row";

      const nameEl = document.createElement("span");
      nameEl.className = "prob-name";
      nameEl.textContent = item.label;

      const valueEl = document.createElement("span");
      valueEl.className = "prob-value";
      valueEl.textContent = `${pct}%`;

      row.append(nameEl, valueEl);

      const track = document.createElement("div");
      track.className = "prob-track";

      const fill = document.createElement("div");
      fill.className = "prob-fill";
      const fillPct = item.probability >= 0.0005 ? Math.max(item.probability * 100, 0.8) : 0;
      fill.style.width = `${fillPct}%`;

      track.append(fill);
      li.append(row, track);
      els.probabilityList.append(li);
    });

    showResultPanel("content");
  }

  async function runInference(imgElement) {
    if (modelState === "loading") {
      pendingInference = true;
      els.loadingTitle.textContent = "Menunggu model selesai dimuat...";
      els.loadingDesc.textContent =
        "Foto sudah siap. Prediksi akan berjalan otomatis segera setelah bobot model (~14,7 MB) selesai dimuat.";
      showResultPanel("loading");
      return;
    }

    if (!model || modelState === "error") {
      showResultError(
        "Model belum siap",
        "Muat ulang model terlebih dahulu sebelum menjalankan klasifikasi gambar.",
        true
      );
      return;
    }

    els.loadingTitle.textContent = "Menjalankan prediksi...";
    els.loadingDesc.textContent =
      "Melakukan center-crop 224x224 dan menghitung probabilitas serta pemeriksaan OOD.";
    showResultPanel("loading");

    // Yield one frame so the loading indicator paints before synchronous tensor work.
    await new Promise((resolve) => requestAnimationFrame(resolve));

    try {
      const start = performance.now();
      let probabilities;
      let embedding = null;

      if (dualModel) {
        const [embTensor, probTensor] = tf.tidy(() => {
          const inputTensor = buildInputTensor(imgElement);
          return dualModel.predict(inputTensor);
        });
        [embedding, probabilities] = await Promise.all([embTensor.data(), probTensor.data()]);
        embTensor.dispose();
        probTensor.dispose();
      } else {
        const probTensor = tf.tidy(() => {
          const inputTensor = buildInputTensor(imgElement);
          return model.predict(inputTensor);
        });
        probabilities = await probTensor.data();
        probTensor.dispose();
      }

      const durationMs = performance.now() - start;
      renderPredictions(probabilities, embedding, durationMs);
    } catch (error) {
      console.error("Inference error:", error);
      showResultError(
        "Gagal memproses gambar",
        `Terjadi kesalahan saat menjalankan prediksi: ${error.message}. Coba unggah foto lain berformat JPG atau PNG.`,
        false
      );
    }
  }

  function validateFile(file) {
    if (!file) return "Tidak ada berkas yang dipilih.";
    if (file.type && !ALLOWED_TYPES.has(file.type)) {
      return `Format berkas (${file.type}) tidak didukung. Gunakan gambar JPG, PNG, WebP, atau BMP.`;
    }
    if (!file.type && !/\.(jpe?g|png|webp|bmp)$/i.test(file.name)) {
      return "Format berkas tidak dikenali. Gunakan gambar JPG, PNG, WebP, atau BMP.";
    }
    if (file.size > MAX_FILE_BYTES) {
      return "Ukuran berkas melebihi batas 15 MB. Kompres atau pilih foto yang lebih kecil.";
    }
    return null;
  }

  function handleSelectedFile(file) {
    const validationError = validateFile(file);
    if (validationError) {
      setInputError(validationError);
      return;
    }

    setInputError(null);
    if (currentObjectUrl) {
      URL.revokeObjectURL(currentObjectUrl);
    }
    currentObjectUrl = URL.createObjectURL(file);

    els.previewImage.onload = () => {
      els.dropzoneEmpty.hidden = true;
      els.previewWrapper.hidden = false;
      els.dropzone.classList.add("has-image");
      els.fileControls.hidden = false;
      const sizeKb = (file.size / 1024).toFixed(1);
      els.fileInfo.textContent = `${file.name || "gambar-clipboard.png"} (${els.previewImage.naturalWidth}x${els.previewImage.naturalHeight} px, ${sizeKb} KB)`;
      runInference(els.previewImage);
    };

    els.previewImage.onerror = () => {
      setInputError("Berkas tidak dapat dibaca sebagai gambar yang valid. Pilih berkas gambar lain.");
      clearImage();
    };

    els.previewImage.src = currentObjectUrl;
  }

  function clearImage() {
    if (currentObjectUrl) {
      URL.revokeObjectURL(currentObjectUrl);
      currentObjectUrl = null;
    }
    pendingInference = false;
    els.fileInput.value = "";
    els.previewImage.removeAttribute("src");
    els.previewWrapper.hidden = true;
    els.dropzoneEmpty.hidden = false;
    els.dropzone.classList.remove("has-image");
    els.fileControls.hidden = true;
    els.fileInfo.textContent = "";
    els.inferenceTime.textContent = "";
    setInputError(null);
    if (modelState !== "error") {
      showResultPanel("empty");
    }
  }

  function openFileDialog() {
    els.fileInput.click();
  }

  els.dropzone.addEventListener("click", openFileDialog);
  els.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openFileDialog();
    }
  });

  els.fileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) handleSelectedFile(file);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    els.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      els.dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    els.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      els.dropzone.classList.remove("is-dragover");
    });
  });

  els.dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) handleSelectedFile(file);
  });

  window.addEventListener("paste", (event) => {
    const items = event.clipboardData && event.clipboardData.items;
    if (!items) return;
    for (const item of items) {
      if (item.type && item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          handleSelectedFile(file);
          break;
        }
      }
    }
  });

  els.changeBtn.addEventListener("click", openFileDialog);
  els.clearBtn.addEventListener("click", clearImage);
  els.retryModelBtn.addEventListener("click", loadModel);

  loadModel();
})();
