(() => {
  const $ = (id) => document.getElementById(id);

  const SVG_EDIT = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>`;
  const SVG_TRASH = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>`;
  const SVG_VOL = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>`;

  const state = {
    job: null,
    cues: [],
    activeIdx: -1,
    loopCue: false,
    shadowMode: false,
    shadowPausedIdx: -1,
    subOffset: 0.0,
    pollTimer: null,
    rafId: null,
    vocab: [],
    lastLookup: null,
    selPayload: null,
    editingIdx: -1,
  };

  function escHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function fmtTime(sec) {
    sec = Math.max(0, Math.floor(Number(sec) || 0));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    const mm = String(m).padStart(2, "0");
    const ss = String(s).padStart(2, "0");
    return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
  }

  function fmtSrtTime(sec) {
    const ms = Math.max(0, Math.round((Number(sec) || 0) * 1000));
    return `${String(Math.floor(ms / 3600000)).padStart(2, "0")}:${String(Math.floor((ms % 3600000) / 60000)).padStart(2, "0")}:${String(Math.floor((ms % 60000) / 1000)).padStart(2, "0")},${String(ms % 1000).padStart(3, "0")}`;
  }

  let toastTimer = null;
  function toast(msg) {
    const el = $("toast");
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
  }

  function getLlmConfig() {
    return {
      llm_url: (localStorage.getItem("ms_llm_url") || "").trim(),
      llm_model: (localStorage.getItem("ms_llm_model") || "").trim(),
      llm_key: (localStorage.getItem("ms_llm_key") || "").trim(),
    };
  }

  function updateLlmButton() {
    const cfg = getLlmConfig();
    $("llmBtnLabel").textContent = cfg.llm_url && cfg.llm_model ? cfg.llm_model : "LLM";
  }

  function updateOffsetLabel() {
    const v = Math.round(state.subOffset * 10) / 10;
    state.subOffset = v;
    $("btnOffReset").textContent = `${v > 0 ? "+" : ""}${v.toFixed(1)}s`;
    syncPlayerTime();
  }

  function tokenizeClickable(text, lang, sentenceIdx) {
    if (!text) return "";
    const noSpace = ["ja", "zh", "zh-cn", "zh-tw", "th", "lo", "km"].includes((lang || "").toLowerCase());
    if (noSpace && typeof Intl !== "undefined" && Intl.Segmenter) {
      try {
        const seg = new Intl.Segmenter(lang || "ja", { granularity: "word" });
        let out = "";
        for (const part of seg.segment(text)) {
          out += part.isWordLike
            ? `<span class="w" data-w="${escHtml(part.segment)}" data-i="${sentenceIdx}">${escHtml(part.segment)}</span>`
            : escHtml(part.segment);
        }
        return out;
      } catch (_) {}
    }
    return text.split(/(\s+)/).map((tok) => {
      if (!tok.trim()) return escHtml(tok);
      const clean = tok.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");
      return clean ? `<span class="w" data-w="${escHtml(clean)}" data-i="${sentenceIdx}">${escHtml(tok)}</span>` : escHtml(tok);
    }).join("");
  }

  // ---------- System Init ----------
  async function initSystem() {
    try {
      const res = await fetch("/api/system");
      const data = await res.json();
      const pills = [
        `<span class="sys-pill ${data.device === "cuda" ? "ok" : "warn"}">${data.device === "cuda" ? "CUDA GPU" : "CPU int8"}</span>`,
        `<span class="sys-pill ok">${escHtml(data.gender_ai || "ECAPA AI")}</span>`,
      ];
      $("sysBadges").innerHTML = pills.join("");

      const srcSel = $("selSourceLang"), tgtSel = $("selTargetLang");
      if (!srcSel.options.length) {
        for (const l of (data.languages || [])) {
          srcSel.add(new Option(l.name, l.code));
          if (l.code !== "auto") tgtSel.add(new Option(l.name, l.code));
        }
        srcSel.value = "auto";
        tgtSel.value = "vi";
      }
    } catch (_) {
      $("sysBadges").innerHTML = `<span class="sys-pill warn">Offline</span>`;
    }
  }

  function showProgress(pct, stage, isError = false) {
    const wrap = $("progressWrap");
    wrap.classList.add("show");
    wrap.classList.toggle("error", isError);
    const p = Math.max(0, Math.min(100, Math.round(pct || 0)));
    $("progressPct").textContent = `${p}%`;
    $("progressStage").textContent = stage || "";
    $("progressFill").style.width = `${p}%`;
  }
  function hideProgress() { $("progressWrap").classList.remove("show", "error"); }

  // ---------- Jobs ----------
  async function startUrlJob() {
    const url = $("urlInput").value.trim();
    if (!url) return toast("Vui lòng dán đường link video trước.");
    const llm = getLlmConfig();
    $("btnStartUrl").disabled = true;
    showProgress(3, "Đang gửi yêu cầu...");
    try {
      const res = await fetch("/api/jobs/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url,
          source_lang: $("selSourceLang").value,
          target_lang: $("selTargetLang").value,
          trans_style: $("selStyle").value,
          whisper_model: $("selModel").value,
          prefer_existing_sub: $("chkPreferSub").checked,
          music_mode: $("chkMusicMode").checked,
          voice_override: $("selVoiceOverride").value,
          ...llm,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Không thể tạo job.");
      pollJob(data.job_id);
    } catch (e) {
      showProgress(100, e.message, true);
    } finally {
      $("btnStartUrl").disabled = false;
    }
  }

  async function startUploadJob(file) {
    if (!file) return;
    const llm = getLlmConfig();
    const fd = new FormData();
    fd.append("file", file);
    fd.append("source_lang", $("selSourceLang").value);
    fd.append("target_lang", $("selTargetLang").value);
    fd.append("trans_style", $("selStyle").value);
    fd.append("whisper_model", $("selModel").value);
    fd.append("music_mode", String($("chkMusicMode").checked));
    fd.append("voice_override", $("selVoiceOverride").value);
    fd.append("llm_url", llm.llm_url);
    fd.append("llm_model", llm.llm_model);
    fd.append("llm_key", llm.llm_key);

    showProgress(5, `Đang tải lên "${file.name}"...`);
    try {
      const res = await fetch("/api/jobs/upload", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload thất bại.");
      pollJob(data.job_id);
    } catch (e) {
      showProgress(100, e.message, true);
    } finally {
      $("fileInput").value = "";
    }
  }

  function pollJob(jobId) {
    clearInterval(state.pollTimer);
    const tick = async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        const job = await res.json();
        if (!res.ok) throw new Error(job.detail || "Không đọc được trạng thái.");
        if (job.status === "error") {
          clearInterval(state.pollTimer);
          return showProgress(100, job.stage || job.error || "Có lỗi xảy ra", true);
        }
        showProgress(job.progress || 0, job.stage || job.status, false);
        if (job.status === "done") {
          clearInterval(state.pollTimer);
          loadJobIntoPlayer(job);
          loadHistory();
          initSystem();
          setTimeout(hideProgress, 2400);
          toast("Đã hoàn tất phụ đề song ngữ!");
        }
      } catch (e) {
        clearInterval(state.pollTimer);
        showProgress(100, e.message, true);
      }
    };
    tick();
    state.pollTimer = setInterval(tick, 900);
  }

  async function openHistoryJob(jobId) {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      const job = await res.json();
      if (!res.ok) throw new Error(job.detail || "Không mở được video.");
      hideProgress();
      loadJobIntoPlayer(job);
      switchTab("transcript");
    } catch (e) {
      toast(e.message);
    }
  }

  function setActionButtonsEnabled(enabled) {
    ["btnRetranslate", "btnExportMenu", "btnImportSub", "btnDownloadSrt", "btnDownloadRaw", "btnExportHard", "btnExportSoft"]
      .forEach((id) => { $(id).disabled = !enabled; });
  }

  function loadJobIntoPlayer(job) {
    state.job = job;
    state.cues = Array.isArray(job.cues) ? job.cues : [];
    state.activeIdx = -1;
    state.shadowPausedIdx = -1;

    $("jobTitle").textContent = job.title || "Video";
    $("cueCount").textContent = String(state.cues.length);
    if (job.target_lang && $("selTargetLang").querySelector(`option[value="${job.target_lang}"]`)) {
      $("selTargetLang").value = job.target_lang;
    }
    if (job.trans_style && $("selStyle").querySelector(`option[value="${job.trans_style}"]`)) {
      $("selStyle").value = job.trans_style;
    }
    if (job.voice_override && $("selVoiceOverride").querySelector(`option[value="${job.voice_override}"]`)) {
      $("selVoiceOverride").value = job.voice_override;
    }
    if (typeof job.music_mode === "boolean") {
      $("chkMusicMode").checked = job.music_mode;
    }

    const player = $("player");
    if (job.media_url && !player.src.endsWith(job.media_url)) {
      player.src = job.media_url;
      player.load();
    }
    $("videoEmpty").style.display = job.media_url ? "none" : "flex";
    setActionButtonsEnabled(Boolean(job.id));
    renderCueList();
    renderOverlay();
  }

  // ---------- Transcript & 60FPS Sync ----------
  function renderCueList() {
    const q = $("cueSearch").value.trim().toLowerCase();
    const showOrig = $("chkShowOrig").checked;
    const showRead = $("chkShowRead").checked;
    const showVi = $("chkShowVi").checked;
    const showGender = $("chkShowGender").checked;

    const list = $("cueList");
    if (!state.cues.length) {
      list.innerHTML = `<div class="empty-msg">Chưa có dòng phụ đề nào.</div>`;
      return;
    }

    const parts = [];
    state.cues.forEach((c, idx) => {
      const tr = c.translated || c.vi || "";
      if (q && !(c.text || "").toLowerCase().includes(q) && !tr.toLowerCase().includes(q)) return;
      const g = c.gender || "unknown";
      const confPct = c.gender_conf ? ` ${Math.round(c.gender_conf * 100)}%` : "";
      const tip = `${c.gender_src || "AI"}${confPct} · Bấm để đổi Nam/Nữ`;
      const gHtml = showGender
        ? (g === "male"
          ? `<span class="g-tag g-male" data-toggle-gender="${idx}" title="${escHtml(tip)}">♂ Nam${confPct}</span>`
          : g === "female"
          ? `<span class="g-tag g-female" data-toggle-gender="${idx}" title="${escHtml(tip)}">♀ Nữ${confPct}</span>`
          : `<span class="g-tag g-unknown" data-toggle-gender="${idx}" title="Bấm để chọn Nam/Nữ">Auto</span>`)
        : "";

      parts.push(`
        <div class="cue-item ${idx === state.activeIdx ? "active" : ""}" data-idx="${idx}">
          <div class="cue-meta">
            <div class="cue-meta-left">
              <button class="ts-btn" data-seek="${c.start}">${fmtTime(c.start)} – ${fmtTime(c.end)}</button>
              ${gHtml}
            </div>
            <button class="cue-edit-btn" data-edit="${idx}" title="Sửa câu này">${SVG_EDIT} Sửa</button>
          </div>
          ${showRead && c.reading ? `<div class="cue-read">${escHtml(c.reading)}</div>` : ""}
          ${showOrig && c.text ? `<div class="cue-orig" data-idx="${idx}">${tokenizeClickable(c.text, c.lang || state.job?.detected_lang, idx)}</div>` : ""}
          ${showVi && tr ? `<div class="cue-vi">${escHtml(tr)}</div>` : ""}
        </div>
      `);
    });
    list.innerHTML = parts.length ? parts.join("") : `<div class="empty-msg">Không tìm thấy câu khớp từ khóa.</div>`;
  }

  function renderOverlay() {
    const box = $("subBox");
    const c = state.cues[state.activeIdx];
    if (!c) {
      box.classList.remove("visible");
      box.innerHTML = "";
      return;
    }
    const showOrig = $("chkShowOrig").checked;
    const showRead = $("chkShowRead").checked;
    const showVi = $("chkShowVi").checked;
    const tr = c.translated || c.vi || "";

    let html = "";
    if (showRead && c.reading) html += `<div class="sub-read">${escHtml(c.reading)}</div>`;
    if (showOrig && c.text) html += `<div class="sub-orig" data-idx="${state.activeIdx}">${tokenizeClickable(c.text, c.lang || state.job?.detected_lang, state.activeIdx)}</div>`;
    if (showVi && tr) html += `<div class="sub-vi">${escHtml(tr)}</div>`;

    box.innerHTML = html;
    box.classList.toggle("visible", Boolean(html));
  }

  function findCueIndexAt(t) {
    const cues = state.cues;
    for (let i = 0; i < cues.length; i++) {
      if (t >= cues[i].start && t <= cues[i].end) return i;
    }
    return -1;
  }

  function syncPlayerTime() {
    const player = $("player");
    const t = player.currentTime - state.subOffset;

    if (state.loopCue && state.activeIdx >= 0) {
      const cur = state.cues[state.activeIdx];
      if (cur && t >= cur.end - 0.02) {
        player.currentTime = Math.max(0, cur.start + state.subOffset);
        return;
      }
    }

    if (state.shadowMode && state.activeIdx >= 0 && state.shadowPausedIdx !== state.activeIdx) {
      const cur = state.cues[state.activeIdx];
      if (cur && !player.paused && t >= cur.end - 0.03) {
        state.shadowPausedIdx = state.activeIdx;
        player.pause();
        return;
      }
    }

    const idx = findCueIndexAt(t);
    if (idx !== state.activeIdx) {
      const prevEl = $("cueList").querySelector(`.cue-item[data-idx="${state.activeIdx}"]`);
      if (prevEl) prevEl.classList.remove("active");
      state.activeIdx = idx;
      if (idx !== state.shadowPausedIdx) state.shadowPausedIdx = -1;
      renderOverlay();
      if (idx >= 0) {
        const curEl = $("cueList").querySelector(`.cue-item[data-idx="${idx}"]`);
        if (curEl) {
          curEl.classList.add("active");
          if ($("chkAutoScroll").checked && $("pane-transcript").classList.contains("active")) {
            curEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
          }
        }
      }
    }
  }

  function startRafSync() {
    cancelAnimationFrame(state.rafId);
    const step = () => {
      syncPlayerTime();
      if (!$("player").paused && !$("player").ended) {
        state.rafId = requestAnimationFrame(step);
      }
    };
    state.rafId = requestAnimationFrame(step);
  }

  function jumpToCue(idx) {
    if (!state.cues.length) return;
    idx = Math.max(0, Math.min(state.cues.length - 1, idx));
    const player = $("player");
    state.shadowPausedIdx = -1;
    player.currentTime = Math.max(0, state.cues[idx].start + state.subOffset + 0.01);
    player.play().catch(() => {});
  }

  async function toggleCueGender(idx) {
    if (!state.job || !state.cues[idx]) return;
    const nextG = state.cues[idx].gender === "male" ? "female" : "male";
    try {
      const res = await fetch(`/api/jobs/${state.job.id}/cues/${idx}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gender: nextG }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Không đổi được giới tính.");
      state.cues[idx] = data.cue;
      renderCueList();
      renderOverlay();
      toast(`Đã đổi câu #${idx + 1} sang ${nextG === "male" ? "♂ Nam" : "♀ Nữ"}`);
    } catch (e) {
      toast(e.message);
    }
  }

  // ---------- Edit Cue ----------
  function openEditCue(idx) {
    const c = state.cues[idx];
    if (!c) return;
    state.editingIdx = idx;
    $("player").pause();
    $("editStart").value = c.start;
    $("editEnd").value = c.end;
    $("editOrig").value = c.text || "";
    $("editVi").value = c.translated || c.vi || "";
    $("editCueModal").classList.add("show");
  }

  async function saveEditedCue() {
    if (!state.job || state.editingIdx < 0) return;
    const payload = {
      start: parseFloat($("editStart").value),
      end: parseFloat($("editEnd").value),
      text: $("editOrig").value.trim(),
      vi: $("editVi").value.trim(),
    };
    try {
      const res = await fetch(`/api/jobs/${state.job.id}/cues/${state.editingIdx}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Lưu thất bại.");
      state.cues[state.editingIdx] = data.cue;
      $("editCueModal").classList.remove("show");
      renderCueList();
      renderOverlay();
      toast("Đã lưu chỉnh sửa câu phụ đề!");
    } catch (e) {
      toast(e.message);
    }
  }

  // ---------- Retranslate, Import, Download & Export ----------
  async function retranslateCurrent() {
    if (!state.job) return;
    const llm = getLlmConfig();
    $("btnRetranslate").disabled = true;
    showProgress(35, "Đang chạy AI ECAPA + Wav2Vec2 phân tích lại giọng & dịch...");
    try {
      const res = await fetch(`/api/jobs/${state.job.id}/retranslate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_lang: $("selTargetLang").value,
          source_lang: $("selSourceLang").value === "auto" ? null : $("selSourceLang").value,
          trans_style: $("selStyle").value,
          music_mode: $("chkMusicMode").checked,
          voice_override: $("selVoiceOverride").value,
          ...llm,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Dịch lại thất bại.");
      showProgress(100, data.stage || "Đã phân tích & dịch lại xong!");
      loadJobIntoPlayer(data);
      loadHistory();
      initSystem();
      setTimeout(hideProgress, 2400);
      toast("Đã cập nhật giới tính giọng & bản dịch!");
    } catch (e) {
      showProgress(100, e.message, true);
    } finally {
      $("btnRetranslate").disabled = false;
    }
  }

  async function importSubtitleFile(file) {
    if (!file || !state.job) return;
    const fd = new FormData();
    fd.append("file", file);
    fd.append("source_lang", $("selSourceLang").value);
    fd.append("target_lang", $("selTargetLang").value);
    fd.append("trans_style", $("selStyle").value);
    fd.append("music_mode", String($("chkMusicMode").checked));
    fd.append("voice_override", $("selVoiceOverride").value);
    showProgress(45, `Đang nạp & phân tích giọng cho "${file.name}"...`);
    try {
      const res = await fetch(`/api/jobs/${state.job.id}/import-sub`, { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Nạp phụ đề thất bại.");
      showProgress(100, data.stage || "Đã nạp xong!");
      loadJobIntoPlayer(data);
      setTimeout(hideProgress, 2200);
      toast("Đã nạp & dịch phụ đề xong!");
    } catch (e) {
      showProgress(100, e.message, true);
    } finally {
      $("subFileInput").value = "";
    }
  }

  function downloadDualSrt() {
    if (!state.cues.length) return;
    const showOrig = $("chkShowOrig").checked, showVi = $("chkShowVi").checked;
    const blocks = [];
    state.cues.forEach((c, i) => {
      const lines = [];
      if (showOrig && c.text) lines.push(c.text);
      const tr = c.translated || c.vi || "";
      if (showVi && tr) lines.push(tr);
      if (lines.length) blocks.push(`${i + 1}\n${fmtSrtTime(c.start)} --> ${fmtSrtTime(c.end)}\n${lines.join("\n")}`);
    });
    const blob = new Blob(["\uFEFF" + blocks.join("\n\n") + "\n"], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${(state.job?.title || "minhsub").replace(/[\\/:*?"<>|]+/g, "_").slice(0, 60)}.srt`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  async function exportVideo(mode) {
    if (!state.job) return;
    $("exportMenu").classList.remove("open");
    $("btnExportMenu").disabled = true;
    showProgress(65, mode === "hardsub" ? "FFmpeg đang đốt phụ đề song ngữ vào MP4..." : "FFmpeg đang gắn phụ đề mềm MP4...");
    try {
      const res = await fetch(`/api/jobs/${state.job.id}/export-video`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode, include_orig: $("chkShowOrig").checked,
          include_read: $("chkShowRead").checked, include_trans: $("chkShowVi").checked,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Xuất video thất bại.");
      showProgress(100, "Xuất video xong! Đang bắt đầu tải xuống...");
      window.location.href = data.download_url;
      setTimeout(hideProgress, 2600);
      toast("Xuất video MP4 thành công!");
    } catch (e) {
      showProgress(100, e.message, true);
    } finally {
      $("btnExportMenu").disabled = false;
    }
  }

  // ---------- Dictionary ----------
  async function openLookup(word, sentenceIdx) {
    const clean = (word || "").trim();
    if (!clean) return;
    if ($("chkPauseLookup").checked) $("player").pause();

    const cue = state.cues[sentenceIdx] || state.cues[state.activeIdx] || {};
    const sentence = cue.text || "";
    const lang = cue.lang || state.job?.detected_lang || "auto";
    const targetLang = $("selTargetLang").value || "vi";

    $("lkWord").textContent = clean;
    $("lkPhonetic").textContent = "";
    $("lkBody").innerHTML = `<div style="color:var(--muted);">Đang tra từ điển...</div>`;
    $("lookupModal").classList.add("show");
    state.lastLookup = null;

    try {
      const qs = new URLSearchParams({ word: clean, sentence, lang, target_lang: targetLang });
      const res = await fetch(`/api/lookup?${qs}`);
      const d = await res.json();
      if (!res.ok) throw new Error(d.detail || "Tra từ thất bại.");

      state.lastLookup = { ...d, sentence };
      $("lkWord").textContent = d.word;
      $("lkPhonetic").textContent = d.phonetic || "";
      const posHtml = (d.pos_tags || []).map((p) => `<span class="dict-pos">${escHtml(p)}</span>`).join("");
      const groupsHtml = (d.vi_dict || []).map((g) => `<div style="margin-top:6px;"><strong style="color:#93c5fd;">${escHtml(g.pos)}:</strong> ${escHtml((g.terms || []).join(", "))}</div>`).join("");

      $("lkBody").innerHTML = `
        <div>${posHtml}</div>
        <div class="dict-main">${escHtml(d.vi_main || "(Không có nghĩa)")}</div>
        ${d.extra_note ? `<div class="dict-block"><strong>Giải thích:</strong> ${escHtml(d.extra_note)}</div>` : ""}
        ${groupsHtml ? `<div class="dict-block">${groupsHtml}</div>` : ""}
        ${sentence ? `<div class="dict-block"><div style="color:#e2e8f0;"><strong>Ngữ cảnh:</strong> ${escHtml(sentence)}</div>${d.sentence_vi ? `<div style="color:var(--sub-vi); margin-top:4px;">${escHtml(d.sentence_vi)}</div>` : ""}</div>` : ""}
      `;
    } catch (e) {
      $("lkBody").innerHTML = `<div style="color:#fca5a5;">${escHtml(e.message)}</div>`;
    }
  }

  function playLookupAudio(item) {
    if (!item) return;
    if (item.audio_url) new Audio(item.audio_url).play().catch(() => speakFallback(item.word, item.lang));
    else speakFallback(item.word, item.lang);
  }
  function speakFallback(text, lang) {
    if (!("speechSynthesis" in window) || !text) return;
    const u = new SpeechSynthesisUtterance(text);
    if (lang && lang !== "auto") u.lang = lang;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  }

  // ---------- History & Vocab ----------
  async function loadHistory() {
    try {
      const items = await (await fetch("/api/history")).json();
      const box = $("historyList");
      if (!items.length) return (box.innerHTML = `<div class="empty-msg">Chưa có video nào được lưu.</div>`);
      box.innerHTML = items.map((h) => `
        <div class="list-card">
          <div class="list-card-head">
            <div class="card-title" data-open-job="${h.id}">${escHtml(h.title)}</div>
            <button class="mini-icon-btn del" data-del-job="${h.id}" title="Xóa">${SVG_TRASH}</button>
          </div>
          <div class="card-sub">${escHtml((h.detected_lang || "?").toUpperCase())} → ${escHtml((h.target_lang || "vi").toUpperCase())} · ${h.cue_count || 0} câu</div>
        </div>
      `).join("");
    } catch (_) {}
  }

  async function loadVocab() {
    try {
      state.vocab = await (await fetch("/api/vocab")).json();
      $("vocabCount").textContent = String(state.vocab.length);
      renderVocabList();
    } catch (_) {}
  }

  function renderVocabList() {
    const q = $("vocabSearch").value.trim().toLowerCase();
    const filtered = state.vocab.filter((v) => !q || (v.word || "").toLowerCase().includes(q) || (v.vi_main || v.meaning || "").toLowerCase().includes(q));
    const box = $("vocabList");
    if (!filtered.length) return (box.innerHTML = `<div class="empty-msg">Sổ từ vựng đang trống. Bấm vào từ bất kỳ trên phụ đề để tra và lưu.</div>`);
    box.innerHTML = filtered.map((v, idx) => `
      <div class="list-card">
        <div class="list-card-head">
          <div><span class="vocab-word">${escHtml(v.word)}</span>${v.phonetic ? `<span class="vocab-ph">${escHtml(v.phonetic)}</span>` : ""}</div>
          <div class="card-actions">
            <button class="mini-icon-btn" data-speak-vocab="${idx}" title="Nghe phát âm">${SVG_VOL}</button>
            <button class="mini-icon-btn del" data-del-vocab="${escHtml(v.id || v.word)}" title="Xóa">${SVG_TRASH}</button>
          </div>
        </div>
        <div class="vocab-mean">${escHtml(v.vi_main || v.meaning || "")}</div>
        ${v.sentence ? `<div class="card-sub">${escHtml(v.sentence)}</div>` : ""}
        ${v.sentence_vi ? `<div class="card-sub vocab-sent-vi">${escHtml(v.sentence_vi)}</div>` : ""}
      </div>
    `).join("");
  }

  function switchTab(name) {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    document.querySelectorAll(".tab-pane").forEach((p) => p.classList.toggle("active", p.id === `pane-${name}`));
  }

  // ---------- UI Drawer & Dropdowns ----------
  $("btnToggleSettings").onclick = () => {
    const open = $("settingsDrawer").classList.toggle("open");
    $("btnToggleSettings").classList.toggle("active", open);
  };

  $("btnExportMenu").onclick = (e) => {
    e.stopPropagation();
    $("exportMenu").classList.toggle("open");
  };
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".menu-wrap")) $("exportMenu").classList.remove("open");
  });

  // ---------- Events ----------
  $("btnStartUrl").onclick = startUrlJob;
  $("urlInput").addEventListener("keydown", (e) => { if (e.key === "Enter") startUrlJob(); });
  $("btnPickFile").onclick = () => $("fileInput").click();
  $("fileInput").onchange = (e) => startUploadJob(e.target.files[0]);

  const stage = $("videoStage");
  ["dragenter", "dragover"].forEach((ev) => stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) => stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.remove("dragover"); }));
  stage.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) startUploadJob(file);
  });

  const player = $("player");
  player.addEventListener("play", startRafSync);
  player.addEventListener("seeked", syncPlayerTime);
  player.addEventListener("timeupdate", syncPlayerTime);
  player.addEventListener("pause", () => cancelAnimationFrame(state.rafId));

  $("selSpeed").onchange = () => { player.playbackRate = Number($("selSpeed").value) || 1; };
  $("btnPrevCue").onclick = () => jumpToCue(state.activeIdx > 0 ? state.activeIdx - 1 : 0);
  $("btnNextCue").onclick = () => jumpToCue(state.activeIdx + 1);
  $("btnLoopCue").onclick = () => { state.loopCue = !state.loopCue; $("btnLoopCue").classList.toggle("active", state.loopCue); };
  $("btnShadowMode").onclick = () => {
    state.shadowMode = !state.shadowMode;
    state.shadowPausedIdx = -1;
    $("btnShadowMode").classList.toggle("active", state.shadowMode);
    toast(state.shadowMode ? "Bật chế độ Shadowing (tự dừng cuối câu)" : "Tắt chế độ Shadowing");
  };
  $("btnFullscreen").onclick = () => { document.fullscreenElement ? document.exitFullscreen() : stage.requestFullscreen().catch(() => {}); };

  $("btnOffMinus").onclick = () => { state.subOffset -= 0.1; updateOffsetLabel(); };
  $("btnOffPlus").onclick = () => { state.subOffset += 0.1; updateOffsetLabel(); };
  $("btnOffReset").onclick = () => { state.subOffset = 0; updateOffsetLabel(); };

  ["chkShowOrig", "chkShowRead", "chkShowVi", "chkShowGender"].forEach((id) => {
    $(id).onchange = () => { renderCueList(); renderOverlay(); };
  });
  $("rngFontSize").oninput = (e) => { $("subBox").style.fontSize = `${e.target.value}px`; };
  $("cueSearch").oninput = renderCueList;
  $("vocabSearch").oninput = renderVocabList;

  $("btnRetranslate").onclick = retranslateCurrent;
  $("btnImportSub").onclick = () => { $("exportMenu").classList.remove("open"); $("subFileInput").click(); };
  $("subFileInput").onchange = (e) => importSubtitleFile(e.target.files[0]);
  $("btnDownloadSrt").onclick = () => { $("exportMenu").classList.remove("open"); downloadDualSrt(); };
  $("btnDownloadRaw").onclick = () => { $("exportMenu").classList.remove("open"); if (state.job) window.location.href = `/api/jobs/${state.job.id}/download-video`; };
  $("btnExportHard").onclick = () => exportVideo("hardsub");
  $("btnExportSoft").onclick = () => exportVideo("softsub");

  document.querySelectorAll(".tab-btn").forEach((b) => { b.onclick = () => switchTab(b.dataset.tab); });
  $("btnOpenVocab").onclick = () => switchTab("vocab");
  $("btnRefreshHistory").onclick = loadHistory;

  function handleSubClick(e) {
    const toggleG = e.target.closest("[data-toggle-gender]");
    if (toggleG) return toggleCueGender(Number(toggleG.dataset.toggleGender));
    const editBtn = e.target.closest("[data-edit]");
    if (editBtn) return openEditCue(Number(editBtn.dataset.edit));
    const seekBtn = e.target.closest("[data-seek]");
    if (seekBtn) {
      state.shadowPausedIdx = -1;
      player.currentTime = Math.max(0, Number(seekBtn.dataset.seek) + state.subOffset + 0.01);
      player.play().catch(() => {});
      return;
    }
    const wordEl = e.target.closest(".w");
    if (wordEl) {
      const sel = window.getSelection()?.toString().trim() || "";
      if (sel.length > 0 && sel !== wordEl.textContent.trim()) return;
      openLookup(wordEl.dataset.w, Number(wordEl.dataset.i));
    }
  }
  $("cueList").addEventListener("click", handleSubClick);
  $("subOverlay").addEventListener("click", handleSubClick);

  document.addEventListener("mouseup", (e) => {
    const selBtn = $("selBtn");
    if (e.target.closest("#selBtn")) return;
    const sel = window.getSelection();
    const txt = (sel ? sel.toString() : "").replace(/\s+/g, " ").trim();
    const anchor = sel?.anchorNode ? (sel.anchorNode.nodeType === 1 ? sel.anchorNode : sel.anchorNode.parentElement) : null;
    const host = anchor?.closest?.(".cue-orig, .sub-orig");
    if (txt.length >= 2 && host) {
      const rect = sel.getRangeAt(0).getBoundingClientRect();
      state.selPayload = { word: txt, idx: Number(host.dataset.idx ?? state.activeIdx) };
      selBtn.style.left = `${Math.max(10, Math.min(window.innerWidth - 135, rect.left + rect.width / 2 - 58))}px`;
      selBtn.style.top = `${Math.max(10, rect.top - 38)}px`;
      selBtn.style.display = "inline-flex";
    } else {
      selBtn.style.display = "none";
    }
  });
  $("selBtn").onclick = () => {
    $("selBtn").style.display = "none";
    if (state.selPayload) openLookup(state.selPayload.word, state.selPayload.idx);
  };

  $("historyList").addEventListener("click", async (e) => {
    const openEl = e.target.closest("[data-open-job]");
    if (openEl) return openHistoryJob(openEl.dataset.openJob);
    const delEl = e.target.closest("[data-del-job]");
    if (delEl && confirm("Xóa video và phụ đề này khỏi máy?")) {
      await fetch(`/api/history/${delEl.dataset.delJob}`, { method: "DELETE" });
      loadHistory();
      toast("Đã xóa khỏi lịch sử.");
    }
  });

  $("vocabList").addEventListener("click", async (e) => {
    const spEl = e.target.closest("[data-speak-vocab]");
    if (spEl) {
      const q = $("vocabSearch").value.trim().toLowerCase();
      const filtered = state.vocab.filter((v) => !q || (v.word || "").toLowerCase().includes(q) || (v.vi_main || v.meaning || "").toLowerCase().includes(q));
      return playLookupAudio(filtered[Number(spEl.dataset.speakVocab)]);
    }
    const delEl = e.target.closest("[data-del-vocab]");
    if (delEl) {
      await fetch(`/api/vocab/${encodeURIComponent(delEl.dataset.delVocab)}`, { method: "DELETE" });
      loadVocab();
    }
  });
  $("btnExportVocab").onclick = () => { window.location.href = "/api/vocab/export"; };
  $("btnClearVocab").onclick = async () => {
    if (!state.vocab.length || !confirm("Xóa toàn bộ sổ từ vựng?")) return;
    await fetch("/api/vocab", { method: "DELETE" });
    loadVocab();
  };

  $("btnLkClose").onclick = () => $("lookupModal").classList.remove("show");
  $("lookupModal").onclick = (e) => { if (e.target === $("lookupModal")) $("lookupModal").classList.remove("show"); };
  $("btnLkAudio").onclick = () => playLookupAudio(state.lastLookup);
  $("btnLkSave").onclick = async () => {
    if (!state.lastLookup) return;
    const res = await fetch("/api/vocab", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state.lastLookup) });
    if (res.ok) { loadVocab(); toast(`Đã lưu "${state.lastLookup.word}"!`); }
  };

  $("btnEditClose").onclick = $("btnEditCancel").onclick = () => $("editCueModal").classList.remove("show");
  $("editCueModal").onclick = (e) => { if (e.target === $("editCueModal")) $("editCueModal").classList.remove("show"); };
  $("btnEditSave").onclick = saveEditedCue;

  $("btnLlmModal").onclick = () => {
    const cfg = getLlmConfig();
    $("llmUrl").value = cfg.llm_url;
    $("llmModel").value = cfg.llm_model;
    $("llmKey").value = cfg.llm_key;
    $("llmModal").classList.add("show");
  };
  $("btnLlmClose").onclick = () => $("llmModal").classList.remove("show");
  $("llmModal").onclick = (e) => { if (e.target === $("llmModal")) $("llmModal").classList.remove("show"); };
  $("btnLlmSave").onclick = () => {
    localStorage.setItem("ms_llm_url", $("llmUrl").value.trim());
    localStorage.setItem("ms_llm_model", $("llmModel").value.trim());
    localStorage.setItem("ms_llm_key", $("llmKey").value.trim());
    updateLlmButton();
    $("llmModal").classList.remove("show");
    toast("Đã lưu cấu hình LLM.");
  };
  $("btnLlmClear").onclick = () => {
    ["ms_llm_url", "ms_llm_model", "ms_llm_key"].forEach((k) => localStorage.removeItem(k));
    $("llmUrl").value = "";
    $("llmModel").value = "";
    $("llmKey").value = "";
    updateLlmButton();
    toast("Đã xóa cấu hình LLM.");
  };

  document.addEventListener("keydown", (e) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (e.key === "Escape") {
      ["lookupModal", "llmModal", "editCueModal"].forEach((id) => $(id).classList.remove("show"));
      $("exportMenu").classList.remove("open");
      return;
    }
    if (!player.src) return;
    if (e.code === "Space") {
      e.preventDefault();
      if (player.paused) {
        if (state.shadowMode && state.shadowPausedIdx >= 0) player.currentTime += 0.08;
        player.play();
      } else {
        player.pause();
      }
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      jumpToCue(state.activeIdx > 0 ? state.activeIdx - 1 : 0);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      jumpToCue(state.activeIdx + 1);
    } else if (e.key.toLowerCase() === "r") {
      $("btnLoopCue").click();
    } else if (e.key.toLowerCase() === "s") {
      $("btnShadowMode").click();
    }
  });

  updateLlmButton();
  initSystem();
  loadHistory();
  loadVocab();
})();