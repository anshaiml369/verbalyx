/* Verbalyx studio — frontend logic. Talks to server.py (/api/*). */
(function () {
  "use strict";
  const { $, $$, el, api, fmt, markdown, initTabs, setBusy, setStatus, callout, toast, copyText, download, storage, debounce } = window.RUI;

  const ALLOWED_EXT = [".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"];
  const YT_HOSTS = ["youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"];
  const JOB_KEY = "videoassistant.job.v1";
  const OUT_KEYS = ["summary", "action_items", "key_decisions", "open_questions", "transcript"];

  const state = { source: "url", file: null, objectUrl: null, maxMb: 500, job: null, pollTimer: null, health: null };

  /* ------------------------------------------------------------------ */
  /* Health / requirements                                              */
  /* ------------------------------------------------------------------ */
  async function loadHealth() {
    try {
      const h = await api("/api/health", { timeout: 8000 });
      state.health = h;
      state.maxMb = h.max_upload_mb;
      $("#max-mb").textContent = h.max_upload_mb;
      if (h.mistral_model) $$(".llm-model").forEach((n) => { n.textContent = h.mistral_model; });
      if (h.youtube_enabled === false) {
        const yt = document.querySelector('[data-source="url"]');
        if (yt) { yt.disabled = true; yt.title = "YouTube links are disabled on this public demo; upload a file instead."; }
        setSource("file");
      }
      const item = (ok, label, note, optional) => el("li", {}, [
        el("span", { class: `req-icon ${ok ? "ok" : optional ? "opt" : "bad"}`, "aria-hidden": "true", text: ok ? "✓" : optional ? "–" : "✕" }),
        el("span", {}, [el("span", { class: "sr-only", text: ok ? "Available: " : "Missing: " }), label, el("span", { class: "req-note", text: note })]),
      ]);
      $("#req-list").replaceChildren(
        item(h.ffmpeg, "ffmpeg", h.ffmpeg ? "Audio decoding available" : "Install ffmpeg and restart the server (e.g. brew install ffmpeg)"),
        item(h.mistral_key, "MISTRAL_API_KEY", h.mistral_key ? "LLM analysis and Q&A available" : "Add it to .env (see .env.example) and restart"),
        item(true, `Whisper model “${h.whisper_model}”`, "Downloaded on first use; runs on CPU"),
        item(h.sarvam_key, "SARVAM_API_KEY", h.sarvam_key ? "Hinglish transcription available" : "Optional — only needed for Hinglish", true),
      );
      if (h.ready) setStatus($("#api-status"), "ok", "Server ready");
      else setStatus($("#api-status"), "warn", "Server not fully configured");
    } catch (err) {
      setStatus($("#api-status"), "err", "Server unreachable");
      $("#req-list").replaceChildren(el("li", { class: "field-error", text: err.message }));
    }
  }

  /* ------------------------------------------------------------------ */
  /* Source selection + preview                                         */
  /* ------------------------------------------------------------------ */
  function setSource(kind) {
    state.source = kind;
    $$("[data-source]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.source === kind)));
    $("#url-field").hidden = kind !== "url";
    $("#file-field").hidden = kind !== "file";
    refreshPreview();
  }

  function youtubeId(raw) {
    let u;
    try { u = new URL(raw.trim()); } catch { return { error: "Enter a full URL starting with https://" }; }
    if (!["http:", "https:"].includes(u.protocol) || !YT_HOSTS.includes(u.hostname.toLowerCase())) return { error: "Only YouTube links (youtube.com or youtu.be) are supported." };
    let id = null;
    if (u.hostname.endsWith("youtu.be")) id = u.pathname.slice(1).split("/")[0];
    else id = u.searchParams.get("v") || (u.pathname.match(/\/(?:embed|shorts|live)\/([A-Za-z0-9_-]{6,})/) || [])[1];
    return id && /^[A-Za-z0-9_-]{6,}$/.test(id) ? { id } : { error: "No video ID found in this link." };
  }

  function validateFile(f) {
    if (!f) return "Choose a media file.";
    const ext = (f.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
    if (!ALLOWED_EXT.includes(ext)) return `Unsupported file type “${ext || "none"}”. Use one of: ${ALLOWED_EXT.join(", ")}.`;
    if (f.size === 0) return "The file is empty.";
    if (f.size > state.maxMb * 1024 * 1024) return `The file is ${(f.size / 1048576).toFixed(0)} MB; the limit is ${state.maxMb} MB.`;
    return "";
  }

  function refreshPreview() {
    const stage = $("#media-stage");
    const empty = () => stage.replaceChildren($("#stage-empty") || el("div", { class: "stage-empty", text: "Preview appears here once you add a link or file" }));
    if (state.source === "url") {
      const v = $("#url-input").value.trim();
      const r = v ? youtubeId(v) : null;
      if (r && r.id) {
        stage.replaceChildren(el("iframe", {
          src: `https://www.youtube-nocookie.com/embed/${encodeURIComponent(r.id)}`, title: "YouTube video preview",
          allow: "encrypted-media; picture-in-picture", allowfullscreen: true, loading: "lazy", referrerpolicy: "strict-origin-when-cross-origin",
        }));
      } else empty();
    } else if (state.file && !validateFile(state.file)) {
      if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = URL.createObjectURL(state.file);
      const isAudio = state.file.type.startsWith("audio/") || /\.(mp3|wav|m4a|aac|ogg|flac)$/i.test(state.file.name);
      stage.replaceChildren(isAudio
        ? el("div", { class: "stage-audio" }, [el("audio", { controls: true, src: state.objectUrl, preload: "metadata" }), el("span", { class: "fname", text: state.file.name })])
        : el("video", { controls: true, src: state.objectUrl, preload: "metadata", playsinline: true }));
    } else empty();
  }

  function pickFile(f) {
    state.file = f || null;
    $("#file-name").textContent = f ? `${f.name} · ${(f.size / 1048576).toFixed(1)} MB` : "Choose or drop a file";
    $("#file-error").textContent = f ? validateFile(f) : "";
    refreshPreview();
  }

  /* ------------------------------------------------------------------ */
  /* Jobs                                                               */
  /* ------------------------------------------------------------------ */
  async function submit(e) {
    e.preventDefault();
    $("#source-alert").replaceChildren();
    const fd = new FormData();
    fd.append("source_type", state.source);
    fd.append("language", $("#language").value);
    if (state.source === "url") {
      const r = youtubeId($("#url-input").value || "");
      $("#url-error").textContent = r.error || "";
      $("#url-input").setAttribute("aria-invalid", r.error ? "true" : "false");
      if (r.error) return $("#url-input").focus();
      fd.append("url", $("#url-input").value.trim());
    } else {
      const msg = validateFile(state.file);
      $("#file-error").textContent = msg;
      if (msg) return;
      fd.append("file", state.file, state.file.name);
    }
    if ($("#language").value === "hinglish" && state.health && !state.health.sarvam_key) {
      $("#source-alert").replaceChildren(callout("warn", "Hinglish needs a Sarvam key", ["SARVAM_API_KEY is not configured on the server. Choose English or add the key."]));
      return;
    }
    const btn = $("#analyse-btn");
    setBusy(btn, true, state.source === "file" ? "Uploading…" : "Starting…");
    try {
      const job = await api("/api/jobs", { method: "POST", body: fd, timeout: 10 * 60 * 1000 });
      resetOutputs();
      follow(job);
    } catch (err) {
      setBusy(btn, false);
      const titles = { 503: "Server is not ready", 429: /being processed/i.test(err.message) ? "Another video is processing" : "Usage limit reached", 413: "File too large", 415: "Unsupported file type", 422: "Invalid input" };
      $("#source-alert").replaceChildren(callout(err.status === 429 ? "warn" : "err", titles[err.status] || "Could not start the analysis", [err.message]));
    }
  }

  function follow(job) {
    clearTimeout(state.pollTimer);
    storage.set(JOB_KEY, job.id);
    setBusy($("#analyse-btn"), true, "Processing…");
    $("#job-progress").hidden = false;
    renderJob(job);
    let failures = 0;
    const poll = async () => {
      try {
        const j = await api(`/api/jobs/${job.id}`, { timeout: 10000 });
        failures = 0;
        renderJob(j);
        if (j.status === "completed" || j.status === "failed") return finish(j);
      } catch (err) {
        if (err.status === 404) { storage.set(JOB_KEY, null); return finish(null, "The job is no longer on the server (it was probably restarted). Start a new analysis."); }
        if (++failures === 3) toast("Lost contact with the server — retrying…", "err");
      }
      state.pollTimer = setTimeout(poll, failures ? 4000 : 1500);
    };
    poll();
  }

  function finish(job, message, silent = false) {
    clearTimeout(state.pollTimer);
    setBusy($("#analyse-btn"), false);
    $("#job-progress").hidden = true;
    if (!job) { $("#job-alert").replaceChildren(callout("err", "Tracking stopped", [message])); return; }
    if (job.status === "completed") {
      $("#question").disabled = false;
      $("#ask-btn").disabled = false;
      if (!job.qa.length) $("#qa-list").replaceChildren(el("div", { class: "empty-state" }, [el("strong", { text: "Index ready" }), el("span", { text: "Ask a question about what was said in the video." })]));
      if (!silent) toast("Analysis complete", "ok");
    } else {
      const err = job.error || {};
      const stageName = err.stage ? $(`[data-stage="${err.stage}"] .s-name`).textContent.toLowerCase() : "start-up";
      const hint = /ffmpeg|ffprobe/i.test(err.message || "") ? "ffmpeg appears to be missing on the server." : /api key|401|unauthor/i.test(err.message || "") ? "Check the API key configured on the server." : "See the server log below for details.";
      $("#job-alert").replaceChildren(callout("err", `Failed during the ${stageName} stage`, [el("p", {}, [el("code", { text: err.type || "Error" }), " ", err.message || "Unknown error"]), hint]));
    }
  }

  function resetOutputs() {
    $("#job-alert").replaceChildren();
    $("#analysis-title").textContent = "Processing…";
    for (const k of OUT_KEYS) $(`[data-out="${k}"]`).replaceChildren(el("p", { class: "muted", text: "Waiting for this stage…" }));
    $("#transcript-count").textContent = "";
    $("#question").disabled = true;
    $("#ask-btn").disabled = true;
    $("#qa-list").replaceChildren(el("div", { class: "empty-state" }, [el("strong", { text: "Questions unlock after indexing" })]));
    $("#copy-transcript").disabled = $("#download-transcript").disabled = true;
  }

  function renderJob(job) {
    state.job = job;
    const badge = $("#job-badge");
    badge.textContent = job.status;
    badge.className = `badge ${{ running: "badge-accent", queued: "badge-info", completed: "badge-ok", failed: "badge-err" }[job.status] || ""}`;
    const src = job.source.type === "url" ? job.source.url : `${job.source.filename} (${(job.source.size_bytes / 1048576).toFixed(1)} MB)`;
    const elapsed = job.started_at ? ((job.finished_at || Date.now() / 1000) - job.started_at) : 0;
    $("#job-sub").textContent = `${src} · ${job.language}${job.started_at ? ` · ${fmt.secs(elapsed)}` : ""}`;
    $("#job-sub").title = src;
    for (const [key, s] of Object.entries(job.stages)) {
      const li = $(`[data-stage="${key}"]`);
      li.dataset.state = job.status === "failed" && s.status === "pending" ? "skipped" : s.status;
      const end = s.ended_at || (s.status === "running" ? Date.now() / 1000 : null);
      $(".s-time", li).textContent = s.started_at && end ? fmt.secs(end - s.started_at) : "—";
    }
    const o = job.outputs || {};
    if (o.title) $("#analysis-title").textContent = o.title.replace(/[*_#`]+/g, "").trim().replace(/^["“']+|["”']+$/g, "").trim();
    for (const k of ["summary", "action_items", "key_decisions", "open_questions"]) {
      const box = $(`[data-out="${k}"]`);
      if (o[k] && box.dataset.rendered !== String(o[k].length)) { box.innerHTML = markdown(o[k]); box.dataset.rendered = String(o[k].length); }
    }
    if (o.transcript) {
      const box = $('[data-out="transcript"]');
      if (box.dataset.rendered !== String(o.transcript.length)) {
        box.textContent = o.transcript;
        box.dataset.rendered = String(o.transcript.length);
        const words = o.transcript.trim().split(/\s+/).length;
        $("#transcript-count").textContent = ` ${fmt.num(words)} words`;
        $("#copy-transcript").disabled = $("#download-transcript").disabled = false;
      }
    }
    const log = $("#job-log");
    if (typeof job.log === "string" && log.textContent !== job.log) { log.textContent = job.log || "(no output yet)"; log.scrollTop = log.scrollHeight; }
    if (job.qa && job.qa.length && !$("#qa-list .qa")) { $("#qa-list").replaceChildren(); job.qa.slice().reverse().forEach((q) => $("#qa-list").append(qaCard(q))); }
  }

  /* ------------------------------------------------------------------ */
  /* Q&A                                                                */
  /* ------------------------------------------------------------------ */
  function qaCard(entry) {
    const passages = entry.passages;
    const foreign = passages ? passages.filter((p) => !p.in_current_transcript).length : 0;
    const evidence = passages === null
      ? [callout("warn", "Retrieved passages unavailable", ["The server could not read the retriever from the chain, so the evidence for this answer cannot be shown."])]
      : passages.map((p) => el("article", { class: `passage${p.in_current_transcript ? "" : " foreign"}` }, [
        el("div", { class: "passage-head" }, [
          el("span", { class: "mono", text: `#${p.rank} · chunk ${p.chunk_index ?? "?"} · L2 distance ${p.distance.toFixed(3)}` }),
          p.in_current_transcript ? el("span", { class: "badge badge-ok", text: "from this transcript" }) : el("span", { class: "badge badge-err", text: "not in this transcript" }),
        ]),
        el("p", { text: p.text }),
      ]));
    return el("article", { class: "qa" }, [
      el("div", { class: "qa-answer" }, [
        el("p", { class: "qa-q", text: entry.question }),
        el("div", { class: "prose", html: markdown(entry.answer) }),
        el("p", { class: "qa-meta", text: `answered in ${fmt.secs(entry.latency_s)} · ${new Date(entry.asked_at * 1000).toLocaleTimeString()}` }),
      ]),
      el("div", { class: "qa-evidence" }, [
        el("p", { class: "eyebrow", text: `Retrieved context (${passages ? passages.length : 0} chunks, lower distance = closer)` }),
        foreign ? callout("err", `${foreign} passage${foreign === 1 ? "" : "s"} are not from this transcript`, ["These chunks do not occur in the current transcript, so the index may not match this video. Treat this answer with caution and re-run the analysis."]) : null,
        ...evidence,
      ]),
    ]);
  }

  async function ask(e) {
    e.preventDefault();
    const q = $("#question").value.trim();
    $("#ask-error").textContent = q.length < 2 ? "Type a question (at least 2 characters)." : "";
    if (q.length < 2 || !state.job) return;
    const btn = $("#ask-btn");
    setBusy(btn, true, "Searching…");
    try {
      const entry = await api(`/api/jobs/${state.job.id}/ask`, { method: "POST", body: { question: q }, timeout: 120000 });
      if (!$("#qa-list .qa")) $("#qa-list").replaceChildren();
      $("#qa-list").prepend(qaCard(entry));
      $("#question").value = "";
    } catch (err) {
      $("#ask-error").textContent = err.message;
    } finally {
      setBusy(btn, false);
    }
  }

  /* ------------------------------------------------------------------ */
  /* Wiring                                                             */
  /* ------------------------------------------------------------------ */
  initTabs($('[role="tablist"]'));
  $$("[data-source]").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.source)));
  $("#url-input").addEventListener("input", debounce(() => { $("#url-error").textContent = ""; refreshPreview(); }, 350));
  $("#file-input").addEventListener("change", (e) => pickFile(e.target.files[0]));
  const dz = $("#dropzone");
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("is-over"); }));
  dz.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) pickFile(e.dataTransfer.files[0]); });
  $("#source-form").addEventListener("submit", submit);
  $("#ask-form").addEventListener("submit", ask);
  $("#copy-transcript").addEventListener("click", () => state.job?.outputs?.transcript && copyText(state.job.outputs.transcript));
  $("#download-transcript").addEventListener("click", () => state.job?.outputs?.transcript && download("transcript.txt", state.job.outputs.transcript));

  (async () => {
    await loadHealth();
    const saved = storage.get(JOB_KEY);
    if (saved) {
      try {
        const job = await api(`/api/jobs/${saved}`, { timeout: 8000 });
        if (job.status === "running" || job.status === "queued") follow(job);
        else { renderJob(job); finish(job, null, true); }
      } catch { storage.set(JOB_KEY, null); }
    }
  })();
})();
