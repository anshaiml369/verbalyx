"""
HTTP API + static frontend for Verbalyx.

Replaces the Streamlit UI (app.py). It calls the existing pipeline functions
unchanged and in the same order as app.py / main.run_pipeline:

    process_input → transcribe_all → generate_title → summarize →
    extract_action_items / extract_key_decisions / extract_questions → build_rag_chain

Questions are answered with the existing core.rag_engine.ask_question(). To show
which transcript passages were retrieved, the server reads the retriever that is
already inside the returned LCEL chain and repeats the same top-k similarity
search; nothing in core/ or utils/ is modified.

Run:  uvicorn server:app --port 8000
"""
import io
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()  # must run before core/ is imported: transcriber reads env vars at import time

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_ROOT = os.path.join("downloades", "uploads")  # inside the folder utils/audio_processor already uses
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"}
LANGUAGES = {"english", "hinglish"}
STAGES = ["audio", "transcript", "title", "summary", "extract", "rag"]

# Limits for a public deployment: each analysis and each question makes paid Mistral calls, and
# transcription is CPU-heavy. 0 means "no limit", so local use is unchanged unless these are set.
JOBS_PER_DAY = int(os.getenv("JOBS_PER_DAY", "0"))
JOBS_PER_IP_PER_HOUR = int(os.getenv("JOBS_PER_IP_PER_HOUR", "0"))
QUESTIONS_PER_JOB = int(os.getenv("QUESTIONS_PER_JOB", "0"))
MAX_AUDIO_MINUTES = float(os.getenv("MAX_AUDIO_MINUTES", "0"))
# YouTube often blocks downloads from cloud servers; a public deployment can accept uploads only.
ALLOW_YOUTUBE = os.getenv("ALLOW_YOUTUBE", "1") == "1"


class _ThreadStdout(io.TextIOBase):
    """Routes print() output from job threads to that job's log; still echoes to the console."""

    def __init__(self, base):
        self.base = base
        self.sinks = {}

    def write(self, s):
        sink = self.sinks.get(threading.get_ident())
        if sink is not None:
            sink(s)
        return self.base.write(s)

    def flush(self):
        self.base.flush()

    def isatty(self):
        return False


_stdout = _ThreadStdout(sys.stdout)
sys.stdout = _stdout

JOBS = {}
JOBS_LOCK = threading.Lock()
_JOB_STARTS = deque()   # start times of accepted analyses in the last 24 h
_CLIENT_STARTS = {}     # client address -> start times of its analyses in the last hour


def _client_id(request):
    """Best-effort client address: a proxy-supplied header when present, else the socket peer."""
    for header in ("true-client-ip", "cf-connecting-ip"):
        if request.headers.get(header):
            return request.headers[header].strip()
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _usage_left(client, now):
    """Drop expired entries; return (analyses left today, left this hour for client), None = unlimited. Hold JOBS_LOCK."""
    while _JOB_STARTS and now - _JOB_STARTS[0] >= 86400:
        _JOB_STARTS.popleft()
    for key in list(_CLIENT_STARTS):
        starts = _CLIENT_STARTS[key]
        while starts and now - starts[0] >= 3600:
            starts.popleft()
        if not starts:
            del _CLIENT_STARTS[key]
    day = JOBS_PER_DAY - len(_JOB_STARTS) if JOBS_PER_DAY else None
    hour = JOBS_PER_IP_PER_HOUR - len(_CLIENT_STARTS.get(client, ())) if JOBS_PER_IP_PER_HOUR else None
    return day, hour


def _minutes(seconds):
    m = max(1, round(seconds / 60))
    return f"{m} minute{'s' if m != 1 else ''}" if m < 120 else f"{round(m / 60)} hours"


def _media_seconds(path):
    """Duration of an uploaded file via ffprobe, or None if it cannot be read."""
    import subprocess
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out)
    except Exception:
        return None


def _redact(text):
    for name in ("MISTRAL_API_KEY", "SARVAM_API_KEY"):
        val = os.getenv(name)
        if val:
            text = text.replace(val, "[redacted]")
    return text


def _requirements():
    return {
        "mistral_key": bool(os.getenv("MISTRAL_API_KEY")),
        "sarvam_key": bool(os.getenv("SARVAM_API_KEY")),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "whisper_model": os.getenv("WHISPER_MODEL", "small"),
        "mistral_model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        "max_upload_mb": MAX_UPLOAD_MB,
        "youtube_enabled": ALLOW_YOUTUBE,
        "max_audio_minutes": MAX_AUDIO_MINUTES or None,
    }


def _youtube_id(url):
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if host.endswith("youtu.be"):
        return p.path.lstrip("/").split("/")[0] or None
    m = re.search(r"(?:v=|/embed/|/shorts/|/live/)([A-Za-z0-9_-]{6,})", url)
    return m.group(1) if m else None


def _public(job):
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _run_job(job_id):
    job = JOBS[job_id]

    def sink(text):
        if len(job["log"]) < 100_000:
            job["log"] += text

    _stdout.sinks[threading.get_ident()] = sink
    job["status"] = "running"
    job["started_at"] = time.time()
    current = {"stage": None}

    def stage(name):
        now = time.time()
        if current["stage"]:
            s = job["stages"][current["stage"]]
            s["status"], s["ended_at"] = "done", now
        current["stage"] = name
        if name:
            job["stages"][name].update(status="running", started_at=now)

    try:
        from core.extractor import extract_action_items, extract_key_decisions, extract_questions
        from core.rag_engine import build_rag_chain
        from core.summarizer import generate_title, summarize
        from core.transcriber import transcribe_all
        from utils.audio_processor import process_input

        out = job["outputs"]
        stage("audio")
        chunks = process_input(job["_source"])
        out["chunk_count"] = len(chunks)

        stage("transcript")
        transcript = transcribe_all(chunks, job["language"])
        out["transcript"] = transcript

        stage("title")
        out["title"] = generate_title(transcript)

        stage("summary")
        out["summary"] = summarize(transcript)

        stage("extract")
        out["action_items"] = extract_action_items(transcript)
        out["key_decisions"] = extract_key_decisions(transcript)
        out["open_questions"] = extract_questions(transcript)

        stage("rag")
        job["_rag_chain"] = build_rag_chain(transcript)
        stage(None)
        job["status"] = "completed"
    except Exception as exc:
        if current["stage"]:
            job["stages"][current["stage"]].update(status="error", ended_at=time.time())
        job["status"] = "failed"
        job["error"] = {"type": type(exc).__name__, "message": _redact(str(exc))[:2000], "stage": current["stage"]}
        job["log"] += "\n" + _redact(traceback.format_exc())[-4000:]
    finally:
        job["finished_at"] = time.time()
        _stdout.sinks.pop(threading.get_ident(), None)
        if job["_upload_dir"]:
            shutil.rmtree(job["_upload_dir"], ignore_errors=True)  # server-created copy + derived WAV chunks


def _retrieved_passages(chain, question, transcript):
    """Repeat the chain's own retrieval (same retriever, same k) to show what it used as context."""
    try:
        retriever = chain.first.steps__["context"].first
        k = retriever.search_kwargs.get("k", 4)
        results = retriever.vectorstore.similarity_search_with_score(question, k=k)
    except Exception:
        return None
    return [
        {
            "rank": i + 1,
            "text": doc.page_content,
            "chunk_index": doc.metadata.get("chunk_index"),
            "distance": float(score),
            "in_current_transcript": doc.page_content in transcript,
        }
        for i, (doc, score) in enumerate(results)
    ]


app = FastAPI(title="Verbalyx API", description="Transcription, LLM analysis and RAG Q&A over a video's audio")


@app.get("/api/health")
def health(request: Request):
    req = _requirements()
    with JOBS_LOCK:
        day_left, hour_left = _usage_left(_client_id(request), time.time())
    return {
        "status": "ok",
        **req,
        "ready": req["mistral_key"] and req["ffmpeg"],
        "active_jobs": sum(1 for j in JOBS.values() if j["status"] in ("queued", "running")),
        "limits": {"analyses_per_day": JOBS_PER_DAY or None, "analyses_per_client_per_hour": JOBS_PER_IP_PER_HOUR or None,
                   "questions_per_analysis": QUESTIONS_PER_JOB or None,
                   "analyses_left_today": None if day_left is None else max(0, day_left),
                   "your_analyses_left_this_hour": None if hour_left is None else max(0, hour_left)},
    }


@app.post("/api/jobs", status_code=202)
async def create_job(
    request: Request,
    source_type: str = Form(...),
    language: str = Form("english"),
    url: str = Form(None),
    file: UploadFile = File(None),
):
    language = language.lower().strip()
    if language not in LANGUAGES:
        raise HTTPException(422, "language must be 'english' or 'hinglish'")
    req = _requirements()
    missing = []
    if not req["mistral_key"]:
        missing.append("MISTRAL_API_KEY is not set (needed for title, summary, extraction and Q&A)")
    if not req["ffmpeg"]:
        missing.append("ffmpeg is not installed or not on PATH (needed to decode audio)")
    if language == "hinglish" and not req["sarvam_key"]:
        missing.append("SARVAM_API_KEY is not set (needed for Hinglish transcription)")
    if missing:
        raise HTTPException(503, {"message": "The server is not ready: " + "; ".join(missing) + ".", "missing": missing})

    if source_type == "url" and not ALLOW_YOUTUBE:
        raise HTTPException(422, "YouTube links are disabled on this public demo (YouTube often blocks cloud servers). "
                                 "Upload an audio or video file instead.")
    client = _client_id(request)
    with JOBS_LOCK:
        if any(j["status"] in ("queued", "running") for j in JOBS.values()):
            raise HTTPException(429, "Another video is being processed. Wait for it to finish, then try again.")
        now = time.time()
        day_left, hour_left = _usage_left(client, now)
        if day_left is not None and day_left <= 0:
            raise HTTPException(429, f"The demo's limit of {JOBS_PER_DAY} analyses per day has been reached (each one makes "
                                     f"paid API calls). Try again in about {_minutes(86400 - (now - _JOB_STARTS[0]))}.")
        if hour_left is not None and hour_left <= 0:
            raise HTTPException(429, f"You have started {JOBS_PER_IP_PER_HOUR} analyses in the last hour. Try again in about "
                                     f"{_minutes(3600 - (now - _CLIENT_STARTS[client][0]))}.")

    job_id = uuid.uuid4().hex[:12]
    upload_dir = None
    source_meta = {"type": source_type}
    if source_type == "url":
        if not url:
            raise HTTPException(422, "Provide a YouTube URL.")
        url = url.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or (parsed.hostname or "").lower() not in YOUTUBE_HOSTS:
            raise HTTPException(422, "Only YouTube links (youtube.com or youtu.be) are accepted.")
        vid = _youtube_id(url)
        if not vid:
            raise HTTPException(422, "Could not find a video ID in that YouTube link.")
        source = url
        source_meta.update(url=url, youtube_id=vid)
    elif source_type == "file":
        if file is None or not file.filename:
            raise HTTPException(422, "Choose a media file to upload.")
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(415, f"Unsupported file type '{ext or 'none'}'. Allowed: {', '.join(sorted(ALLOWED_EXT))}.")
        upload_dir = os.path.join(UPLOAD_ROOT, job_id)
        os.makedirs(upload_dir, exist_ok=True)
        dest = os.path.join(upload_dir, f"upload{ext}")
        size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
        with open(dest, "wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    fh.close()
                    shutil.rmtree(upload_dir, ignore_errors=True)
                    raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
                fh.write(chunk)
        if size == 0:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(422, "The uploaded file is empty.")
        if MAX_AUDIO_MINUTES:
            seconds = _media_seconds(dest)
            if seconds is None:
                shutil.rmtree(upload_dir, ignore_errors=True)
                raise HTTPException(422, "Could not read the length of that file. Upload a standard audio or video file.")
            if seconds > MAX_AUDIO_MINUTES * 60:
                shutil.rmtree(upload_dir, ignore_errors=True)
                raise HTTPException(413, f"That recording is {seconds / 60:.1f} minutes long; this demo accepts up to "
                                         f"{MAX_AUDIO_MINUTES:g} minutes.")
        source = dest
        source_meta.update(filename=os.path.basename(file.filename)[:200], size_bytes=size)
    else:
        raise HTTPException(422, "source_type must be 'url' or 'file'.")

    with JOBS_LOCK:
        now = time.time()
        _JOB_STARTS.append(now)
        _CLIENT_STARTS.setdefault(client, deque()).append(now)
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "language": language,
            "source": source_meta,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "stages": {s: {"status": "pending", "started_at": None, "ended_at": None} for s in STAGES},
            "outputs": {},
            "qa": [],
            "error": None,
            "log": "",
            "_source": source,
            "_upload_dir": upload_dir,
            "_rag_chain": None,
            "_ask_lock": threading.Lock(),
        }
    threading.Thread(target=_run_job, args=(job_id,), daemon=True, name=f"job-{job_id}").start()
    return _public(JOBS[job_id])


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found (jobs are kept in memory and are lost when the server restarts).")
    snap = _public(job)
    snap["log"] = _redact(job["log"][-20_000:])
    return snap


class Question(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)


@app.post("/api/jobs/{job_id}/ask")
def ask(job_id: str, q: Question):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    if job["status"] != "completed" or job["_rag_chain"] is None:
        raise HTTPException(409, "The video has not finished processing, so there is no index to search yet.")
    from core.rag_engine import ask_question

    question = " ".join(q.question.split())
    if QUESTIONS_PER_JOB and len(job["qa"]) >= QUESTIONS_PER_JOB:
        raise HTTPException(429, f"This demo allows {QUESTIONS_PER_JOB} questions per analysis (each one makes a paid API call).")
    with job["_ask_lock"]:
        started = time.time()
        try:
            answer = ask_question(job["_rag_chain"], question)
        except Exception as exc:
            raise HTTPException(502, f"The language model call failed: {_redact(str(exc))[:500]}")
        passages = _retrieved_passages(job["_rag_chain"], question, job["outputs"].get("transcript", ""))
        entry = {"question": question, "answer": answer, "passages": passages,
                 "latency_s": time.time() - started, "asked_at": started}
        job["qa"].append(entry)
    return entry


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
