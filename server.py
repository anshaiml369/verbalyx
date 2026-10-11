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
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()  # must run before core/ is imported: transcriber reads env vars at import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
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
def health():
    req = _requirements()
    return {
        "status": "ok",
        **req,
        "ready": req["mistral_key"] and req["ffmpeg"],
        "active_jobs": sum(1 for j in JOBS.values() if j["status"] in ("queued", "running")),
    }


@app.post("/api/jobs", status_code=202)
async def create_job(
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

    with JOBS_LOCK:
        if any(j["status"] in ("queued", "running") for j in JOBS.values()):
            raise HTTPException(429, "Another video is being processed. Wait for it to finish, then try again.")

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
        source = dest
        source_meta.update(filename=os.path.basename(file.filename)[:200], size_bytes=size)
    else:
        raise HTTPException(422, "source_type must be 'url' or 'file'.")

    with JOBS_LOCK:
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
