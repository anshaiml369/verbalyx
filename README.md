# Verbalyx — transcribe, analyse and question spoken content

A portfolio project that began from a YouTube tutorial and was extended with a web studio (replacing the original Streamlit UI), display of the retrieved evidence behind each answer, a fix for cross-video retrieval contamination and documented limitations. Given a YouTube link or an uploaded audio/video file, the assistant:

1. extracts the audio and splits it into 10-minute chunks (`utils/audio_processor.py`, yt-dlp + pydub/ffmpeg);
2. transcribes it with local **Whisper** (English) or the **Sarvam** speech-to-text-translate API (Hinglish → English);
3. uses **Mistral** (`MISTRAL_MODEL`, default `mistral-small-latest`) to generate a title, a map-reduce summary, action items, key decisions
   and open questions;
4. indexes the transcript in **Chroma** with `all-MiniLM-L6-v2` embeddings and answers questions with a
   retrieval-augmented chain (top-4 chunks).

It analyses speech only, not the images in the video. The pipeline does not keep timestamps or speaker labels.

## Interface

A plain HTML/CSS/JavaScript studio served by FastAPI (`server.py`):

- **Source:** YouTube link (embedded preview) or file upload (local preview); language selection.
- **Processing:** the six real stages with live status and durations, plus the server log for the job.
- **Analysis:** title, summary, action items, decisions, open questions and the full transcript (copy/download),
  each appearing as soon as its stage finishes.
- **Ask:** each answer is shown next to the transcript passages that were retrieved for it, with chunk index and
  L2 distance. As a safeguard, any passage that does not occur in the current transcript is flagged.

## Architecture

```
static/                     HTML/CSS/vanilla-JS studio
  css/core.css, js/core.js  shared design system (synced from ../../_design_system)
  css/app.css, js/app.js    project UI
server.py                   NEW FastAPI server: jobs, uploads, Q&A, static files
core/, utils/               EXISTING pipeline (unchanged)
main.py                     EXISTING CLI entry point (unchanged)
test.py                     EXISTING manual test script (unchanged)
```

`server.py` calls the existing functions in the same order as the former Streamlit `app.py` and
`main.run_pipeline()`. For Q&A it calls the existing `ask_question()`. To show evidence, it reads the retriever
already inside the chain returned by `build_rag_chain()` and repeats the same top-k similarity search. Apart from the approved vector-store fix
(see below), no file in `core/` or `utils/` was modified. The Streamlit UI (`app.py`) was removed.

Web-specific safeguards in `server.py`: only YouTube URLs are accepted (no arbitrary local paths or other hosts);
uploads are limited by extension and size (`MAX_UPLOAD_MB`, default 500); each upload is stored in its own folder
under `downloades/uploads/` and deleted, with its derived WAV chunks, when the job finishes; one job runs at a time.
API keys stay on the server and are redacted from error messages and logs.

## Run locally

Requirements: Python 3.10+ and the **ffmpeg** binary (`brew install ffmpeg` on macOS, `apt install ffmpeg` on
Debian/Ubuntu).

```bash
cd GENAI_AGENTIC_AI/AI_VIDEO_ASSISTANT_RAG
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r Requirements.txt
cp .env.example .env        # add MISTRAL_API_KEY (and SARVAM_API_KEY for Hinglish)
uvicorn server:app --port 8000
```

Open <http://localhost:8000>. The first transcription downloads the Whisper model (`small` ≈ 460 MB), and the first
index build downloads the MiniLM embedding model (≈ 90 MB). The CLI still works: `python main.py`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Key presence (booleans), ffmpeg availability, Whisper model, upload limit |
| POST | `/api/jobs` (multipart: `source_type`, `url` or `file`, `language`) | Start an analysis (202); 503 not configured, 429 busy, 413/415/422 invalid input |
| GET | `/api/jobs/{id}` | Stage status/timings, outputs so far, Q&A history, log |
| POST | `/api/jobs/{id}/ask` `{"question": "…"}` | Answer + retrieved passages |

## Changes to the existing pipeline

- **Vector store reset (approved fix, `core/vector_store.py`).** `build_vector_store()` used
  `Chroma.from_documents(...)` on a persisted collection, which *added* chunks to previous videos' chunks; a question
  about one video could retrieve passages from another (4 of 4 in a local test). The function now deletes the
  `meeting_transcript` collection before indexing a new transcript. After the fix, the same test indexed only the new
  video's chunks and retrieved 0 passages from the previous one. Consequence: only the most recently analysed video
  can be questioned; the UI always works with the latest job.

- **Configurable Mistral model (approved change, `core/summarizer.py`, `core/extractor.py`, `core/rag_engine.py`).**
  The model name was hard-coded as `mistral-small-latest`. Each of the three `get_llm()` functions now reads
  `os.getenv("MISTRAL_MODEL", "mistral-small-latest")`, so behaviour is unchanged unless the variable is set. Set
  `MISTRAL_MODEL=ministral-8b-latest` (or another chat model your Mistral plan allows) in `.env` if
  `mistral-small-latest` is rate-limited on your plan.

## Known limitations in the pipeline

1. No timestamps: `transcribe_chunk_whisper` keeps only `result["text"]`, so answers cannot cite moments in the video.
2. Extraction prompts send the full transcript in one request and may exceed the model's context for long recordings.
3. One shared index: analysing a new video replaces the previous index (see above).

## Deployment notes

- Required: `MISTRAL_API_KEY`, ffmpeg on the host. Optional: `SARVAM_API_KEY`, `WHISPER_MODEL`, `MAX_UPLOAD_MB`.
- Whisper on CPU is slow for long videos; size the host accordingly or use a smaller `WHISPER_MODEL`.
- `downloades/` and `vector_db/` are runtime data and are git-ignored.
- YouTube downloading via yt-dlp can break when YouTube changes and may be blocked from cloud IPs; respect
  YouTube's terms and only analyse media you have rights to.
- Public deployment should add authentication and rate limiting (paid LLM calls, heavy CPU work).
