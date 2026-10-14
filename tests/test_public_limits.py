"""Tests for the public-deployment limits in server.py (no paid API calls: dummy keys, fake pipeline).

Run from the project root:  python -m unittest discover tests -v
"""
import io
import os
import sys
import time
import types
import unittest
import wave

# Dummy keys and limits must be set before server.py runs load_dotenv(), which never overrides set variables.
os.environ.update({
    "MISTRAL_API_KEY": "test-dummy-mistral", "SARVAM_API_KEY": "test-dummy-sarvam",
    "JOBS_PER_DAY": "3", "JOBS_PER_IP_PER_HOUR": "2", "QUESTIONS_PER_JOB": "2",
    "MAX_AUDIO_MINUTES": "0.05", "ALLOW_YOUTUBE": "0", "MAX_UPLOAD_MB": "5",
})
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def fake_run_job(job_id):
    job = server.JOBS[job_id]
    job.update(status="completed", started_at=time.time(), finished_at=time.time(), _rag_chain=object())
    job["outputs"]["transcript"] = "the launch moved to november"


fake_rag = types.ModuleType("core.rag_engine")
fake_rag.ask_question = lambda chain, q: "fake answer"
sys.modules["core.rag_engine"] = fake_rag


def wav_bytes(seconds):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(b"\x00\x00" * int(8000 * seconds))
    return buf.getvalue()


class PublicLimitTests(unittest.TestCase):
    def setUp(self):
        server.JOBS.clear(); server._JOB_STARTS.clear(); server._CLIENT_STARTS.clear()
        server._run_job = fake_run_job
        server._retrieved_passages = lambda chain, q, transcript: []
        self.client = TestClient(server.app)

    def upload(self, ip, seconds=1.0, data=None):
        files = {"file": ("clip.wav", data if data is not None else wav_bytes(seconds), "audio/wav")}
        return self.client.post("/api/jobs", data={"source_type": "file", "language": "english"}, files=files,
                                headers={"X-Forwarded-For": ip})

    def finish_all(self):
        for j in server.JOBS.values():
            j["status"] = "completed"

    def test_youtube_disabled(self):
        r = self.client.post("/api/jobs", data={"source_type": "url", "language": "english",
                                                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("YouTube links are disabled", r.json()["detail"])
        self.assertFalse(self.client.get("/api/health").json()["youtube_enabled"])

    def test_upload_accepted(self):
        r = self.upload("203.0.113.5")
        self.assertEqual(r.status_code, 202, r.text)

    def test_audio_length_limit(self):
        r = self.upload("203.0.113.5", seconds=5)          # limit is 0.05 min = 3 s
        self.assertEqual(r.status_code, 413)
        self.assertIn("accepts up to", r.json()["detail"])
        self.assertEqual(len(server._JOB_STARTS), 0)        # rejected uploads use no slot

    def test_unreadable_file(self):
        count = lambda: len(os.listdir(server.UPLOAD_ROOT)) if os.path.isdir(server.UPLOAD_ROOT) else 0
        before = count()
        r = self.upload("203.0.113.5", data=b"not audio at all")
        self.assertEqual(r.status_code, 422)
        self.assertEqual(count(), before)                      # the rejected upload was deleted
        self.assertEqual(len(server._JOB_STARTS), 0)

    def test_per_client_hourly_limit(self):
        for _ in range(2):
            self.assertEqual(self.upload("203.0.113.5").status_code, 202); self.finish_all()
        r = self.upload("203.0.113.5")
        self.assertEqual(r.status_code, 429)
        self.assertIn("in the last hour", r.json()["detail"])
        self.assertEqual(self.upload("198.51.100.7").status_code, 202)   # another visitor is unaffected

    def test_daily_limit(self):
        for i in range(3):
            self.assertEqual(self.upload(f"203.0.113.{i + 10}").status_code, 202); self.finish_all()
        r = self.upload("198.51.100.99")
        self.assertEqual(r.status_code, 429)
        self.assertIn("per day", r.json()["detail"])

    def test_limits_expire(self):
        server._JOB_STARTS.extend([time.time() - 86401] * 3)
        server._CLIENT_STARTS["203.0.113.5"] = server.deque([time.time() - 3601] * 2)
        self.assertEqual(self.upload("203.0.113.5").status_code, 202)

    def test_question_limit(self):
        job_id = self.upload("203.0.113.5").json()["id"]
        for _ in range(2):
            self.assertEqual(self.client.post(f"/api/jobs/{job_id}/ask", json={"question": "when is the launch?"}).status_code, 200)
        r = self.client.post(f"/api/jobs/{job_id}/ask", json={"question": "and the budget?"})
        self.assertEqual(r.status_code, 429)
        self.assertIn("questions per analysis", r.json()["detail"])

    def test_health_reports_limits(self):
        self.upload("203.0.113.5"); self.finish_all()
        lim = self.client.get("/api/health", headers={"X-Forwarded-For": "203.0.113.5"}).json()["limits"]
        self.assertEqual(lim["analyses_left_today"], 2)
        self.assertEqual(lim["your_analyses_left_this_hour"], 1)
        self.assertEqual(lim["questions_per_analysis"], 2)


if __name__ == "__main__":
    unittest.main()
