
"""Thin HTTP wrapper around the existing pipeline. ML code is NOT changed."""
import hmac
import os
import sys

import psutil
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from pipeline import analyze_feedback  # noqa: E402

API_KEY = os.environ.get("API_KEY", "")
RENDER_MEMORY_LIMIT_MB = 512

app = FastAPI(title="Feedback analysis API")


class AnalyzeIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


def verify_api_key(x_api_key: str | None):
    if not API_KEY or not x_api_key or not hmac.compare_digest(
        x_api_key, API_KEY
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/memory")
def memory(x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)

    process = psutil.Process(os.getpid())
    rss_bytes = process.memory_info().rss
    rss_mb = round(rss_bytes / (1024 * 1024), 2)
    usage_percent = round(
        (rss_mb / RENDER_MEMORY_LIMIT_MB) * 100, 2
    )

    return {
        "process_id": process.pid,
        "memory_usage_mb": rss_mb,
        "render_memory_limit_mb": RENDER_MEMORY_LIMIT_MB,
        "usage_percent": usage_percent,
        "headroom_mb": round(RENDER_MEMORY_LIMIT_MB - rss_mb, 2),
        "target_below_450_mb": rss_mb < 450,
    }


@app.post("/analyze")
def analyze(body: AnalyzeIn, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text is empty")

    result = analyze_feedback(text)
    result["needs_review"] = result["intent_confidence"] < 0.3
    return result
