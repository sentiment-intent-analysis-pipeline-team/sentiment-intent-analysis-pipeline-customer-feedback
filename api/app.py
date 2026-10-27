"""Thin HTTP wrapper around the existing pipeline. ML code is NOT changed."""
import hmac
import os
import sys

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from pipeline import analyze_feedback  # noqa: E402  (loads all models once at startup)

API_KEY = os.environ.get("API_KEY", "")
app = FastAPI(title="Feedback analysis API")


class AnalyzeIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(body: AnalyzeIn, x_api_key: str | None = Header(default=None)):
    if not API_KEY or not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text is empty")
    result = analyze_feedback(text)
    result["needs_review"] = result["intent_confidence"] < 0.3
    return result
