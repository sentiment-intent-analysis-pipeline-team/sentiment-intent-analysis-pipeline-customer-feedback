from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import analyze_feedback
from database import save_feedback_result, get_all_feedback

app = FastAPI(title="Sentiment & Intent Analysis API")

class FeedbackRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Sentiment & Intent Analysis API is running"}

@app.post("/analyze")
def analyze(request: FeedbackRequest):
    result = analyze_feedback(request.text)
    result["needs_review"] = result["intent_confidence"] < 0.3

    save_feedback_result(result)

    return result

@app.get("/history")
def history():
    records = get_all_feedback()
    return [
        {
            "id": r.id,
            "original_text": r.original_text,
            "sentiment": r.sentiment,
            "sentiment_confidence": r.sentiment_confidence,
            "intent": r.intent,
            "intent_confidence": r.intent_confidence,
            "needs_review": r.needs_review,
            "created_at": r.created_at.isoformat()
        }
        for r in records
    ]