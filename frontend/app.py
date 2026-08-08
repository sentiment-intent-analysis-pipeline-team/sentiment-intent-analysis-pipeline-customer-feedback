import streamlit as st
import pandas as pd
import sys
import os

# Allow importing from backend/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from pipeline import analyze_feedback
from database import save_feedback_result, get_all_feedback

st.set_page_config(page_title="Sentiment & Intent Analysis", layout="wide")

st.title("Sentiment and Intent Analysis Pipeline for Customer Feedback")
st.caption("Submit feedback below to see sentiment, intent, and confidence scores.")

with st.form("feedback_form"):
    text_input = st.text_area("Enter customer feedback", height=100)
    submitted = st.form_submit_button("Analyze")

if submitted and text_input.strip():
    result = analyze_feedback(text_input)
    result["needs_review"] = result["intent_confidence"] < 0.3
    save_feedback_result(result)

    col1, col2, col3 = st.columns(3)
    col1.metric("Sentiment", result["sentiment"], f"{result['sentiment_confidence']:.0%} confidence")
    col2.metric("Intent", result["intent"], f"{result['intent_confidence']:.0%} confidence")
    col3.metric("Needs Review", "Yes" if result["needs_review"] else "No")

    if result["needs_review"]:
        st.warning("Low confidence prediction — flagged for human review.")
    else:
        st.success("Analysis complete.")

st.divider()
st.subheader("Feedback History")

records = get_all_feedback()
if records:
    df = pd.DataFrame([{
        "id": r.id,
        "original_text": r.original_text,
        "sentiment": r.sentiment,
        "intent": r.intent,
        "sentiment_confidence": r.sentiment_confidence,
        "intent_confidence": r.intent_confidence,
        "needs_review": r.needs_review,
        "created_at": r.created_at.isoformat()
    } for r in records])

    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Sentiment distribution")
        st.bar_chart(df["sentiment"].value_counts())
    with col2:
        st.write("Top intents")
        st.bar_chart(df["intent"].value_counts().head(10))
else:
    st.info("No feedback analyzed yet. Submit some above.")