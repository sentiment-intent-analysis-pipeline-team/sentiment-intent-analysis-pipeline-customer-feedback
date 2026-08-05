import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Sentiment & Intent Analysis", layout="wide")

st.title("Sentiment and Intent Analysis Pipeline for Customer Feedback")
st.caption("Submit feedback below to see sentiment, intent, and confidence scores.")

# --- Feedback submission form ---
with st.form("feedback_form"):
    text_input = st.text_area("Enter customer feedback", height=100)
    submitted = st.form_submit_button("Analyze")

if submitted and text_input.strip():
    try:
        response = requests.post(f"{API_URL}/analyze", json={"text": text_input})
        response.raise_for_status()
        result = response.json()

        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment", result["sentiment"], f"{result['sentiment_confidence']:.0%} confidence")
        col2.metric("Intent", result["intent"], f"{result['intent_confidence']:.0%} confidence")
        col3.metric("Needs Review", "Yes" if result["needs_review"] else "No")

        if result["needs_review"]:
            st.warning("Low confidence prediction — flagged for human review.")
        else:
            st.success("Analysis complete.")

    except requests.exceptions.ConnectionError:
        st.error("Could not reach the backend API. Make sure the FastAPI server is running on port 8000.")

st.divider()

# --- History / trends ---
st.subheader("Feedback History")

try:
    history_response = requests.get(f"{API_URL}/history")
    history_response.raise_for_status()
    history_data = history_response.json()

    if history_data:
        df = pd.DataFrame(history_data)
        df = df[["id", "original_text", "sentiment", "intent", "sentiment_confidence", "intent_confidence", "needs_review", "created_at"]]
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

except requests.exceptions.ConnectionError:
    st.error("Could not reach the backend API.")