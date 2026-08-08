import os
import nltk
nltk.download('stopwords', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # this file's own folder (backend/)
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')     # go up one level, into models/

import json
import re
import spacy
import joblib
from nltk.corpus import stopwords

# Load models once when this file is imported (not every time a function is called)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stop_words = set(stopwords.words('english'))

from transformers import pipeline as hf_pipeline

sentiment_pipeline = hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
intent_model = joblib.load(os.path.join(MODELS_DIR, 'intent_model.pkl'))
intent_vectorizer = joblib.load(os.path.join(MODELS_DIR, 'intent_vectorizer.pkl'))

with open(os.path.join(MODELS_DIR, 'intent_label_map.json'), 'r') as f:
    INTENT_LABEL_MAP = {int(k): v for k, v in json.load(f).items()}

SENTIMENT_LABELS = ['negative', 'neutral', 'positive']

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and len(token.text) > 1]
    return ' '.join(tokens)

def analyze_feedback(text):
    """
    Takes raw feedback text, returns sentiment + intent + confidence scores.
    intent_label_map: a dict mapping intent label numbers to intent names
    """
    cleaned = clean_text(text)

    # Sentiment prediction with confidence
    sentiment_result = sentiment_pipeline(text)[0]  # note: use original text, not cleaned - transformers handle raw text better
    sentiment_label_raw = sentiment_result['label'].lower()
    sentiment_confidence = float(sentiment_result['score'])

    # Intent prediction with confidence
    intent_vec = intent_vectorizer.transform([cleaned])
    intent_pred = intent_model.predict(intent_vec)[0]
    intent_proba = intent_model.predict_proba(intent_vec)[0]
    intent_confidence = float(max(intent_proba))

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "sentiment": sentiment_label_raw,
        "sentiment_confidence": round(sentiment_confidence, 3),
        "intent": INTENT_LABEL_MAP.get(intent_pred, "unknown"),
        "intent_confidence": round(intent_confidence, 3)
    }