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

# All rule-based logic lives in heuristics.py (plain Python, unit tested in
# test_heuristics.py). The detect_* functions are re-exported here so any
# existing code that does `from pipeline import detect_sarcasm_cue` still works.
try:
    from heuristics import (
        detect_sarcasm_cue, detect_resolution_cue, detect_negated_positive,
        detect_wrong_item_cue, detect_malfunction_cue, detect_negative_event_cue,
        detect_damage_cue, has_unnegated_negative_context, resolve_labels,
    )
except ImportError:  # imported as a package, e.g. backend.pipeline
    from .heuristics import (
        detect_sarcasm_cue, detect_resolution_cue, detect_negated_positive,
        detect_wrong_item_cue, detect_malfunction_cue, detect_negative_event_cue,
        detect_damage_cue, has_unnegated_negative_context, resolve_labels,
    )

# Load models once when this file is imported (not every time a function is called)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stop_words = set(stopwords.words('english'))

sentiment_model = joblib.load(os.path.join(MODELS_DIR, 'sentiment_model.pkl'))
sentiment_vectorizer = joblib.load(os.path.join(MODELS_DIR, 'sentiment_vectorizer.pkl'))

SENTIMENT_LABELS = ['negative', 'neutral', 'positive']

intent_model = joblib.load(os.path.join(MODELS_DIR, 'intent_model.pkl'))
intent_vectorizer = joblib.load(os.path.join(MODELS_DIR, 'intent_vectorizer.pkl'))

with open(os.path.join(MODELS_DIR, 'intent_label_map.json'), 'r') as f:
    INTENT_LABEL_MAP = {int(k): v for k, v in json.load(f).items()}

# Feedback-type model (complaint / praise / suggestion / query).
# Trained directly on string labels (no label-map JSON, unlike intent), so
# type_model.predict() returns the label string itself.
type_model = joblib.load(os.path.join(MODELS_DIR, 'type_model.pkl'))
type_vectorizer = joblib.load(os.path.join(MODELS_DIR, 'type_vectorizer.pkl'))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and len(token.text) > 1]
    return ' '.join(tokens)


def light_clean(text):
    """Lighter cleaning for the feedback-type model. Deliberately keeps
    stopwords (how, can, would, should...) and question marks, since those
    are exactly what distinguish a query or suggestion from a complaint or
    praise. No stopword removal, no lemmatization.

    IMPORTANT: this must stay identical to light_clean in the training
    notebook (cell 43). Punctuation is replaced with a SPACE, not deleted -
    otherwise 'two-factor' is trained as 'two factor' but served as
    'twofactor', a word the model has never seen."""
    text = str(text).lower().replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r"[^a-z\s'?]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def analyze_feedback(text):
    """
    Takes raw feedback text, returns sentiment + intent + feedback_type +
    confidence scores.

    Flow: the three TF-IDF models make their raw predictions, then
    heuristics.resolve_labels() applies the rule layer on top of the
    sentiment and feedback-type predictions (negation, sarcasm, relief,
    wrong-item, malfunction, damage, neutral-by-structure for plain
    questions and suggestions).
    """
    cleaned = clean_text(text)

    # Raw sentiment prediction
    sent_vec = sentiment_vectorizer.transform([cleaned])
    sentiment_pred = sentiment_model.predict(sent_vec)[0]
    sentiment_proba = sentiment_model.predict_proba(sent_vec)[0]
    sentiment_confidence = float(max(sentiment_proba))
    sentiment_label_raw = SENTIMENT_LABELS[sentiment_pred]

    # Intent prediction with confidence
    intent_vec = intent_vectorizer.transform([cleaned])
    intent_pred = intent_model.predict(intent_vec)[0]
    intent_proba = intent_model.predict_proba(intent_vec)[0]
    intent_confidence = float(max(intent_proba))

    # Raw feedback-type prediction (complaint / praise / suggestion / query)
    type_vec = type_vectorizer.transform([light_clean(text)])
    type_pred = type_model.predict(type_vec)[0]
    type_proba = type_model.predict_proba(type_vec)[0]
    type_confidence = float(max(type_proba))

    # Rule layer on top of the raw predictions
    final = resolve_labels(text, sentiment_label_raw, sentiment_confidence,
                           type_pred, type_confidence)
    flags = final['flags']

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "sentiment": final['sentiment'],
        "sarcasm_detected": flags['sarcasm'],
        "resolution_detected": flags['resolution'],
        "negated_positive_detected": flags['negated_positive'],
        "wrong_item_detected": flags['wrong_item'],
        "malfunction_detected": flags['malfunction'],
        "negative_event_detected": flags['negative_event'],
        "damage_detected": flags['damage'],
        "structural_neutral_correction": final['structural_neutral'],
        "hypothetical_query": final['hypothetical_query'],
        "type_correction": final['type_fix'],
        "sentiment_confidence": round(final['sentiment_confidence'], 3),
        "intent": INTENT_LABEL_MAP.get(intent_pred, "unknown"),
        "intent_confidence": round(intent_confidence, 3),
        "feedback_type": final['feedback_type'],
        "feedback_type_confidence": round(type_confidence, 3)
    }