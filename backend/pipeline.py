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

sentiment_model = joblib.load(os.path.join(MODELS_DIR, 'sentiment_model.pkl'))
sentiment_vectorizer = joblib.load(os.path.join(MODELS_DIR, 'sentiment_vectorizer.pkl'))

SENTIMENT_LABELS = ['negative', 'neutral', 'positive']

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

def has_unnegated_negative_context(text_lower, negative_context):
    """Check for negative-context words, but ignore ones that are negated
    (e.g. 'has not crashed' should NOT count as a negative event)."""
    negation_words = ['not ', "n't ", 'never ', 'no longer ', 'without ']
    for word in negative_context:
        for m in re.finditer(re.escape(word), text_lower):
            start = m.start()
            preceding = text_lower[max(0, start - 20):start]
            if any(neg in preceding for neg in negation_words):
                continue
            return True
    return False



def detect_sarcasm_cue(text):
    """Detect likely sarcasm: positive-sounding words paired with an actual
    (non-negated) negative event."""
    text_lower = text.lower()

    positive_cues = ['great', 'wow', 'wonderful', 'fantastic', 'brilliant', 'perfect',
                      'love', 'thanks', 'thank you', 'amazing', 'best', 'nice', 'awesome',
                      'excellent', 'superb', 'outstanding']

    negative_context = ['broke', 'break', 'broken', 'crash', 'crashed', 'fail', 'failed',
                         'failure', 'declin', 'block', 'stuck', 'delay', 'wait',
                         'charged twice', 'double charg', 'error', 'bug', 'glitch', 'down',
                         'outage', 'cancel', 'lost', 'lose', 'hold', 'refund', 'complain',
                         'annoy', 'frustrat', 'terrible', 'wrong', 
                         'log me out', 'logged out', 'log out', 'kicked out', 'timed out']

    has_positive_cue = any(word in text_lower for word in positive_cues)
    has_real_negative_context = has_unnegated_negative_context(text_lower, negative_context)

    return has_positive_cue and has_real_negative_context

def detect_resolution_cue(text):
    """Detect genuine positive resolution: an explicit fix, a negated negative
    event, or a contrastive turnaround ('was ready to complain, but it worked')."""
    text_lower = text.lower()

    resolution_cues = ['fix', 'fixed', 'fixing', 'resolve', 'resolved', 'resolving',
                        'solve', 'solved', 'solving', 'sorted', 'sorted out',
                        'worked perfectly', 'works perfectly', 'turned out fine',
                        'everything worked', 'all good now']
    positive_cues = ['great', 'wow', 'wonderful', 'fantastic', 'brilliant', 'perfect',
                      'love', 'thanks', 'thank you', 'amazing', 'best', 'nice', 'awesome',
                      'excellent', 'superb', 'outstanding']
    negative_context = ['broke', 'break', 'broken', 'crash', 'crashed', 'fail', 'failed',
                         'failure', 'declin', 'block', 'stuck', 'delay', 'wait',
                         'error', 'bug', 'glitch', 'down', 'outage', 'cancel',
                         'lost', 'lose', 'hold', 'refund', 'complain', 'annoy',
                         'frustrat', 'terrible', 'wrong']

    has_resolution_cue = any(word in text_lower for word in resolution_cues)
    has_positive_cue = any(word in text_lower for word in positive_cues)
    has_negated_negative = any(word in text_lower for word in negative_context) and not has_unnegated_negative_context(text_lower, negative_context)

    # Contrastive turnaround: a negative-context word followed later by "but"
    # and then a resolution cue - e.g. "ready to complain, but it worked perfectly"
    has_contrastive_resolution = False
    if ' but ' in text_lower:
        before_but, after_but = text_lower.split(' but ', 1)
        if any(w in before_but for w in negative_context) and any(w in after_but for w in resolution_cues):
            has_contrastive_resolution = True

    return (has_positive_cue and (has_resolution_cue or has_negated_negative)) or has_contrastive_resolution  

def analyze_feedback(text):
    """
    Takes raw feedback text, returns sentiment + intent + confidence scores.
    intent_label_map: a dict mapping intent label numbers to intent names
    """
    cleaned = clean_text(text)

    # Sentiment prediction with confidence
    sent_vec = sentiment_vectorizer.transform([cleaned])
    sentiment_pred = sentiment_model.predict(sent_vec)[0]
    sentiment_proba = sentiment_model.predict_proba(sent_vec)[0]
    sentiment_confidence = float(max(sentiment_proba))
    sentiment_label_raw = SENTIMENT_LABELS[sentiment_pred]
    sarcasm_flag = detect_sarcasm_cue(text)
    resolution_flag = detect_resolution_cue(text)

    if resolution_flag:
        sentiment_label_raw = 'positive'
        sentiment_confidence = 0.6
    elif sarcasm_flag and sentiment_label_raw != 'negative':
        sentiment_label_raw = 'negative'
        sentiment_confidence = 0.55

    # Intent prediction with confidence
    intent_vec = intent_vectorizer.transform([cleaned])
    intent_pred = intent_model.predict(intent_vec)[0]
    intent_proba = intent_model.predict_proba(intent_vec)[0]
    intent_confidence = float(max(intent_proba))

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "sentiment": sentiment_label_raw,
        "sarcasm_detected": sarcasm_flag,
        "sentiment_confidence": round(sentiment_confidence, 3),
        "intent": INTENT_LABEL_MAP.get(intent_pred, "unknown"),
        "intent_confidence": round(intent_confidence, 3)
    }