"""
Rule-based layer that sits on top of the TF-IDF sentiment and feedback-type
models. Everything in this file is plain Python (no model files, no spaCy),
so it can be unit tested on its own - see test_heuristics.py.

Why this layer exists: the base models are word-frequency models. They cannot
see negation ('not good' is cleaned to 'good'), sentence structure ('ordered X
but got Y'), or plain bad-news facts that contain no emotional words ('the
watch is dialing backwards'). Each detector below targets one of those gaps.

Term-list convention used throughout:
    'word'      matches the whole word only        ('down' does not match 'download')
    'stem*'     matches any word starting with it  ('crash*' matches 'crashed')
    're:<rx>'   a raw regular expression, used as-is
"""

import re
from functools import lru_cache


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _prep(text):
    """Lowercase and normalise curly apostrophes so "wasn't" always matches."""
    return str(text).lower().replace("\u2019", "'").replace("\u2018", "'")


@lru_cache(maxsize=None)
def _pattern(term):
    if term.startswith('re:'):
        return re.compile(term[3:])
    if term.endswith('*'):
        return re.compile(r'\b' + re.escape(term[:-1]))
    return re.compile(r'\b' + re.escape(term) + r'\b')


_NEGATORS = {
    'not', 'no', 'never', 'nothing', 'nobody', 'without', 'neither', 'nor',
    'hardly', 'barely', 'cannot', 'cant', 'wont', 'dont', 'didnt', 'doesnt',
    'isnt', 'wasnt', 'arent', 'werent', 'hasnt', 'havent', 'hadnt',
    'couldnt', 'wouldnt', 'shouldnt',
}


def _is_negated(text, idx):
    """True if one of the 3 words just before position idx (within the same
    clause) is a negator. 'has not crashed', "wasn't declined", 'nothing
    broke' are all negated; 'I have no idea why it crashed' is not, because
    the 'no' is more than 3 words away from 'crashed'."""
    before = text[:idx]
    clause = re.split(r"[,.;:!?]|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\byet\b", before)[-1]
    tokens = re.findall(r"[a-z']+", clause)[-3:]
    return any(t in _NEGATORS or t.endswith("n't") for t in tokens)


def has_any(text, terms):
    """Any term present, negated or not."""
    return any(_pattern(t).search(text) for t in terms)


def has_unnegated(text, terms):
    """Any term present in at least one non-negated occurrence."""
    for t in terms:
        for m in _pattern(t).finditer(text):
            if not _is_negated(text, m.start()):
                return True
    return False


def only_negated(text, terms):
    """Terms are present, but every occurrence is negated ('has not crashed')."""
    return has_any(text, terms) and not has_unnegated(text, terms)


def has_unnegated_negative_context(text_lower, negative_context):
    """Backward-compatible wrapper for the old function of the same name."""
    terms = [t if (t.endswith('*') or t.startswith('re:')) else t + '*'
             for t in negative_context]
    return has_unnegated(_prep(text_lower), terms)


# ---------------------------------------------------------------------------
# Term lists
# ---------------------------------------------------------------------------

POSITIVE_CUES = [
    'great', 'wow', 'wonderful', 'fantastic', 'brilliant', 'perfect*', 'love*',
    'thank*', 'amazing', 'best', 'nice', 'awesome', 'excellent', 'superb',
    'outstanding', 'phew', 'what a relief',
]

# Only counted as a positive signal when a bad event is also negated:
# "the payment didn't fail this time" is relief, "this time it failed" is not.
RELIEF_CUES = ['for once', 'this time', 'after all', 'finally']

# 'fix' on its own is excluded on purpose: "please fix the login bug, thanks"
# is a request, not a resolution.
RESOLUTION_CUES = [
    'fixed', 'fixing', 'the fix', 'resolved', 'resolving', 'solved', 'solving',
    'sorted', 'sorted out', 'worked perfectly', 'works perfectly',
    'turned out fine', 'everything worked', 'all good now',
]

# 'refund' is deliberately NOT here: it appears constantly in neutral and
# positive messages ("the refund came through, thank you").
BAD_EVENT_TERMS = [
    'broke', 'broken', 'break', 'breaks', 'breaking',
    'crash*', 'fail*', 'declin*', 'block*', 'stuck', 'delay*',
    'wait', 'waited', 'waiting',
    'charged twice', 'double charg*', 'overcharg*',
    'error*', 'bug*', 'glitch*', 'outage*', 'down',
    'cancel', 'cancels', 'cancelled', 'canceled', 'cancelling', 'canceling',
    'lost', 'lose', 'loses', 'losing', 'hold', 'complain*', 'annoy*',
    'frustrat*', 'terrible', 'wrong',
    'log me out', 'logged out', 'log out', 'kicked out', 'timed out',
    'suspend*', 'reject*', 'denied', 'deny', 'late', 'misplaced',
    'damage*', 're:\\bcrushed\\b(?! it)', 'scratched', 'dented', 'cracked',
    'shattered', 'torn', 'ripped', 'smashed', 'leaking', 'leaked', 'stained',
    'chipped', 'rude',
]

NEGATED_POSITIVE_WORDS = [
    'good', 'great', 'nice', 'fine', 'amazing', 'awesome', 'excellent',
    'perfect*', 'wonderful', 'best', 'satisfied', 'happy', 'same',
    'as expected', 'as described', 'as shown', 'as advertised', 'correct',
    're:\\bright\\b(?! now| away| back| after| before)',
    'worth it', 'worth the price', 'helpful', 'useful', 'easy', 'comfortable',
    'satisfactory', 'acceptable',
]

WRONG_ITEM_PHRASES = [
    're:\\binstead of\\b.{0,80}\\b(i got|i received|they sent|they gave|sent me|gave me|delivered|received|got)\\b',
    're:\\bbut instead\\b',
    'but they gave', 'but i got', 'but i received', 'but got', 'but received',
    'but sent', 'but they sent',
    'wrong item', 'wrong size', 'wrong color', 'wrong colour', 'wrong product',
    'wrong model', 'wrong order', 'different item', 'not what i ordered',
    'not what i asked for', 'not the one i ordered', 'someone else\'s order',
    'sent me the wrong', 'gave me the wrong', 'received the wrong',
]

# Phrases that are already negative in themselves ('not working'), so a
# negation check in front of them would be wrong ("I can't believe it's not
# working" must still count).
MALFUNCTION_SELF_NEGATIVE = [
    'not working', "doesn't work", 'does not work', "won't turn on",
    'wont turn on', 'not functioning', 'not charging', "won't charge",
]
MALFUNCTION_PHRASES = [
    'dialing backwards', 'going backwards', 'running backwards', 'upside down',
    'stopped working', 'stopped functioning', 'malfunction*', 'defective',
    'faulty',
]

NEGATIVE_EVENT_SELF_NEGATIVE = [
    'never arrived', 'never received', 'not delivered', 'not received',
    "hasn't arrived", 'has not arrived', "haven't received", 'have not received',
    'nobody bothered', 'nobody responded', 'nobody replied', 'no response',
    'no reply', 'gone silent', 'went silent',
]
NEGATIVE_EVENT_PHRASES = [
    'charged twice', 'charged me twice', 'double charged', 'double charge',
    'overcharged', 'payment failed', 'card was blocked', 'card got blocked',
    'account was suspended', 'account suspended', 'went missing',
    'items were missing', 'items missing', 'extremely disappointed',
    'very disappointed', 'worst customer service', 'worst experience',
    'unacceptable service', 'rude staff', 'rude on the phone',
    'ignored my emails', 'ignored my messages',
]

DAMAGE_TERMS = [
    'damaged', 'damage', 're:\\bcrushed\\b(?! it)', 'scratched', 'scratches',
    'dented', 'cracked', 'shattered', 'torn', 'ripped', 'smashed', 'leaking',
    'leaked', 'stained', 'chipped', 'broken', 'arrived bent', 'came bent',
]

# Words that carry sentiment of their own. If NONE of these appear in a plain
# question or suggestion, the base model's positive/negative guess has no
# lexical basis and is treated as neutral.
POLAR_EXTRA = [
    'good', 'bad', 'poor', 'awful', 'horrible', 'worst', 'hate*', 'disappoint*',
    'useless', 'angry', 'upset', 'ridiculous', 'pathetic', 'unacceptable',
    'happy', 'unhappy', 'satisfied', 'unsatisfied', 'pleased', 'helpful',
    'unhelpful',
]


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def detect_sarcasm_cue(text):
    """Positive-sounding word + a real (non-negated) bad event.

    Exception: 'delivery was delayed but the shirt fits perfectly and looks
    great' is honest mixed feedback, not sarcasm. Sarcasm in that shape almost
    always puts an interjection ('wow', 'thanks') right after the 'but'
    ('declined, but wow, thanks for nothing'), so that is what separates them."""
    t = _prep(text)

    if ' but ' in t:
        before_but, after_but = t.split(' but ', 1)
        if (has_any(before_but, BAD_EVENT_TERMS) and has_any(after_but, POSITIVE_CUES)
                and not has_any(after_but, ['wow', 'thank*'])):
            return False

    return has_any(t, POSITIVE_CUES) and has_unnegated(t, BAD_EVENT_TERMS)


def detect_resolution_cue(text):
    """Looks bad on the surface but is actually relief or a fix:
    'has not crashed again', 'wasn't declined this time', 'was ready to
    complain, but everything worked perfectly'."""
    t = _prep(text)

    negated_bad = only_negated(t, BAD_EVENT_TERMS)
    has_positive = has_any(t, POSITIVE_CUES)
    has_relief = has_any(t, RELIEF_CUES)
    has_resolution = has_any(t, RESOLUTION_CUES)

    core = (has_positive or (has_relief and negated_bad)) and (has_resolution or negated_bad)

    contrastive = False
    if ' but ' in t:
        before_but, after_but = t.split(' but ', 1)
        contrastive = has_any(before_but, BAD_EVENT_TERMS) and has_any(after_but, RESOLUTION_CUES)

    return core or contrastive


def detect_negated_positive(text):
    """'not good', "wasn't the same as the picture", "isn't worth it".

    clean_text() strips stopwords with NLTK's list, which contains 'not', so
    the base sentiment model never sees negation. This checks the RAW text."""
    t = _prep(text)
    for term in NEGATED_POSITIVE_WORDS:
        for m in _pattern(term).finditer(t):
            if _is_negated(t, m.start()):
                return True
    return False


def detect_wrong_item_cue(text):
    """'ordered a shirt, but they gave me a skirt' / 'instead of X, i got Y'.
    No negative word anywhere in these - the sentiment is in the structure."""
    return has_any(_prep(text), WRONG_ITEM_PHRASES)


def detect_malfunction_cue(text):
    """A malfunction described as a plain fact: 'the watch is dialing backwards.'"""
    t = _prep(text)
    return has_any(t, MALFUNCTION_SELF_NEGATIVE) or has_unnegated(t, MALFUNCTION_PHRASES)


def detect_negative_event_cue(text):
    """Bad news stated as a fact: 'I have been charged twice', 'my package
    never arrived'. The list is deliberately narrow - words like 'refund',
    'cancel' and 'wait' also appear in perfectly neutral questions."""
    t = _prep(text)
    return has_any(t, NEGATIVE_EVENT_SELF_NEGATIVE) or has_unnegated(t, NEGATIVE_EVENT_PHRASES)


def detect_damage_cue(text):
    """'the box arrived crushed and the item inside was scratched'. Negation
    aware: 'arrived without any scratches' and 'nothing was damaged' do not count."""
    return has_unnegated(_prep(text), DAMAGE_TERMS)


def has_polar_words(text):
    """True if the text contains any word that carries sentiment by itself."""
    t = _prep(text)
    return (has_any(t, POSITIVE_CUES) or has_any(t, BAD_EVENT_TERMS)
            or has_any(t, POLAR_EXTRA))


# ---------------------------------------------------------------------------
# Combining the base model outputs with the rules
# ---------------------------------------------------------------------------

STRUCTURAL_NEUTRAL_TYPE_CONFIDENCE = 0.4

# A question that opens like this is asking about a hypothetical, not reporting
# a problem: 'can i return it if it arrives damaged?', 'are returns free for
# defective items?'. ('why does your app always crash?' opens with 'why', so it
# is NOT treated as hypothetical and stays negative.)
HYPOTHETICAL_START = re.compile(
    r"^\s*(are|is|do|does|did|can|could|will|would|should|what happens|what if|if)\b")


def resolve_labels(text, sentiment_label, sentiment_conf, type_label, type_conf):
    """Take the raw sentiment + type predictions from the two models and apply
    the rule layer. Returns the final labels plus which rules fired.

    Priority for the sentiment label (first match wins):
        resolution > sarcasm > negated positive > wrong item >
        malfunction > negative event > damage
    """
    flags = {
        'resolution': detect_resolution_cue(text),
        'sarcasm': detect_sarcasm_cue(text),
        'negated_positive': detect_negated_positive(text),
        'wrong_item': detect_wrong_item_cue(text),
        'malfunction': detect_malfunction_cue(text),
        'negative_event': detect_negative_event_cue(text),
        'damage': detect_damage_cue(text),
    }
    hypothetical_query = (type_label == 'query'
                          and type_conf >= STRUCTURAL_NEUTRAL_TYPE_CONFIDENCE
                          and bool(HYPOTHETICAL_START.match(_prep(text))))
    if hypothetical_query:
        for k in ('negated_positive', 'wrong_item', 'malfunction',
                  'negative_event', 'damage'):
            flags[k] = False

    negative_evidence = any(flags[k] for k in (
        'sarcasm', 'negated_positive', 'wrong_item', 'malfunction',
        'negative_event', 'damage'))

    # 1. Rules that force the sentiment label.
    if flags['resolution']:
        sentiment_label, sentiment_conf = 'positive', 0.6
    elif flags['sarcasm'] and sentiment_label != 'negative':
        sentiment_label, sentiment_conf = 'negative', 0.55
    elif negative_evidence and sentiment_label != 'negative':
        sentiment_label, sentiment_conf = 'negative', 0.6

    # 2. A plain question or suggestion with no sentiment words of its own has
    #    no polarity. Any positive/negative from the base model is a guess.
    structural_neutral = False
    if (type_label in ('query', 'suggestion')
            and type_conf >= STRUCTURAL_NEUTRAL_TYPE_CONFIDENCE
            and not flags['resolution'] and not negative_evidence
            and (hypothetical_query or not has_polar_words(text))
            and sentiment_label != 'neutral'):
        sentiment_label, sentiment_conf = 'neutral', 0.5
        structural_neutral = True

    # 3. Keep the type consistent with the evidence behind the sentiment.
    #    A complaint with resolution evidence is really praise; praise with
    #    bad-news evidence is really a complaint.
    #    If the type model is also unsure (under 0.5) and a rule found a
    #    concrete bad-news fact, the rule wins over the shaky query/suggestion
    #    guess ('the watch is dialing backwards' -> complaint).
    plain_fact = (flags['wrong_item'] or flags['malfunction']
                  or flags['negative_event'] or flags['damage'])
    type_fix = None
    if flags['resolution'] and type_label == 'complaint':
        type_label, type_fix = 'praise', 'complaint_to_praise'
    elif negative_evidence and not flags['resolution'] and type_label == 'praise':
        type_label, type_fix = 'complaint', 'praise_to_complaint'
    elif (plain_fact and not flags['resolution']
          and type_label in ('query', 'suggestion') and type_conf < 0.5):
        type_label, type_fix = 'complaint', 'unsure_to_complaint'

    return {
        'sentiment': sentiment_label,
        'sentiment_confidence': sentiment_conf,
        'feedback_type': type_label,
        'flags': flags,
        'structural_neutral': structural_neutral,
        'hypothetical_query': hypothetical_query,
        'type_fix': type_fix,
    }