"""
Text normalizer for informal customer feedback (emojis, slang, typos, 'ok/k/kk', stretched words).

Where it sits: analyze_feedback() calls normalize_text(raw) FIRST. The result feeds the
cleaners, the three models and the rule layer. The ORIGINAL text is still what gets
stored and shown to users.

Design rules
  1. Standard English must pass through UNCHANGED (checked by test_normalize.py).
  2. Informal forms are mapped to words the models and heuristics.py already know
     (great, love, good, angry, bad, awful, disappointed ...), so negation, sarcasm
     and relief rules keep working on slang for free.
  3. Ambiguous items are decided by context, not by a blind dictionary
     (e.g. the crying-face emoji is dropped when the sentence already has a polar word).
  4. Everything is optional: if symspellpy is missing, spelling correction is skipped
     and the rest still works. This module never raises on normal input.
"""
import os
import re

NORMALIZER_VERSION = "1.0"

try:  # the rule layer's own polar-word check, so both stay consistent
    from heuristics import has_polar_words as _has_polar
except Exception:  # pragma: no cover
    try:
        from .heuristics import has_polar_words as _has_polar
    except Exception:
        _FALLBACK = re.compile(r"\b(good|great|love|happy|bad|awful|angry|hate|terrible|horrible|poor|worst|best)\b", re.I)
        _has_polar = lambda t: bool(_FALLBACK.search(t))

# ---------------------------------------------------------------------------
# Emojis
# ---------------------------------------------------------------------------
def _expand(spec):
    out = {}
    for chars, word in spec:
        for ch in chars:
            out[ch] = word
    return out

EMOJI_MAP = _expand([
    ("\U0001F600\U0001F603\U0001F604\U0001F601\U0001F606\U0001F60A\U0001F642\U0001F607\u263A\U0001F917", "happy"),
    ("\U0001F60D\U0001F970\U0001F618\U0001F63B\U0001F495\U0001F496\U0001F497\U0001F498\u2764\U0001F9E1\U0001F49B\U0001F49A\U0001F499\U0001F49C\u2665\U0001F60B", "love"),
    ("\U0001F44D\U0001F44C\U0001F64C\U0001F44F\U0001F4AA\U0001F91D\U0001F197\u2705\u2714", "good"),
    ("\U0001F389\U0001F973\u2728\U0001F31F\u2B50\U0001F4AF\U0001F3C6\U0001F525\U0001F60E\U0001F929", "great"),
    ("\U0001F64F", "thanks"),
    ("\U0001F621\U0001F620\U0001F92C\U0001F47F\U0001F4A2", "angry"),
    ("\U0001F624\U0001F629\U0001F62B", "frustrated"),
    ("\U0001F61E\U0001F614\U0001F61F\U0001F926\U0001F625", "disappointed"),
    ("\U0001F641\u2639\U0001F615\U0001F622\U0001F972\U0001F494", "sad"),
    ("\U0001F44E\u274C\U0001F6AB\u26D4", "bad"),
    ("\U0001F92E\U0001F922", "awful"),
    ("\U0001F612\U0001F644", "annoyed"),
    ("\U0001F610\U0001F611", "meh"),
    ("\U0001F631\U0001F628\U0001F630", "shocked"),
])
# Gen Z uses these in both directions, so context decides (see _resolve_ambiguous)
AMBIGUOUS_EMOJI = {"\U0001F62D": "sad", "\U0001F602": "funny", "\U0001F923": "funny",
                   "\U0001F480": "", "\U0001F643": "", "\U0001F605": ""}
_AMB_MARK = {ch: chr(0xE000 + i) for i, ch in enumerate(AMBIGUOUS_EMOJI)}
_MARK_BACK = {v: k for k, v in _AMB_MARK.items()}

_STRIP = {"\uFE0F", "\uFE0E", "\u200D", "\u20E3"} | {chr(c) for c in range(0x1F3FB, 0x1F400)}
_ANY_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2300-\u23FF\u2B00-\u2BFF\u2190-\u21FF\u3030\u303D]")

EMOTICONS = [
    (re.compile(r"(?<![\w:])(?::-?\)+|=\)+|\^_\^|:-?D+|<3|xD)(?![\w])"), " happy "),
    (re.compile(r"(?<![\w:])(?::-?\(+|:'\(|T_T|;_;|>:\()(?![\w])"), " sad "),
]

def _emoji_pass(text):
    out = []
    for ch in text:
        if ch in _STRIP:
            continue
        if ch in EMOJI_MAP:
            out.append(" " + EMOJI_MAP[ch] + " ")
        elif ch in AMBIGUOUS_EMOJI:
            out.append(" " + _AMB_MARK[ch] + " ")
        elif _ANY_EMOJI.match(ch):
            out.append(" ")  # unknown pictograph: carries no mapped meaning, drop it
        else:
            out.append(ch)
    return "".join(out)

# ---------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------
ACK = {"k", "kk", "kkk", "kay", "okk", "okkk", "okie", "okey", "oki", "oky", "okayy", "okayyy", "okay", "ok"}

# '' means drop the word (it adds intensity or tone, not polarity)
SLANG = {
    # abbreviations / texting
    "u": "you", "ur": "your", "r": "are", "pls": "please", "plz": "please", "plss": "please", "pleaseee": "please",
    "thx": "thanks", "thnx": "thanks", "thanx": "thanks", "tnx": "thanks", "ty": "thanks", "tq": "thanks",
    "gud": "good", "gr8": "great", "luv": "love", "luvv": "love", "b4": "before", "l8r": "later",
    "2day": "today", "tmrw": "tomorrow", "tmr": "tomorrow", "2moro": "tomorrow", "2morrow": "tomorrow",
    "cuz": "because", "coz": "because", "bcoz": "because", "bcz": "because", "bc": "because",
    "ppl": "people", "msg": "message", "msgs": "messages", "acct": "account", "pic": "picture", "pics": "pictures",
    "rn": "right now", "tho": "though", "thru": "through", "btw": "by the way", "nvm": "forget it",
    "gonna": "going to", "wanna": "want to", "gotta": "got to", "kinda": "kind of", "sorta": "sort of",
    "lemme": "let me", "gimme": "give me", "ain't": "is not", "aint": "is not",
    # hedges, fillers, tone (drop: the rest of the sentence carries the sentiment)
    "idk": "unsure", "idek": "unsure", "dunno": "unsure", "idc": "indifferent",
    "ngl": "honestly", "tbh": "honestly", "imo": "honestly", "imho": "honestly",
    "fr": "really", "frfr": "really", "deadass": "seriously", "highkey": "really", "hella": "very",
    "af": "extremely", "asf": "extremely", "lowkey": "",
    "bruh": "", "bro": "", "fam": "", "dude": "", "bruv": "", "yaar": "", "yar": "", "bhai": "",
    "omg": "", "oml": "", "omfg": "", "lol": "", "lmao": "", "lmfao": "", "rofl": "",
    "haha": "", "hahaha": "", "hehe": "", "xd": "", "ikr": "",
    # strong reactions
    "wtf": "outrageous", "wth": "outrageous", "ffs": "outrageous", "smh": "disappointed",
    # positive slang -> words the sentiment lexicon already has
    "lit": "great", "goated": "excellent", "slaps": "great", "bussin": "great", "dope": "great",
    "banger": "great", "slay": "great", "slayed": "great", "fye": "great", "w": "win",
    # negative slang
    "mid": "poor", "wack": "bad", "janky": "poor", "cringe": "awful", "ick": "awful", "yikes": "bad",
    "ghosted": "ignored", "ghosting": "ignoring", "salty": "angry", "shook": "shocked",
    "cooked": "broken", "bricked": "broken", "sus": "suspicious",
    "nah": "no",
}
# Apostrophe-less contractions: NEVER spell-correct these ('dont' must not become 'done').
# heuristics.py already treats them as negators.
PROTECTED = set("""dont cant wont didnt doesnt isnt wasnt arent werent couldnt shouldnt wouldnt havent hasnt hadnt
im ive youre thats whats lets theyre weve youve ill id hes shes wasn isn aren""".split())

# Very common typos where the highest-frequency neighbour is the WRONG word
TYPOS = {"realy": "really", "alot": "a lot", "definately": "definitely", "abt": "about", "thier": "their",
         "recieve": "receive", "recieved": "received", "wich": "which", "becuase": "because", "adress": "address",
         "shiping": "shipping", "refudn": "refund", "refnd": "refund", "oder": "order", "ordr": "order",
         "wher": "where", "whre": "where", "waht": "what", "delievery": "delivery", "deliverd": "delivered", "recived": "received", "shipement": "shipment"}

def _rule_vocab():
    """Every word that heuristics.py looks for. If the rules match it, the normalizer must not alter it."""
    words = set()
    try:
        try:
            import heuristics as h
        except Exception:
            from . import heuristics as h
        for name in dir(h):
            val = getattr(h, name)
            if name.isupper() and isinstance(val, (list, tuple, set, frozenset)):
                for item in val:
                    if isinstance(item, str):
                        words.update(re.findall(r"[a-z]+", item.lower().replace("re:", " ")))
    except Exception:
        pass
    return words

# words with these spellings are real words the spell-checker must not touch
DOMAIN_WORDS = (
    "refund refunds chargeback otp upi emi cod sku wifi bluetooth smartphone login signup logout username checkout "
    "unsubscribe cashback paytm gpay phonepe amazon flipkart swiggy zomato myntra ajio nykaa whatsapp instagram ios "
    "android iphone ipad usb hdmi app apps feedback dashboard website tracking courier gst invoice "
    "installment installments instalment instalments wishlist unboxing refurbished preorder preorders subscription "
    "subscriptions resubscribe reorder reordering repurchase packaging unpackaged restock restocked backorder "
    "dialing inbox emailed texted googled downloaded uninstall uninstalled reinstall reinstalled"
).split()
PROTECTED_ALL = None  # filled lazily: PROTECTED + DOMAIN_WORDS + rule vocabulary

PHRASES_CASE = [  # case-sensitive on purpose: 'L' / 'W' as words only when capitalised
    (re.compile(r"\b(a|an|the|big|huge|massive|total|major)\s+L\b"), r"\1 failure"),
    (re.compile(r"\b(a|the|big|huge|massive)\s+W\b"), r"\1 win great"),
]
PHRASES = [
    (re.compile(r"\bno cap\b", re.I), " honestly "),
    (re.compile(r"\b(?:that'?s|it'?s|thats|its)\s+cap\b", re.I), " that is false "),
    (re.compile(r"\bon god\b|\bong\b", re.I), " honestly "),
    (re.compile(r"\bhits different\b", re.I), " is great "),
    (re.compile(r"\bno notes\b", re.I), " perfect "),
    (re.compile(r"\bchef'?s kiss\b", re.I), " perfect "),
    (re.compile(r"\bthe goat\b", re.I), " excellent "),
    (re.compile(r"\b(is|are|was|so|straight|pure|total)\s+fire\b", re.I), r"\1 great"),
    (re.compile(r"\b(is|was|so|total|complete)\s+ass\b", re.I), r"\1 awful"),
    (re.compile(r"\bw/o\b", re.I), "without"),
    (re.compile(r"\bw/(?=\s|$)", re.I), "with"),
    (re.compile(r"\bfor real\b(?!\s*\?)", re.I), "really"),
]

def _rating_pass(text):
    def repl(m):
        a, b = float(m.group(1)), float(m.group(2))
        if b not in (5.0, 10.0) or a > b:
            return m.group(0)
        share = a / b
        return " excellent " if share >= 0.8 else (" terrible " if share <= 0.4 else m.group(0))
    return re.sub(r"(?<![\w/.])(\d{1,2}(?:\.\d)?)\s*/\s*(5|10)(?![\w/])", repl, text)

# ---------------------------------------------------------------------------
# Spell correction (optional)
# ---------------------------------------------------------------------------
_SYM = None
_SYM_TRIED = False

def _get_sym():
    global _SYM, _SYM_TRIED
    if _SYM_TRIED:
        return _SYM
    _SYM_TRIED = True
    try:
        import importlib.resources as ir
        from symspellpy import SymSpell
        if os.environ.get("NORMALIZE_SPELL", "1") != "1":
            return None
        import itertools, tempfile
        sym = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
        path = str(ir.files("symspellpy") / "frequency_dictionary_en_82_765.txt")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:  # file is sorted by frequency: keep the top N
            top = "".join(itertools.islice(fh, int(os.environ.get("NORMALIZE_DICT_SIZE", "60000"))))
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp.write(top)
        try:
            sym.load_dictionary(tmp.name, 0, 1)
        finally:
            os.remove(tmp.name)
        for w in DOMAIN_WORDS + list(SLANG) + list(ACK):
            sym.create_dictionary_entry(w, 10_000_000)
        _SYM = sym
    except Exception as exc:  # spelling correction is optional; say so instead of failing silently
        print(f"[normalize] spelling correction disabled: {exc!r}")
        _SYM = None
    return _SYM

_TINY_KNOWN = set("so good great bad love nice soon too cool very really yes no please thanks hello hi sad happy long late slow "
                  "fast worst best awesome amazing excellent terrible horrible awful poor fine okay ok well".split())

def _is_known(w):
    sym = _get_sym()
    return (w in sym.words) if sym else (w in _TINY_KNOWN)

def _unstretch(w):
    """'sooo' -> 'so', 'goood' -> 'good', 'loooove' -> 'love'. Only when a real word results."""
    if not re.search(r"(.)\1\1", w):
        return w
    two = re.sub(r"(.)\1{2,}", r"\1\1", w)
    one = re.sub(r"(.)\1{2,}", r"\1", w)
    for cand in (two, one):
        if _is_known(cand):
            return cand
    return w

def _correct(w):
    sym = _get_sym()
    if not sym or len(w) < 4 or _is_known(w):
        return w
    from symspellpy import Verbosity
    sug = sym.lookup(w, Verbosity.TOP, max_edit_distance=1, include_unknown=False)
    if sug and sug[0].distance == 1 and sug[0].term[0] == w[0]:
        return sug[0].term
    return w

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9']*")

def _protected():
    global PROTECTED_ALL
    if PROTECTED_ALL is None:
        PROTECTED_ALL = set(PROTECTED) | set(DOMAIN_WORDS) | _rule_vocab()
    return PROTECTED_ALL

def _token_pass(text):
    def repl(m):
        tok = m.group(0)
        low = tok.lower()
        # hyphenated compounds ('mid-month', 'well-made') are not slang: leave them alone
        if (m.start() > 0 and text[m.start() - 1] == "-") or text[m.end():m.end() + 1] == "-":
            return tok
        if low in _protected() and low not in ACK and low not in SLANG:
            return tok
        if low in ACK:
            return "ok"
        if low in SLANG:
            return SLANG[low]
        if "'" in tok or any(c.isdigit() for c in tok) or low in PROTECTED:
            return tok
        if low in TYPOS:
            return TYPOS[low]
        start = m.start()
        prev = text[:start].rstrip()
        sentence_start = (not prev) or prev[-1] in ".!?\n"
        if (tok[0].isupper() and not sentence_start) or (tok.isupper() and len(tok) > 1):
            return tok  # likely a name / acronym: leave alone
        fixed = _unstretch(low)
        if fixed != low:
            return SLANG.get(fixed, fixed)
        fixed = _correct(low)
        return fixed if fixed != low else tok
    return _WORD.sub(repl, text)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _resolve_ambiguous(text):
    marks = [c for c in text if c in _MARK_BACK]
    if not marks:
        return text
    bare = "".join(c for c in text if c not in _MARK_BACK)
    polar = _has_polar(bare)
    for mark in set(marks):
        word = "" if polar else AMBIGUOUS_EMOJI[_MARK_BACK[mark]]
        text = text.replace(mark, " " + word + " ")
    return text

def normalize_text(text, spell=True):
    """Return an informal-language-aware version of `text`. Standard English is unchanged."""
    if not text:
        return text
    t = str(text).replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = _rating_pass(t)
    for rx, rep in PHRASES_CASE:
        t = rx.sub(rep, t)
    for rx, rep in PHRASES:
        t = rx.sub(rep, t)
    for rx, rep in EMOTICONS:
        t = rx.sub(rep, t)
    t = _emoji_pass(t)
    t = _token_pass(t) if spell else _token_pass_nospell(t)
    t = _resolve_ambiguous(t)
    return re.sub(r"\s+", " ", t).strip()

def _token_pass_nospell(text):
    global _SYM, _SYM_TRIED
    saved = (_SYM, _SYM_TRIED)
    _SYM, _SYM_TRIED = None, True
    try:
        return _token_pass(text)
    finally:
        _SYM, _SYM_TRIED = saved

if os.environ.get("NORMALIZE_PRELOAD", "1") == "1":
    _get_sym()  # load the dictionary once at start-up instead of on the first request
