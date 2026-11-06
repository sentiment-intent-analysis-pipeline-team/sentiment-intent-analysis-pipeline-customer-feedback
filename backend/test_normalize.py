"""Unit tests for normalize.py. Run from the backend folder:  python test_normalize.py"""
from normalize import normalize_text as nt

# (input, expected output)
CASES = [
    # emojis become words the models and rules know
    ("The product is good \U0001F60A", "The product is good happy"),
    ("\U0001F621\U0001F621\U0001F621", "angry angry angry"),
    ("\U0001F44D", "good"),
    ("\U0001F44E", "bad"),
    # ambiguous emoji: dropped when the sentence already has a polar word, used otherwise
    ("I love this \U0001F62D", "I love this"),
    ("the delivery is late \U0001F62D", "the delivery is late"),
    ("my parcel \U0001F62D", "my parcel sad"),
    # ok / k / kk family
    ("ok", "ok"), ("k", "ok"), ("kk", "ok"), ("okkk", "ok"), ("okay", "ok"),
    # stretched words
    ("The product is sooo good", "The product is so good"),
    ("loooove it", "love it"), ("greeeat service", "great service"),
    # abbreviations and slang
    ("can u pls refund me", "can you please refund me"),
    ("ur app is gr8", "your app is great"),
    ("this is lit", "this is great"), ("that's mid", "that's poor"),
    ("bruh the app crashed again", "the app crashed again"),
    ("no cap best purchase ever", "honestly best purchase ever"),
    ("the new update is fire", "the new update is great"),
    ("total L from the delivery guys", "total failure from the delivery guys"),
    ("10/10 would buy again", "excellent would buy again"),
    ("1/10 never again", "terrible never again"),
    # spelling
    ("the delievery was very late", "the delivery was very late"),
    ("awsome product, realy happy", "awesome product, really happy"),
    ("wher is my refund", "where is my refund"),
    # must NOT be changed
    ("dont buy this", "dont buy this"),            # negation must survive spell-check
    ("it's not working", "it's not working"),
    ("mid-month sale", "mid-month sale"),
    ("Flipkart delivered it on time", "Flipkart delivered it on time"),
    ("my IKEA table arrived", "my IKEA table arrived"),
    ("the watch is dialing backwards", "the watch is dialing backwards"),
    ("5/10 would not recommend", "5/10 would not recommend"),
    ("", ""),
]

def run():
    bad = [(i, nt(i), e) for i, e in CASES if nt(i) != e]
    return len(CASES), bad

if __name__ == "__main__":
    n, bad = run()
    if bad:
        print(f"{len(bad)} of {n} normalizer checks FAILED:")
        for i, got, exp in bad: print(f"  input {i!r}\n    got      {got!r}\n    expected {exp!r}")
        raise SystemExit(1)
    print(f"All {n} normalizer checks passed.")
