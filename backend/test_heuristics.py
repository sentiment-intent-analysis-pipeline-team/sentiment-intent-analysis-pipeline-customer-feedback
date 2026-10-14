"""
Unit tests for the rule layer (heuristics.py).

Run from the backend folder:
    python test_heuristics.py
or, if pytest is installed:
    pytest test_heuristics.py

No model files are needed - these test only the rule logic.
"""

from heuristics import (
    detect_sarcasm_cue, detect_resolution_cue, detect_negated_positive,
    detect_wrong_item_cue, detect_malfunction_cue, detect_negative_event_cue,
    detect_damage_cue, has_polar_words, resolve_labels,
)

DETECTORS = {
    'sarcasm': detect_sarcasm_cue,
    'resolution': detect_resolution_cue,
    'negated_positive': detect_negated_positive,
    'wrong_item': detect_wrong_item_cue,
    'malfunction': detect_malfunction_cue,
    'negative_event': detect_negative_event_cue,
    'damage': detect_damage_cue,
}

# (sentence, flags that MUST be true, flags that MUST be false)
FLAG_CASES = [
    # --- negation: the base model never sees "not" ---
    ("the quality of cloth from this website is good.", [], ['negated_positive']),
    ("the quality of cloth from this website is not good.", ['negated_positive'], []),
    ("the quality of cloth from this website is not good. other websites are better.", ['negated_positive'], []),
    ("the material was so stiff and was not same as of that in the picture", ['negated_positive'], []),
    ("I can't talk right now", [], ['negated_positive']),

    # --- sarcasm vs relief ---
    ("oh wonderful, the app crashed again right before my presentation", ['sarcasm'], ['resolution']),
    ("oh wonderful, the app has not crashed again right before my presentation", ['resolution'], []),
    ("Great, another update that broke the login page.", ['sarcasm'], ['resolution']),
    ("Oh wonderful, the app has crashed again.", ['sarcasm'], ['resolution']),
    ("Nice job losing my order, really impressive.", ['sarcasm'], ['resolution']),
    ("Thanks a lot for cancelling my booking at the last minute.", ['sarcasm'], ['resolution']),
    ("my card was declined, but wow, thanks for nothing", ['sarcasm'], []),
    ("The app has not crashed once since the update, great work.", ['resolution'], []),
    ("amazing, the refund actually wasn't declined this time", ['resolution'], ['sarcasm']),
    ("wow, my account was not suspended after all, what a relief", ['resolution'], ['sarcasm']),
    ("oh wonderful, nothing broke during setup this time", ['resolution'], ['sarcasm']),
    ("oh great, the delivery wasn't delayed for once", ['resolution'], ['sarcasm']),
    ("thank goodness, the payment didn't fail this time", ['resolution'], ['sarcasm']),
    ("the payment didn't fail this time", ['resolution'], []),
    ("I was ready to write a complaint, but then everything worked perfectly.", ['resolution'], ['sarcasm']),
    ("Thanks for fixing that bug so quickly.", ['resolution'], []),
    ("Everything worked perfectly on the first try.", ['resolution'], []),
    ("nothing was damaged, great packaging", ['resolution'], ['damage']),
    ("Please fix the login bug, thanks", [], ['resolution']),
    ("The refund came through the same day, thank you.", [], ['sarcasm']),
    ("love how fast the download is, great app", [], ['sarcasm']),
    ("Great breakfast menu in the app", [], ['sarcasm']),

    # --- mixed feedback is not sarcasm ---
    ("the delivery was a bit delayed but the shirt fits perfectly and looks great", [], ['sarcasm', 'resolution']),

    # --- wrong item ---
    ("i ordered shirt, but they gave me skirt", ['wrong_item'], []),
    ("instead of male clothes, i got female clothes", ['wrong_item'], []),
    ("instead of the black shoes i ordered, they sent brown ones", ['wrong_item'], []),
    ("i ordered a jacket but instead got a pair of gloves", ['wrong_item'], []),
    ("i asked for a red shirt but they sent me a blue one", ['wrong_item'], []),
    ("the size i ordered was medium, but i received small", ['wrong_item'], []),
    ("this is not what i ordered", ['wrong_item'], []),
    ("they sent me the wrong item entirely", ['wrong_item'], []),
    ("wrong color received, very disappointed", ['wrong_item', 'negative_event'], []),
    ("Can I get store credit instead of a refund?", [], ['wrong_item']),
    ("offer discounts instead of raising prices", [], ['wrong_item']),

    # --- malfunction ---
    ("the watch is dialing backwards.", ['malfunction'], []),
    ("the watch is dialing backwards. i bought it from alibaba", ['malfunction'], []),
    ("my phone is not working properly", ['malfunction'], []),
    ("I can't believe it's not working", ['malfunction'], []),
    ("The charge was reversed", [], ['malfunction']),

    # --- plain bad-news facts ---
    ("I have been charged twice for the same purchase.", ['negative_event'], []),
    ("My order arrived two weeks late and nobody bothered to tell me.", ['negative_event'], []),
    ("my package never arrived and support has gone silent", ['negative_event'], []),
    ("Why is my refund taking so long?", [], ['negative_event', 'sarcasm', 'resolution']),
    ("How long does a refund usually take?", [], ['negative_event', 'sarcasm']),
    ("Is there a fee for cancelling my subscription?", [], ['negative_event', 'sarcasm', 'resolution']),

    # --- damage ---
    ("the box arrived crushed and the item inside was scratched", ['damage'], []),
    ("the screen arrived cracked", ['damage'], []),
    ("Packaging was damaged and half the items were missing.", ['damage'], []),
    ("the item arrived without any scratches", [], ['damage']),
    ("you guys crushed it, thank you", [], ['damage']),
]

POLAR_CASES = [
    ("could you add a way to filter by price range?", False),
    ("Do you ship internationally?", False),
    ("Is customer support available on weekends?", False),
    ("How do I reset my password?", False),
    ("Why does your app always crash when I try to log in?", True),
    ("It would be great if you added a dark mode.", True),
    ("Terrible experience, I want my money back.", True),
]

# Replay of REAL model outputs from the test sessions:
# (sentence, raw sentiment, raw sentiment conf, raw type, raw type conf,
#  expected final sentiment, expected final type)
# Raw values are what the TF-IDF models returned before any rule was applied
# (taken from the logged results; where a rule had already overridden the
# logged value, a plausible raw value is used - the rule outcome does not
# depend on it).
RESOLVE_CASES = [
    # the 11 results that showed errors
    ("oh wonderful, the app has not crashed again right before my presentation", 'negative', 0.50, 'complaint', 0.58, 'positive', 'praise'),
    ("the box arrived crushed and the item inside was scratched", 'neutral', 0.627, 'praise', 0.626, 'negative', 'complaint'),
    ("could you add a way to filter by price range?", 'negative', 0.70, 'suggestion', 0.469, 'neutral', 'suggestion'),
    ("how do I enable two-factor authentication?", 'neutral', 0.53, 'query', 0.909, 'neutral', 'query'),
    ("my package never arrived and support has gone silent", 'negative', 0.74, 'complaint', 0.446, 'negative', 'complaint'),
    ("the staff were rude and unhelpful, worst visit ever", 'negative', 0.834, 'complaint', 0.433, 'negative', 'complaint'),
    ("amazing, the refund actually wasn't declined this time", 'negative', 0.492, 'praise', 0.397, 'positive', 'praise'),
    ("wow, my account was not suspended after all, what a relief", 'negative', 0.705, 'complaint', 0.57, 'positive', 'praise'),
    ("oh wonderful, nothing broke during setup this time", 'negative', 0.50, 'complaint', 0.493, 'positive', 'praise'),
    ("oh great, the delivery wasn't delayed for once", 'positive', 0.60, 'praise', 0.34, 'positive', 'praise'),
    ("thank goodness, the payment didn't fail this time", 'negative', 0.504, 'complaint', 0.397, 'positive', 'praise'),
    # earlier rounds - must not regress
    ("Do you ship internationally?", 'negative', 0.475, 'query', 0.816, 'neutral', 'query'),
    ("Is customer support available on weekends?", 'positive', 0.528, 'query', 0.762, 'neutral', 'query'),
    ("Is there a way to change my username?", 'neutral', 0.507, 'query', 0.797, 'neutral', 'query'),
    ("Why does your app always crash when I try to log in?", 'negative', 0.555, 'query', 0.44, 'negative', 'query'),
    ("I have been charged twice for the same purchase.", 'neutral', 0.431, 'complaint', 0.712, 'negative', 'complaint'),
    ("the material was so stiff and was not same as of that in the picture", 'neutral', 0.895, 'complaint', 0.735, 'negative', 'complaint'),
    ("tv is well packed and warranty card was there. early deliver.", 'positive', 0.474, 'praise', 0.741, 'positive', 'praise'),
    ("instead of male clothes, i got female clothes", 'negative', 0.557, 'complaint', 0.745, 'negative', 'complaint'),
    ("the delivery was a bit delayed but the shirt fits perfectly and looks great", 'positive', 0.863, 'praise', 0.525, 'positive', 'praise'),
    ("the watch is dialing backwards.", 'neutral', 0.731, 'complaint', 0.461, 'negative', 'complaint'),
    ("I suggest sending a confirmation email after every order.", 'neutral', 0.493, 'suggestion', 0.709, 'neutral', 'suggestion'),
    ("It would be great if you added a dark mode.", 'positive', 0.894, 'suggestion', 0.847, 'positive', 'suggestion'),
    ("Absolutely love the new design of the app.", 'positive', 0.938, 'praise', 0.72, 'positive', 'praise'),
    ("The app crashes every time I try to open my account page.", 'negative', 0.762, 'complaint', 0.75, 'negative', 'complaint'),
    ("Everything worked perfectly on the first try.", 'positive', 0.60, 'praise', 0.737, 'positive', 'praise'),
    ("Best customer service I have experienced in years.", 'positive', 0.694, 'praise', 0.703, 'positive', 'praise'),
    ("My order arrived two weeks late and nobody bothered to tell me.", 'negative', 0.843, 'complaint', 0.729, 'negative', 'complaint'),
    ("What is the status of my order?", 'neutral', 0.378, 'query', 0.792, 'neutral', 'query'),
    ("How do I reset my password?", 'neutral', 0.519, 'query', 0.828, 'neutral', 'query'),
    ("You should consider offering free returns.", 'neutral', 0.617, 'suggestion', 0.752, 'neutral', 'suggestion'),
    ("oh wonderful, the app crashed again right before my presentation", 'negative', 0.484, 'complaint', 0.615, 'negative', 'complaint'),
    ("Can I return an item without the original packaging?", 'neutral', 0.603, 'query', 0.778, 'neutral', 'query'),
    ("What is the maximum amount I can withdraw per day?", 'neutral', 0.54, 'query', 0.827, 'neutral', 'query'),
    # unsure type model + concrete bad-news fact -> complaint
    ("the watch is dialing backwards.", 'neutral', 0.731, 'query', 0.39, 'negative', 'complaint'),
    # hypothetical questions that merely mention a bad word are still questions
    ("are returns free for defective items", 'negative', 0.60, 'query', 0.70, 'neutral', 'query'),
    ("can i return it if it arrives damaged", 'negative', 0.60, 'query', 0.70, 'neutral', 'query'),
    ("What happens if my parcel is lost in transit?", 'negative', 0.65, 'query', 0.70, 'neutral', 'query'),
    # ...but a real report of the same problem is still a complaint
    ("my parcel arrived damaged and nobody will help", 'neutral', 0.55, 'complaint', 0.70, 'negative', 'complaint'),
    ("the item I received is defective and the seller ignores me", 'neutral', 0.55, 'complaint', 0.70, 'negative', 'complaint'),
]


def run_all():
    failures = []

    for sentence, must_true, must_false in FLAG_CASES:
        for name in must_true:
            if not DETECTORS[name](sentence):
                failures.append(f"FLAG  {name} should be TRUE : {sentence}")
        for name in must_false:
            if DETECTORS[name](sentence):
                failures.append(f"FLAG  {name} should be FALSE: {sentence}")

    for sentence, expected in POLAR_CASES:
        if has_polar_words(sentence) != expected:
            failures.append(f"POLAR expected {expected}: {sentence}")

    for sentence, s_lab, s_conf, t_lab, t_conf, exp_s, exp_t in RESOLVE_CASES:
        out = resolve_labels(sentence, s_lab, s_conf, t_lab, t_conf)
        if (out['sentiment'], out['feedback_type']) != (exp_s, exp_t):
            failures.append(
                f"LABEL expected ({exp_s}, {exp_t}) got "
                f"({out['sentiment']}, {out['feedback_type']}) : {sentence}")

    total = (sum(len(t) + len(f) for _, t, f in FLAG_CASES)
             + len(POLAR_CASES) + len(RESOLVE_CASES))
    return total, failures


def test_all_rules():
    total, failures = run_all()
    assert not failures, "\n" + "\n".join(failures)


if __name__ == '__main__':
    total, failures = run_all()
    if failures:
        print(f"{len(failures)} of {total} checks FAILED:\n")
        for f in failures:
            print("  " + f)
        raise SystemExit(1)
    print(f"All {total} checks passed.")