from pipeline import analyze_feedback

test_texts = [
    "My card hasn't arrived yet, it's been two weeks.",
    "Terrible service, I want a refund immediately.",
    "Can you tell me your working hours?"
]

for text in test_texts:
    result = analyze_feedback(text)
    print(result)
    print()