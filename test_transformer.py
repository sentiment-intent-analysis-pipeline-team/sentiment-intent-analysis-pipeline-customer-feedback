from transformers import pipeline

print("Loading model...")
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

test_sentences = [
    "Oh great, another Monday. Just what I needed.",
    "Wow, thanks for spilling coffee on my laptop right before my exam. Really appreciate it.",
    "Sure, because waiting three hours in line is exactly how I wanted to spend my day.",
    "Fantastic, the WiFi went down again during my online interview. Perfect timing as always.",
    "Love how my flight got delayed by six hours. Best trip ever.",
]

for sentence in test_sentences:
    result = sentiment_pipeline(sentence)[0]
    print(f"{sentence}\n  -> {result['label']} ({result['score']:.2%})\n")