from transformers import pipeline

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Sample dataset
texts = [
    "Trump's tariffs are ruining our economy.",
    "Finally, someone is standing up to China. Good job Trump!",
    "The market just tanked after Trump's announcement.",
    "Tariffs might protect US jobs in the long term.",
]

# Analyze
results = sentiment_pipeline(texts)

# Print results
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Label: {result['label']}, Confidence: {round(result['score'], 2)}\n")