from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load models once for efficiency
model = SentenceTransformer("all-MiniLM-L6-v2")
analyzer = SentimentIntensityAnalyzer()

def get_embeddings(texts):
    return model.encode(texts)

def get_sentiments(texts):
    return [analyzer.polarity_scores(text)['compound'] for text in texts]