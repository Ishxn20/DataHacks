import os
import time
import numpy as np
import random
import torch
import tweepy
import pandas as pd
import plotly.express as px
import json
import re
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import umap
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from tweepy.errors import TooManyRequests
import glob
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Setup ─────────────────────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load the most recent tweet data file
json_files = "tweet_data_2025-04-05_21-56-52.json"
if not json_files:
    print("No cached tweet data files found.")
    exit()

with open(json_files, "r") as f:
    print(f"📁 Using cached file: {json_files[0]}")
    cached_data = json.load(f)

texts = [item["Original_Tweet"] for item in cached_data]

# ─── Model Setup ───────────────────────────────────────────────────────────────
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# ─── Smart Summarization ───────────────────────────────────────────────────────
def summarize_tweet(text):
    try:
        sentences = [s.strip() for s in re.split(r'[.\n]', text) if s.strip()]
        if len(sentences) == 0:
            return text.strip()
        if len(sentences) == 1:
            return sentences[0]
        embeddings = model.encode(sentences)
        center = util.pytorch_cos_sim(embeddings, embeddings).sum(dim=1)
        return sentences[center.argmax().item()].strip()
    except Exception as e:
        print(f"⚠️ Error summarizing: {e}")
        return text.strip()

summaries = [
    item["Summary"].strip() if item.get("Summary") and item["Summary"].strip()
    else summarize_tweet(item["Original_Tweet"])
    for item in cached_data
]
# summaries = [summarize_tweet(t) for t in texts]  # Skip re-summarizing since summaries exist
print(f"🔍 Loaded {len(texts)} tweets from fixed dataset.")

# ─── Twitter API Setup ─────────────────────────────────────────────────────────
"""
BEARER_TOKEN = "YOUR_BEARER_TOKEN_HERE"  # Replace with your actual token
client = tweepy.Client(bearer_token=BEARER_TOKEN)

query = '("Trump tariffs" OR "Trump tariff" OR "trade war" OR "Trump China tariffs" OR "Trump trade policy") -is:retweet lang:en'

# ─── Tweet Fetching with Retry ─────────────────────────────────────────────────
max_retries = 3
for attempt in range(max_retries):
    try:
        time.sleep(1.1)  # Respect 1 req/sec
        tweets = client.search_recent_tweets(query=query, max_results=100)
        texts = [tweet.text for tweet in tweets.data if tweet.text]
        break
    except TooManyRequests:
        print(f"Rate limit hit, retrying in 60s... (attempt {attempt+1})")
        time.sleep(60)
else:
    print("Still rate-limited after retries. No fresh tweets.")
    texts = []

if not texts:
    print("No tweets to process. Try again later.")
    exit()
"""

# ─── Embeddings, Sentiment, UMAP ───────────────────────────────────────────────
embeddings = model.encode(texts)
analyzer = SentimentIntensityAnalyzer()
sentiments = [analyzer.polarity_scores(t)["compound"] for t in texts]

reducer = umap.UMAP(random_state=42)
reduced = reducer.fit_transform(embeddings)

# ─── Clustering with KMeans ────────────────────────────────────────────────────
clusterer = KMeans(n_clusters=4, random_state=42)
labels = clusterer.fit_predict(reduced)

# ─── Cluster Theme Extraction ──────────────────────────────────────────────────
def get_tfidf_theme(texts, top_n=3):
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    X = tfidf.fit_transform(texts)
    scores = np.asarray(X.sum(axis=0)).ravel()
    top_indices = scores.argsort()[::-1][:top_n]
    return ", ".join(np.array(tfidf.get_feature_names_out())[top_indices])

def get_central_summary(texts):
    if not texts:
        return ""
    embs = model.encode(texts)
    center = util.pytorch_cos_sim(embs, embs).sum(dim=1)
    return texts[np.argmax(center)]

cluster_texts = defaultdict(list)
for label, summary in zip(labels, summaries):
    cluster_texts[label].append(summary)

cluster_themes = {}
for label, summaries in cluster_texts.items():
    tfidf_theme = get_tfidf_theme(summaries)
    central_summary = get_central_summary(summaries)
    cluster_themes[label] = f"{tfidf_theme} → {central_summary}"

print(f"texts: {len(texts)}")
print(f"summaries: {len(summaries)}")
print(f"sentiments: {len(sentiments)}")
print(f"reduced: {len(reduced)}")
print(f"labels: {len(labels)}")

cleaned = [
    (t, s, se, r, l)
    for t, s, se, r, l in zip(texts, summaries, sentiments, reduced, labels)
    if t and s and se is not None and r is not None and l is not None
]

if not cleaned:
    print("No complete tweet records to process.")
    exit()

texts, summaries, sentiments, reduced, labels = zip(*cleaned)
reduced = np.array(reduced)

# ─── Prepare DataFrame for Plotting ────────────────────────────────────────────
df = pd.DataFrame({
    "Sentiment": sentiments,
    "Semantic_Y": reduced[:, 1],
    "Summary": summaries,
    "Original_Tweet": texts,
    "Cluster": labels
})
df["Theme"] = df["Cluster"].map(cluster_themes)

# ─── Interactive Plot ──────────────────────────────────────────────────────────
fig = px.scatter(
    df,
    x="Sentiment",
    y="Semantic_Y",
    color="Theme",
    hover_data=["Summary", "Original_Tweet", "Cluster"],
    title="Trump Tariff Tweets: Sentiment and Thematic Clusters (KMeans)",
    template="plotly_white"
)

fig.show()