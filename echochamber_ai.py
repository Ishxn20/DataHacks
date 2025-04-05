from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import umap
import hdbscan
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import torch
from collections import defaultdict, Counter
import tweepy
import json
import os.path
from tweepy.errors import TooManyRequests
import time

# Ensure reproducibility
os.environ["OMP_NUM_THREADS"] = "1"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Fetch tweets from Twitter API
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAP2r0QEAAAAAMZf5st30BtfQagxIIYuCa0ffPbQ%3DpZsRXzlpm5jnyhsmI8LQEwbFvs4vMDiM9dGth5jrBfKork1VS4"  # TODO: Replace with your actual bearer token

client = tweepy.Client(bearer_token=BEARER_TOKEN)
query = '("Trump" OR #Trump) (think OR believe OR feel OR support OR against OR hate OR love) -is:retweet lang:en'

max_retries = 3
for attempt in range(max_retries):
    try:
        time.sleep(1.1)  # enforce 1 req/sec minimum
        tweets = client.search_recent_tweets(query=query, max_results=10)
        texts = [tweet.text for tweet in tweets.data if tweet.text]
        break  # success, exit loop
    except TooManyRequests:
        print(f"Rate limit hit, retrying in 60 seconds... (attempt {attempt+1})")
        time.sleep(60)
else:
    print("Still rate-limited after retries. No fresh tweets available.")
    texts = []

# Step 1: Convert text to vectors
if not texts:
    print("No tweets to process. Try again later.")
    exit()

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings = model.encode(texts)

# Step 2: Reduce dimensions
reducer = umap.UMAP(random_state=42)
reduced = reducer.fit_transform(embeddings)

# Step 3: Cluster
clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
labels = clusterer.fit_predict(reduced)

# Step 4: Analyze sentiment
analyzer = SentimentIntensityAnalyzer()
sentiments = [analyzer.polarity_scores(t)["compound"] for t in texts]

# Step 5: Plot - Sentiment on x-axis, UMAP dimension on y-axis
x_axis = np.array(sentiments)
y_axis = reduced[:, 1]  # semantic dimension from UMAP

plt.figure(figsize=(10, 6))
scatter = plt.scatter(x_axis, y_axis, c=sentiments, cmap='coolwarm', s=100)
plt.colorbar(scatter, label='Sentiment Score')
plt.xlabel("Sentiment (Negative → Positive)")
plt.ylabel("Semantic Dimension (UMAP)")
for i, label in enumerate(labels):
    plt.text(x_axis[i], y_axis[i] + 0.05, f"{label}", ha='center', fontsize=9)

for i, text in enumerate(texts):
    plt.text(x_axis[i], y_axis[i] - 0.05, f"\"{text}\"", fontsize=7, ha='center', wrap=True)

def get_keywords(texts, top_n=2):
    words = " ".join(texts).lower().split()
    keywords = [w for w in words if len(w) > 4 and w.isalpha()]
    return " + ".join([word for word, _ in Counter(keywords).most_common(top_n)])

cluster_texts = defaultdict(list)
for label, text in zip(labels, texts):
    cluster_texts[label].append(text)

for label in cluster_texts:
    if label == -1:
        continue  # skip noise
    x_mean = np.mean([x_axis[i] for i in range(len(texts)) if labels[i] == label])
    y_mean = np.mean([y_axis[i] for i in range(len(texts)) if labels[i] == label])
    short_summary = get_keywords(cluster_texts[label])
    plt.text(x_mean, y_mean + 0.2, f"Cluster {label}: {short_summary}", fontsize=9, color="black", weight='bold', ha='center')

plt.title("Tweets by Sentiment and Semantic Meaning")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()