# import os
# import time
# import numpy as np
# import random
# import torch
# import tweepy
# import pandas as pd
# import plotly.express as px
# import json
# import sys
# from flask import request
# from sentence_transformers import SentenceTransformer, util
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import umap
# from sklearn.cluster import KMeans
# from collections import defaultdict, Counter
# from tweepy.errors import TooManyRequests

# # ─── Setup ─────────────────────────────────────────────────────────────────────
# os.environ["OMP_NUM_THREADS"] = "1"
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

# # ─── Twitter API Setup ─────────────────────────────────────────────────────────
# BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAADK30QEAAAAAIEmdbNAlzKr052%2Fle4g9m9k9keE%3DJZ8BuuCilu1ZZoJthy1D4BmG0rV7bIhGGoO8o2SVtme7vuNYXj"  # Replace with your actual token
# client = tweepy.Client(bearer_token=BEARER_TOKEN)

# query = request.form.get("query", default='("Trump tariffs" OR "trade war")')

# # ─── Tweet Fetching with Retry ─────────────────────────────────────────────────
# max_retries = 3
# try:
#     time.sleep(4)
#     tweets = client.search_recent_tweets(query=query, max_results=10)
#     texts = [tweet.text for tweet in tweets.data if tweet.text]
# except TooManyRequests:
#     print("Rate limit hit. Loading from latest cached file instead.")
#     latest_file = sorted([f for f in os.listdir() if f.startswith("tweet_data_") and f.endswith(".json")])[-1]
#     with open(latest_file, "r") as f:
#         output_data = json.load(f)
#     texts = [item["Original_Tweet"] for item in output_data]

# # ─── Model Setup ───────────────────────────────────────────────────────────────
# model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# # ─── Smart Summarization ───────────────────────────────────────────────────────
# def summarize_tweet(text):
#     sentences = text.split(". ")
#     if len(sentences) == 1:
#         return text.strip()
#     embeddings = model.encode(sentences)
#     center = util.pytorch_cos_sim(embeddings, embeddings).sum(dim=1)
#     return sentences[center.argmax().item()].strip()

# summaries = [summarize_tweet(t) for t in texts]

# # ─── Embeddings, Sentiment, UMAP ───────────────────────────────────────────────
# embeddings = model.encode(texts)
# analyzer = SentimentIntensityAnalyzer()
# sentiments = [analyzer.polarity_scores(t)["compound"] for t in texts]

# reducer = umap.UMAP(random_state=42)
# reduced = reducer.fit_transform(embeddings)

# # ─── Clustering with KMeans ────────────────────────────────────────────────────
# clusterer = KMeans(n_clusters=4, random_state=42)
# labels = clusterer.fit_predict(reduced)

# # ─── Cluster Theme Extraction ──────────────────────────────────────────────────
# def get_keywords(texts, top_n=3):
#     words = " ".join(texts).lower().split()
#     keywords = [w for w in words if len(w) > 4 and w.isalpha()]
#     return ", ".join([word for word, _ in Counter(keywords).most_common(top_n)])

# cluster_texts = defaultdict(list)
# for label, summary in zip(labels, summaries):
#     cluster_texts[label].append(summary)

# cluster_themes = {label: get_keywords(texts) for label, texts in cluster_texts.items()}

# # ─── Prepare DataFrame for Plotting ────────────────────────────────────────────
# df = pd.DataFrame({
#     "Sentiment": sentiments,
#     "Semantic_Y": reduced[:, 1],
#     "Summary": summaries,
#     "Original_Tweet": texts,
#     "Cluster": labels
# })
# df["Theme"] = df["Cluster"].map(cluster_themes)

# # ─── Save to JSON ──────────────────────────────────────────────────────────────
# output_data = df[["Original_Tweet", "Summary", "Sentiment", "Cluster", "Theme"]].to_dict(orient="records")
# timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
# filename = f"tweet_data_{timestamp}.json"

# with open(filename, "w") as f:
#     json.dump(output_data, f, indent=2)

# print(f"✅ Saved {len(output_data)} tweets to {filename}")

# # ─── Interactive Plot ──────────────────────────────────────────────────────────
# fig = px.scatter(
#     df,
#     x="Sentiment",
#     y="Semantic_Y",
#     color="Theme",
#     hover_data=["Summary", "Original_Tweet", "Cluster"],
#     title="Trump Tariff Tweets: Sentiment and Thematic Clusters (KMeans)",
#     template="plotly_white"
# )

# fig.show()



import os
import time
import json
import tweepy
from sentence_transformers import SentenceTransformer, util
import umap
from sklearn.cluster import KMeans
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from collections import defaultdict, Counter
import random
import numpy as np
import torch
# Load environment variables for authentication

# ─── Setup ─────────────────────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ─── Twitter API Setup ─────────────────────────────────────────────────────────
# Replace with your actual bearer token

BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAJ2t0QEAAAAAz674NB1tfFSd5%2BnSpqv8cqlcoEM%3DFusVdPAZULnHFsZSBmYFU9YoGSJh19g0jsXDEyLpgy4903TtEl"
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# ─── Model & Analyzer Setup ─────────────────────────────────────────────────────
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
analyzer = SentimentIntensityAnalyzer()

def fetch_all_data(query):
    # Fetch tweets using the Twitter API
    try:
        time.sleep(15)  # delay to help with rate limits
        tweets = client.search_recent_tweets(query=query, max_results=10)
        texts = [tweet.text for tweet in tweets.data if tweet.text]
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        texts = []
    
    if not texts:
        raise Exception("No tweets fetched.")
    
    # Instead of our previous "smart" summarization, display a summary as the first few words.
    summaries = [ " ".join(t.split()[:10]) + "..." if len(t.split()) > 10 else t for t in texts ]
    
    # Perform sentiment analysis
    sentiments = [analyzer.polarity_scores(t)["compound"] for t in texts]
    
    # Get embeddings and reduce dimensions using UMAP.
    embeddings = model.encode(texts)
    # Set n_neighbors dynamically to avoid UMAP error on small datasets
    n_neighbors = max(2, min(5, len(texts) - 1))
    reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors)
    reduced = reducer.fit_transform(embeddings)
    
    # Clustering with KMeans (4 clusters)
    clusterer = KMeans(n_clusters=4, random_state=42)
    labels = clusterer.fit_predict(reduced)
    
    # ─── Smart Cluster Theme Extraction ─────────────────────────────────────────
    def get_keywords(texts, top_n=3):
        words = " ".join(texts).lower().split()
        keywords = [w for w in words if len(w) > 4 and w.isalpha()]
        return ", ".join([word for word, _ in Counter(keywords).most_common(top_n)])
    
    # Group summaries by cluster label
    cluster_texts = defaultdict(list)
    for label, summary in zip(labels, summaries):
        cluster_texts[label].append(summary)
    
    # Generate a smart theme for each cluster
    cluster_themes = {}
    for label, summ_list in cluster_texts.items():
        theme = get_keywords(summ_list)
        cluster_themes[label] = theme
    
    # Prepare a DataFrame for plotting
    df = pd.DataFrame({
        "Sentiment": sentiments,
        "Semantic_Y": reduced[:, 1],
        "Summary": summaries,
        "Original_Tweet": texts,
        "Cluster": labels
    })
    df["Theme"] = df["Cluster"].map(cluster_themes)
    
    # Create a Plotly scatter plot
    fig = px.scatter(
        df,
        x="Sentiment",
        y="Semantic_Y",
        color="Theme",
        hover_data=["Summary", "Original_Tweet", "Cluster"],
        title=f"Tweets about: {query}",
        template="plotly_white"
    )
    
    # Save output data to a JSON file (optional)
    output_data = df[["Original_Tweet", "Summary", "Sentiment", "Cluster", "Theme"]].to_dict(orient="records")
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"tweet_data.json"
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"✅ Saved {len(output_data)} tweets to {filename}")
    
    return output_data, fig

if __name__ == "__main__":
    import sys
    user_query = sys.argv[1] if len(sys.argv) > 1 else '("Trump tariffs" OR "trade war")'
    fetch_all_data(user_query)