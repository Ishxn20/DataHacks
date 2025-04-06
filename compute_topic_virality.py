import pandas as pd
from collections import Counter

def compute_topic_features(tweets):
    # Create DataFrame
    df = pd.DataFrame(tweets)

    # Sanity check
    if 'Sentiment' not in df.columns or 'Cluster' not in df.columns or 'Theme' not in df.columns:
        raise ValueError("Missing required keys: 'Sentiment', 'Cluster', or 'Theme'")

    # Core metrics
    avg_sentiment = df['Sentiment'].mean()
    sentiment_volatility = df['Sentiment'].std()
    
    cluster_counts = dict(Counter(df['Cluster']))
    num_clusters = len(cluster_counts)
    dominant_cluster_ratio = max(cluster_counts.values()) / len(df)

    theme_counts = dict(Counter(df['Theme']))
    num_unique_themes = len(theme_counts)

    total_tweets = len(df) 

    # Assemble feature dictionary
    return {
        "avg_sentiment": round(avg_sentiment, 4),
        "sentiment_volatility": round(sentiment_volatility, 4),
        "num_clusters": num_clusters,
        "dominant_cluster_ratio": round(dominant_cluster_ratio, 4),
        "num_unique_themes": num_unique_themes,
        "total_tweets": total_tweets
    }

# Example usage:
if __name__ == "__main__":
    import json
    with open("tweet_data.json", "r") as f:
        tweets = json.load(f)

    features = compute_topic_features(tweets)
    print("📊 Topic Virality Feature Vector:")
    for k, v in features.items():
        print(f"{k}: {v}")