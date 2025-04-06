import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import json
from collections import defaultdict

analyzer = SentimentIntensityAnalyzer()

def load_tweets(json_path):
    with open(json_path, 'r') as f:
        tweets = json.load(f)
    return tweets

def process_tweets(tweets):
    processed = []

    for tweet in tweets:
        text = tweet.get('text', '')
        created_at = tweet.get('created_at') or tweet.get('timestamp')
        if not created_at:
            continue
        ts = pd.to_datetime(created_at)
        retweets = tweet.get('public_metrics', {}).get('retweet_count', 0)
        likes = tweet.get('public_metrics', {}).get('like_count', 0)
        user = tweet.get('author_id', 'unknown')
        
        sentiment = analyzer.polarity_scores(text)['compound']

        processed.append({
            'timestamp': ts,
            'text': text,
            'retweets': retweets,
            'likes': likes,
            'user_id': user,
            'sentiment': sentiment
        })

    return pd.DataFrame(processed)

def compute_topic_features(df, time_granularity='1H'):
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Tweet volume growth
    tweet_volume = df['text'].resample(time_granularity).count()
    growth_rate = tweet_volume.pct_change().fillna(0).mean()

    # Sentiment volatility
    sentiment_volatility = df['sentiment'].std()

    # Retweet velocity (average retweets per hour)
    retweet_velocity = df['retweets'].resample(time_granularity).sum().mean()

    # Unique users
    unique_users = df['user_id'].nunique()

    # Cluster count placeholder (if available)
    num_clusters = len(df['cluster'].unique()) if 'cluster' in df.columns else -1

    return {
        'tweet_volume_growth': growth_rate,
        'sentiment_volatility': sentiment_volatility,
        'retweet_velocity': retweet_velocity,
        'unique_users': unique_users,
        'num_clusters': num_clusters
    }

if __name__ == "__main__":
    # Path to tweet JSON file
    topic_name = "Taylor Swift Grammy Speech"
    tweet_file = "taylor_swift_grammys.json"  # Replace with your local file

    tweets = load_tweets(tweet_file)
    df = process_tweets(tweets)
    features = compute_topic_features(df)

    print(f"\n🔍 Topic: {topic_name}")
    for k, v in features.items():
        print(f"{k}: {round(v, 4)}")