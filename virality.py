import pandas as pd
import numpy as np
import re
from collections import Counter
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

##############################
# TWEET-LEVEL MODEL PIPELINE #
##############################

def load_tweet_csv(csv_filename):
    return pd.read_csv(csv_filename)

def clean_tweet_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_tweet_features(df):
    df['text'] = df['text'].apply(clean_tweet_text)
    df['text_length'] = df['text'].apply(len)
    df['hashtag_count'] = df['text'].apply(lambda x: len(re.findall(r"#\w+", x)))
    return df

def train_tweet_level_model(df, retweet_threshold=100):
    df['viral'] = (df['RetweetCount'] > retweet_threshold).astype(int)
    features = ['Sentiment', 'text_length', 'hashtag_count', 'Reach', 'Likes', 'Klout']
    X = df[features]
    y = df['viral']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Tweet-Level Model Classification Report:")
    print(classification_report(y_test, y_pred))
    return model

#####################################
# TOPIC-LEVEL AGGREGATION & VISUALIZATION (JSON)
#####################################

def load_json_data(json_filename):
    return pd.read_json(json_filename)

def aggregate_by_theme_and_cluster(df):
    """
    Aggregates tweet-level data by both Theme and Cluster.
    Returns a DataFrame with columns:
      Theme, Cluster, avg_sentiment, sentiment_volatility, total_tweets,
      favorability, emotional_intensity, composite_score.
    """
    df['Cluster'] = df['Cluster'].astype(str)  # ensure consistent grouping
    records = []
    grouped = df.groupby(["Theme", "Cluster"])
    for (theme, cluster), group in grouped:
        avg_sent = group["Sentiment"].mean()
        sentiment_vol = group["Sentiment"].std() if len(group) > 1 else 0.0
        total_tweets = len(group)
        favorability = (group["Sentiment"] > 0).mean()
        emotional_intensity = group["Sentiment"].abs().mean()
        composite_score = total_tweets * emotional_intensity * favorability
        records.append({
            "Theme": theme,
            "Cluster": cluster,
            "avg_sentiment": avg_sent,
            "sentiment_volatility": sentiment_vol,
            "total_tweets": total_tweets,
            "favorability": favorability,
            "emotional_intensity": emotional_intensity,
            "composite_score": composite_score
        })
    return pd.DataFrame(records)

def visualize_all_in_one(agg_df):
    """
    Creates a single scatter plot showing:
      - X-axis = emotional_intensity
      - Y-axis = favorability
      - color = Theme
      - symbol = Cluster
      - size = total_tweets
      - hover_data = composite_score, avg_sentiment, etc.
    """
    fig = px.scatter(
        agg_df,
        x="emotional_intensity",
        y="favorability",
        color="Theme",
        symbol="Cluster",
        size="total_tweets",
        hover_data=["composite_score", "avg_sentiment", "sentiment_volatility"],
        title="All Themes & Clusters in One Graph: Emotional Intensity vs. Favorability"
    )
    fig.show()

def aggregate_and_visualize(json_filename):
    df_json = load_json_data(json_filename)
    agg_df = aggregate_by_theme_and_cluster(df_json)
    print("Aggregated Data by Theme and Cluster:")
    print(agg_df)
    visualize_all_in_one(agg_df)

#########################
# INTEGRATED MAIN BLOCK #
#########################

def main():
    # Part 1: Tweet-Level Virality Model (using CSV)
    tweet_csv_filename = "tweets-engagement-metrics_clean.csv"  # Ensure this file exists
    df_tweets = load_tweet_csv(tweet_csv_filename)
    df_tweets = extract_tweet_features(df_tweets)
    tweet_model = train_tweet_level_model(df_tweets, retweet_threshold=100)
    
    # Part 2: Single-Chart Topic-Level Aggregation & Visualization (JSON)
    json_filename = "tweet_data.json"  # Ensure this file has the needed structure
    aggregate_and_visualize(json_filename)

if __name__ == "__main__":
    main()