import pandas as pd
import numpy as np
import re
from collections import Counter
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

def predict_new_tweet(model, tweet_text, reach, likes, klout):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(tweet_text)['compound']
    text_clean = clean_tweet_text(tweet_text)
    text_length = len(text_clean)
    hashtag_count = len(re.findall(r"#\w+", text_clean))
    features = pd.DataFrame([{
        "Sentiment": sentiment,
        "text_length": text_length,
        "hashtag_count": hashtag_count,
        "Reach": reach,
        "Likes": likes,
        "Klout": klout
    }])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    return prediction, probability

#####################################
# TOPIC-LEVEL AGGREGATION & VISUALIZATION (JSON)
#####################################

def load_json_data(json_filename):
    return pd.read_json(json_filename)

def aggregate_by_theme_and_cluster(df):
    df['Cluster'] = df['Cluster'].astype(str)
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

def get_virality_figure_html(json_filename):
    """
    Loads the JSON data, aggregates by Theme and Cluster,
    and returns a single Plotly scatter plot as an HTML string.
    The plot shows:
      - X-axis: emotional_intensity
      - Y-axis: favorability
      - Color: Theme
      - Symbol: Cluster
      - Size: total_tweets
      - Hover data: composite_score, avg_sentiment, sentiment_volatility.
    """
    df_json = load_json_data(json_filename)
    agg_df = aggregate_by_theme_and_cluster(df_json)
    fig = px.scatter(
        agg_df,
        x="emotional_intensity",
        y="favorability",
        color="Theme",
        symbol="Cluster",
        size="total_tweets",
        hover_data=["composite_score", "avg_sentiment", "sentiment_volatility"],
        title="All Themes & Clusters: Emotional Intensity vs. Favorability"
    )
    return fig.to_html(full_html=False)

def aggregate_and_visualize(json_filename):
    df_json = load_json_data(json_filename)
    agg_df = aggregate_by_theme_and_cluster(df_json)
    print("Aggregated Data by Theme and Cluster:")
    print(agg_df)
    # For debugging purposes we can show the plot here:
    fig = px.scatter(
        agg_df,
        x="emotional_intensity",
        y="favorability",
        color="Theme",
        symbol="Cluster",
        size="total_tweets",
        hover_data=["composite_score", "avg_sentiment", "sentiment_volatility"],
        title="All Themes & Clusters: Emotional Intensity vs. Favorability"
    )
    fig.show()

#########################
# INTEGRATED MAIN BLOCK #
#########################

def main():
    # Part 1: Tweet-Level Virality Model (using CSV)
    tweet_csv_filename = "tweets-engagement-metrics_clean.csv"
    df_tweets = load_tweet_csv(tweet_csv_filename)
    df_tweets = extract_tweet_features(df_tweets)
    tweet_model = train_tweet_level_model(df_tweets, retweet_threshold=100)
    
    # Example Prediction
    sample_tweet = "Trump's new tariff announcement has everyone talking! #TradeWar"
    sample_reach = 50000
    sample_likes = 200
    sample_klout = 40
    pred, prob = predict_new_tweet(tweet_model, sample_tweet, sample_reach, sample_likes, sample_klout)
    print("\nNew Tweet Prediction:")
    print(f"Tweet: {sample_tweet}")
    print(f"Predicted Viral: {pred} (1 = viral, 0 = not viral)")
    print(f"Probability of being viral: {prob:.2f}")
    
    # Part 2: Topic-Level Aggregation & Visualization (using JSON)
    json_filename = "tweet_data.json"
    # Here we simply print and show the plot using get_virality_figure_html() for testing.
    graph_html = get_virality_figure_html(json_filename)
    print("\nGraph HTML generated successfully.")
    
if __name__ == "__main__":
    main()