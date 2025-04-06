import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ----- Step 1: Load the Tweet-Level CSV Dataset -----
csv_filename = "tweets-engagement-metrics.csv"  # update path if needed
df = pd.read_csv(csv_filename)

# Print out available columns for verification
print("Columns in the CSV file:")
print(df.columns.tolist())

# ----- Step 2: Define a Function to Aggregate Topic-Level Features -----
def aggregate_topic_features(keyword, df):
    """
    Filter tweets that contain the keyword (case-insensitive) and aggregate features.
    
    Returns a dictionary with aggregated topic features or None if no tweets match.
    """
    # Filter tweets containing the keyword in the 'text' column
    keyword = keyword.lower()
    topic_df = df[df['text'].str.lower().str.contains(keyword)]
    
    if topic_df.empty:
        return None
    
    avg_sent = topic_df['Sentiment'].mean()
    # Use 0 if only one tweet, else std deviation
    sentiment_vol = topic_df['Sentiment'].std() if len(topic_df) > 1 else 0.0
    total_tweets = len(topic_df)
    unique_users = topic_df['UserID'].nunique()
    
    # Since we don't have clustering, we'll default these values
    num_clusters = 1
    dominant_cluster_ratio = 1.0

    return {
        "topic": keyword,
        "avg_sentiment": avg_sent,
        "sentiment_volatility": sentiment_vol,
        "num_clusters": num_clusters,
        "dominant_cluster_ratio": dominant_cluster_ratio,
        "unique_users": unique_users,
        "total_tweets": total_tweets
    }

# ----- Step 3: Create an Aggregated Dataset for Several Topics -----
def create_aggregated_dataset(keywords, df, tweet_threshold=100):
    """
    For each keyword in 'keywords', aggregate topic features.
    Also, label a topic as viral (1) if total_tweets >= tweet_threshold, else 0.
    
    Returns a DataFrame of aggregated topic features.
    """
    records = []
    for kw in keywords:
        features = aggregate_topic_features(kw, df)
        if features is not None:
            # Define the label: viral = 1 if total_tweets meets or exceeds threshold, else 0.
            features['viral'] = 1 if features['total_tweets'] >= tweet_threshold else 0
            records.append(features)
    return pd.DataFrame(records)

# ----- Step 4: Define Keywords (Topics) for Training -----
# You can adjust this list based on what topics you expect to analyze.
topics = ["trump", "tariff", "MAGA", "economy", "covid", "sports"]

aggregated_df = create_aggregated_dataset(topics, df, tweet_threshold=100)
if aggregated_df.empty:
    raise ValueError("No topics found with the given keywords. Please check your CSV or keywords.")
    
print("\nAggregated Topic-Level Data:")
print(aggregated_df)

# ----- Step 5: Train a Topic Virality Model -----
# Feature columns used for training:
feature_cols = ['avg_sentiment', 'sentiment_volatility', 'num_clusters', 
                'dominant_cluster_ratio', 'unique_users', 'total_tweets']
X = aggregated_df[feature_cols]
y = aggregated_df['viral']

# Split data (if dataset is very small, training might be trivial)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nClassification Report for Topic Virality Model:")
print(classification_report(y_test, y_pred))

# ----- Step 6: Define a Function to Predict Virality for a New Topic -----
def predict_topic_virality(keyword, df, model):
    """
    Given a keyword for a topic, aggregate its features and use the trained model to predict virality.
    Returns a prediction (1 = viral, 0 = not viral) and the probability of being viral.
    """
    features = aggregate_topic_features(keyword, df)
    if features is None:
        raise ValueError(f"No tweets found for topic '{keyword}'.")
    # Create a DataFrame for model input (only use the feature columns)
    input_features = pd.DataFrame([{
        'avg_sentiment': features['avg_sentiment'],
        'sentiment_volatility': features['sentiment_volatility'],
        'num_clusters': features['num_clusters'],
        'dominant_cluster_ratio': features['dominant_cluster_ratio'],
        'unique_users': features['unique_users'],
        'total_tweets': features['total_tweets']
    }])
    probability = model.predict_proba(input_features)[0][1]
    prediction = model.predict(input_features)[0]
    return features, prediction, probability

# ----- Step 7: Example Usage: Predict Virality for a New Topic -----
new_topic = "trump tariffs"  # You can change this keyword to test another topic
try:
    features, pred, prob = predict_topic_virality(new_topic, df, model)
    print("\n--- New Topic Prediction ---")
    print(f"Topic Keyword: {new_topic}")
    print("Aggregated Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    print(f"Predicted Viral: {pred} (1 means viral, 0 means not viral)")
    print(f"Probability of being viral: {prob:.2f}")
except ValueError as e:
    print(e)