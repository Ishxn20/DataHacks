import pandas as pd
import numpy as np

# ----- Step 1: Load the JSON Data -----
json_filename = "tweet_data.json"  # make sure this file is in your working directory
df = pd.read_json(json_filename)

# Display columns to verify
print("Columns in the JSON file:")
print(df.columns.tolist())

# ----- Step 2: Define Functions to Compute Virality Scores -----
def compute_overall_topic_score(topic_df):
    """
    Computes aggregated features for the overall topic.
    Returns a dictionary with:
      - avg_sentiment: Mean sentiment.
      - sentiment_volatility: Standard deviation of sentiment.
      - total_tweets: Count of tweets.
      - favorability: Proportion of tweets with positive sentiment.
      - emotional_intensity: Average absolute sentiment.
      - composite_score: A composite metric that combines the above.
    """
    avg_sent = topic_df["Sentiment"].mean()
    sentiment_vol = topic_df["Sentiment"].std() if len(topic_df) > 1 else 0.0
    total_tweets = len(topic_df)
    favorability = (topic_df["Sentiment"] > 0).mean()  # ratio of positive tweets
    emotional_intensity = topic_df["Sentiment"].abs().mean()
    
    # Here, composite_score is defined as a function of total volume, emotional intensity, and favorability.
    composite_score = total_tweets * emotional_intensity * favorability
    
    return {
        "avg_sentiment": avg_sent,
        "sentiment_volatility": sentiment_vol,
        "total_tweets": total_tweets,
        "favorability": favorability,
        "emotional_intensity": emotional_intensity,
        "composite_score": composite_score
    }

def compute_subtopic_scores(topic_df):
    """
    Groups the topic data by 'Cluster' (subtopic) and computes the virality score for each group.
    Returns a dictionary with each cluster's aggregated features.
    """
    subtopic_scores = {}
    groups = topic_df.groupby("Cluster")
    
    for cluster, group in groups:
        features = compute_overall_topic_score(group)
        subtopic_scores[cluster] = features
        
    return subtopic_scores

# ----- Step 3: Filter for a Specific Topic and Compute Scores -----
# For example, assume the user wants the virality for the topic "trump, trade, tariffs"
topic_filter = "trump, trade, tariffs"
topic_df = df[df["Theme"] == topic_filter]

if topic_df.empty:
    print(f"No tweets found for topic: {topic_filter}")
else:
    # Compute overall topic-level features and composite score.
    overall_features = compute_overall_topic_score(topic_df)
    
    # Compute subtopic (cluster) virality scores.
    subtopic_features = compute_subtopic_scores(topic_df)
    
    # ----- Step 4: Display the Results -----
    print("\n--- Overall Topic Virality Score ---")
    print(f"Theme: {topic_filter}")
    print(f"Composite Score (Overall Virality): {overall_features['composite_score']:.2f}")
    print("Detailed Overall Features:")
    for key, value in overall_features.items():
        print(f"  {key}: {value}")
    
    print("\n--- Subtopic Virality Scores (by Cluster) ---")
    for cluster, feats in subtopic_features.items():
        print(f"\nCluster {cluster}:")
        print(f"  Composite Score: {feats['composite_score']:.2f}")
        for key, value in feats.items():
            print(f"   {key}: {value}")