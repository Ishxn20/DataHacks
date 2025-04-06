import os
import tweepy
import praw
import instaloader
from dotenv import load_dotenv

# Load environment variables from .env file in the project root
load_dotenv()

def fetch_twitter_data(query, max_results=50):
    """
    Fetch tweets matching the query.
    Requires environment variables:
      - TWITTER_API_KEY
      - TWITTER_API_SECRET
      - TWITTER_ACCESS_TOKEN
      - TWITTER_ACCESS_TOKEN_SECRET
    """
    consumer_key = os.getenv("TWITTER_API_KEY")
    consumer_secret = os.getenv("TWITTER_API_SECRET")
    access_token = os.getenv("TWITTER_ACCESS_TOKEN")
    access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
    
    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        raise ValueError("Twitter API credentials not fully set in environment variables.")

    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
    api = tweepy.API(auth)
    
    # Search for tweets excluding retweets
    tweets = api.search_tweets(q=f"{query} -filter:retweets", lang="en", count=max_results)
    results = []
    for tweet in tweets:
        results.append({"text": tweet.text, "platform": "twitter"})
    return results

def fetch_reddit_data(query, limit=50):
    """
    Fetch Reddit posts matching the query.
    Requires environment variables:
      - REDDIT_CLIENT_ID
      - REDDIT_CLIENT_SECRET
      - REDDIT_USER_AGENT (optional)
    """
    reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
    reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "EchoChamberAI/0.1")
    
    if not reddit_client_id or not reddit_client_secret:
        raise ValueError("Reddit API credentials not set in environment variables.")
    
    reddit = praw.Reddit(client_id=reddit_client_id,
                         client_secret=reddit_client_secret,
                         user_agent=reddit_user_agent)
    
    results = []
    for submission in reddit.subreddit("all").search(query, limit=limit):
        text = submission.title
        if submission.selftext:
            text += " " + submission.selftext
        results.append({"text": text, "platform": "reddit"})
    return results

def fetch_instagram_data(query, max_results=10):
    """
    Fetch Instagram data.
    Instagram's API is limited. This function uses Instaloader as a placeholder.
    """
    L = instaloader.Instaloader()
    # Placeholder: Return dummy data
    results = [{"text": f"Instagram placeholder data for query: {query}", "platform": "instagram"}]
    return results

def fetch_all_data(query):
    """
    Aggregate data from Twitter, Reddit, and Instagram.
    """
    data = []
    try:
        twitter_data = fetch_twitter_data(query)
    except Exception as e:
        print("Error fetching Twitter data:", e)
        twitter_data = []
        
    try:
        reddit_data = fetch_reddit_data(query)
    except Exception as e:
        print("Error fetching Reddit data:", e)
        reddit_data = []
        
    try:
        instagram_data = fetch_instagram_data(query)
    except Exception as e:
        print("Error fetching Instagram data:", e)
        instagram_data = []
        
    data.extend(twitter_data)
    data.extend(reddit_data)
    data.extend(instagram_data)
    return data

if __name__ == "__main__":
    query = "Barbenheimer"
    all_data = fetch_all_data(query)
    for item in all_data:
        print(item)