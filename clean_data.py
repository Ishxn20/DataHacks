import pandas as pd
import re

# Load the CSV data
df = pd.read_csv("tweets-engagement-metrics.csv")

# Example: Drop columns that might contain sensitive data
# (adjust these column names based on what you consider sensitive)
df_clean = df.drop(columns=["UserID", "TweetID"])

# Define a function to clean tweet text:
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Optionally, remove extra spaces and line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Apply the cleaning function to the 'text' column
df_clean["text"] = df_clean["text"].apply(clean_text)

# If you want to redact potential AWS keys or similar secrets from text,
# you can add extra regex patterns. For example:
def remove_secrets(text):
    # Example pattern: AWS Access Key IDs usually start with AKIA and have 16 alphanumeric characters.
    text = re.sub(r"AKIA[0-9A-Z]{16}", "[REDACTED]", text)
    # Example pattern for AWS Secret Access Keys (40-character strings, this is just an example)
    text = re.sub(r"(?i)aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40}", "[REDACTED]", text)
    return text

# Combine both cleaning functions if needed:
def full_clean(text):
    text = clean_text(text)
    text = remove_secrets(text)
    return text

df_clean["text"] = df_clean["text"].apply(full_clean)

# Save the cleaned CSV file
df_clean.to_csv("tweets-engagement-metrics_clean.csv", index=False)

print("Data cleaned and saved to tweets-engagement-metrics_clean.csv")