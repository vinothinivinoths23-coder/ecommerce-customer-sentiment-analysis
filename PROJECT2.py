# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

# -----------------------------
# 2. LOAD DATASET
# -----------------------------
df = pd.read_csv("E:\\PROJECTS\\ecommerce_reviews PJCT2.csv")
df.head()

# -----------------------------
# 3. CLEAN TEXT FUNCTION
# -----------------------------
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # remove special characters
    text = text.lower()                   # convert to lowercase
    text = text.strip()                   # remove spaces
    return text

df["clean_review"] = df["review"].apply(clean_text)

# -----------------------------
# 4. SENTIMENT ANALYSIS
# -----------------------------
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["clean_review"].apply(get_sentiment)

# -----------------------------
# 5. SENTIMENT DISTRIBUTION
# -----------------------------
plt.figure(figsize=(6,4))
df["sentiment"].value_counts().plot(kind="bar")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -----------------------------
# 6. RATING DISTRIBUTION
# -----------------------------
plt.figure(figsize=(6,4))
df["rating"].value_counts().sort_index().plot(kind="bar")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -----------------------------
# 7. WORD CLOUD
# -----------------------------
all_words = " ".join(df["clean_review"])

wordcloud = WordCloud(width=800, height=400).generate(all_words)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# -----------------------------
# 8. SAVE FINAL OUTPUT
# -----------------------------
df.to_csv("processed_sentiment_output.csv", index=False)
print("Processed output saved as processed_sentiment_output.csv")
