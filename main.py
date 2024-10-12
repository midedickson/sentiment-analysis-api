import re

# Download the 'punkt' resource
# pip install nltk
import pandas as pd
from fastapi import FastAPI
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

twitter_data = pd.read_csv("Twitter_Data_sentiment.csv")
twitter_data = twitter_data[["clean_text", "sentiment"]]
twitter_data.columns = ["text", "sentiment"]
twitter_data["sentiment"] = twitter_data["sentiment"].map(
    {"positive": 1, "negative": -1, "neutral": 0}
)


# Define a preprocessing function
# Load IMDB Movie Reviews dataset
imdb_reviews = pd.read_csv(
    "reviews_with_ids_and_ratings.csv"
)  # Adjust the file path if needed
# The dataset has 4 columns: 'ID', 'Review', 'Sentiment', 'Rating'
# Rename columns for consistency
imdb_reviews.columns = ["ID", "Review", "Sentiment", "Rating"]
# Map sentiment if necessary: positive -> 1, negative -> 0
imdb_reviews["Sentiment"] = imdb_reviews["Sentiment"].map(
    {"positive": 1, "negative": 0}
)
# Load train.csv
train_data = pd.read_csv("train.csv")
train_data = train_data[["textID", "text", "sentiment"]]
train_data.columns = ["ID", "text", "sentiment"]
train_data["sentiment"] = train_data["sentiment"].map(
    {"positive": 1, "negative": -1, "neutral": 0}
)


# Define a preprocessing function
def preprocess_text(text):
    if isinstance(text, str):  # Check if the text is a string
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"@\w+", "", text)  # Remove mentions
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
        tokens = word_tokenize(text)  # Tokenize the text
        tokens = [
            word for word in tokens if word not in stopwords.words("english")
        ]  # Remove stopwords
        return " ".join(tokens)
    else:
        return ""  # Return an empty string if text is not a string


# Apply preprocessing to all datasets
imdb_reviews["Review"] = imdb_reviews["Review"].apply(preprocess_text)
train_data["text"] = train_data["text"].apply(preprocess_text)
twitter_data["text"] = twitter_data["text"].apply(preprocess_text)
# Select only the 'text' and 'sentiment' columns and combine
combined_data = pd.concat(
    [
        imdb_reviews[["Review", "Sentiment"]],
        train_data[["text", "sentiment"]],
        twitter_data[["text", "sentiment"]],
    ],
    ignore_index=True,
)
# Drop rows with NaN values in 'text' and 'sentiment' columns
combined_data = combined_data.dropna(subset=["text", "sentiment"])
# Drop rows with empty strings in 'text' column
combined_data = combined_data[combined_data["text"] != ""]
# Ensure the sentiment column has valid numerical values
combined_data = combined_data[combined_data["sentiment"].notna()]
# Ensure the sentiment column has valid numerical values
combined_data = combined_data[combined_data["sentiment"].notna()]
# Split data into training and testing sets
train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)


# Create a pipeline that first vectorizes the text, then applies Naive Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB())
# Train the model on the combined dataset
model.fit(train_data["text"], train_data["sentiment"])


class TextInput(BaseModel):
    text: str


app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/analyze")
async def read_item(text_input: TextInput):
    # Example input text
    # test_input = [
    #     "I love this movie! It's amazing.",
    #     "This product is terrible and I want a refund.",
    #     "The weather is okay, not too bad.",
    # ]
    # Preprocess the input text
    preprocessed_input = [preprocess_text(text_input.text)]
    # Predict sentiment
    predicted_sentiments = model.predict(preprocessed_input)
    # Map numeric predictions back to sentiment labels if necessary
    # Assuming the sentiment mapping is: 1 = positive, 0 = neutral, -1 = negative
    sentiment_mapping = {1: "positive", 0: "neutral", -1: "negative"}
    predicted_labels = [
        sentiment_mapping.get(pred, "unknown") for pred in predicted_sentiments
    ]
    # Display the results
    results = []
    for text, sentiment in zip([text_input.text], predicted_labels):
        results.append({"text": text, "sentiment": sentiment})
        # print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")
    return results
