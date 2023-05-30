import os

RAW_DIR = os.path.join("data", "raw")
CLEANED_DIR = os.path.join("data", "cleaned")
VECTORIZED_DIR = os.path.join("data", "vectorized")
SENTIMENT_DIR = os.path.join("data", "sentiment")
VECTORIZER_DIR = "vectorizers"
MODEL_DIR = "models"

MAX_USERS = 100

vectorizers = ["doc2vec-50", "doc2vec-200", "doc2vec-600", "tfidf-1", "tfidf-3"]
sias = ["vader", "linear"]
similarities = ["cosine", "euclidean"]
