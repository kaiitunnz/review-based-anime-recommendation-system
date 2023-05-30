import os
from typing import Dict, List, Optional

import pandas as pd

import config


def get_anime_reviews(animes: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    return (
        animes[["id", "title"]]
        .rename(columns={"id": "anime_id"})
        .join(
            reviews[["user_id", "anime_id", "text", "score"]]
            .rename(columns={"text": "review"})
            .set_index("anime_id"),
            on="anime_id",
            how="inner",
        )
    )


def get_users_reviews(
    reviews: pd.DataFrame, user_ids: Optional[List[int]]
) -> pd.DataFrame:
    if user_ids is not None:
        reviews = reviews[reviews["user_id"].isin(user_ids)]
    return reviews


def get_users_with_most_reviews(reviews: pd.DataFrame, num: int = 1) -> List[int]:
    return reviews.groupby("user_id")["user_id"].count().nlargest(num).index


def load_data(data_dir) -> Dict[str, pd.DataFrame]:
    return {
        "animes": pd.read_csv(os.path.join(data_dir, "animes.csv")),
        "users": pd.read_csv(os.path.join(data_dir, "users.csv")),
        "reviews": pd.read_csv(os.path.join(data_dir, "reviews.csv")),
    }


def get_model_path(vectorizer: str, sia: str, similarity: str) -> str:
    model_name = "_".join(["model", vectorizer, sia, similarity]) + ".pkl"
    return os.path.join(config.MODEL_DIR, model_name)


def get_vectorized_path(vectorizer: str) -> str:
    return os.path.join(config.VECTORIZED_DIR, "vectorized_" + vectorizer + ".pkl")


def get_sentiment_path(sia: str) -> str:
    return os.path.join(config.SENTIMENT_DIR, "sentiment_" + sia + ".pkl")


def get_vectorizer_path(vectorizer: str) -> str:
    return os.path.join(config.VECTORIZER_DIR, "vectorizer_" + vectorizer + ".pkl")


def init_directory(verbose=False):
    directories = [
        config.CLEANED_DIR,
        config.VECTORIZED_DIR,
        config.SENTIMENT_DIR,
        config.VECTORIZER_DIR,
        config.MODEL_DIR,
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            if verbose:
                print("Created", directory)
