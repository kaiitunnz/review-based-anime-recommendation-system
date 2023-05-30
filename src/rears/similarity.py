from typing import Iterable

import numpy as np
import sklearn.metrics as metrics


def cosine_similarity(
    user_reviews: np.ndarray, anime_list_reviews: Iterable[np.ndarray]
) -> Iterable[np.ndarray]:
    return (
        metrics.pairwise.cosine_similarity(user_reviews, anime_reviews)
        for anime_reviews in anime_list_reviews
    )


def euclidean_similarity(
    user_reviews: np.ndarray, anime_list_reviews: Iterable[np.ndarray]
) -> Iterable[np.ndarray]:
    for anime_reviews in anime_list_reviews:
        user = user_reviews.reshape(user_reviews.shape[0], 1, -1)
        anime = anime_reviews.reshape(1, anime_reviews.shape[0], -1)
        distance = np.linalg.norm(user - anime, 2, axis=2)
        yield 1.0 / (1.0 + distance)
