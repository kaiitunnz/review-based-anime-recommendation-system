import time

import nltk
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score

import config
from rears import logger
from utils import get_users_with_most_reviews, load_data


def spearman_correlation(user_ratings: pd.DataFrame, relevance_scores: pd.DataFrame):
    actual = user_ratings[["anime_id", "score"]].set_index("anime_id")
    predicted = (
        relevance_scores[["anime_id", "relevance"]]
        .set_index("anime_id")
        .filter(items=actual.index, axis=0)
    )
    actual = actual.rank(method="min").to_dict()["score"]
    predicted = predicted.rank(method="min").to_dict()["relevance"]
    return nltk.spearman_correlation(actual, predicted)


def average_precision(
    user_ratings: pd.DataFrame, relevance_scores: pd.DataFrame, n: int = 20
):
    user_ratings = user_ratings.set_index("anime_id")
    relevance_scores = relevance_scores.set_index("anime_id")
    filtered = relevance_scores.filter(items=user_ratings.index, axis=0)
    user_ratings = user_ratings.assign(actual_relevance=0)
    top_n = user_ratings.nlargest(n, columns=["score"]).index
    user_ratings["actual_relevance"][user_ratings.index.isin(top_n)] = 1
    joined = user_ratings.join(filtered, how="inner")
    actual = joined["actual_relevance"].to_numpy()
    predicted = joined["relevance"].to_numpy()
    return average_precision_score(actual, predicted)


def get_top_rated_anime(reviews, user_id):
    temp = reviews[reviews["user_id"] == user_id].sort_values("score", ascending=False)
    anime_id = temp.iloc[0]["anime_id"]
    return anime_id


def get_corr_vector(anime_id, corr_matrix, column_names):
    index = column_names.index(anime_id)
    corr = corr_matrix[index]
    return corr


def calculate_relevance_scores(corr_vector, anime_id, column_names):
    anime_list = []
    for i, corr in enumerate(corr_vector):
        other_anime_id = column_names[i]
        if other_anime_id != anime_id:
            anime_list.append((other_anime_id, corr))
    return pd.DataFrame(anime_list, columns=["anime_id", "relevance"])


def generate_recommendations(user_id, corr_matrix, reviews, column_names):
    top_anime_id = get_top_rated_anime(reviews, user_id)
    corr_vector = get_corr_vector(top_anime_id, corr_matrix, column_names)
    result = calculate_relevance_scores(corr_vector, top_anime_id, column_names)
    return result


def evaluate(corr_matrix, reviews, column_names, verbose):
    data = load_data(config.CLEANED_DIR)
    user_list = get_users_with_most_reviews(data["reviews"], num=config.MAX_USERS)
    correlations = []
    average_precisions = []
    for user_id in logger.tqdm(user_list, verbose):
        user_ratings = reviews[["anime_id", "score"]][reviews["user_id"] == user_id]
        relevance_scores = generate_recommendations(
            user_id, corr_matrix, reviews, column_names
        )
        correlations.append(spearman_correlation(user_ratings, relevance_scores))
        average_precisions.append(average_precision(user_ratings, relevance_scores))
    logger.log("Evaluation summary:", True, 1)
    logger.log(
        f"Spearman correlation: {sum(correlations) / len(correlations)}", True, 1
    )
    logger.log(
        f"Mean Average Precision (mAP): {sum(average_precisions) / len(average_precisions)}",
        True,
        1,
    )
    logger.log("Done", True, 1)


def benchmark(verbose: bool = False, timed: bool = False):
    start_time = time.time()

    logger.log("Evaluating the benchmark model...", verbose, 0)

    logger.log("Loading and preparing data...", verbose, 1)
    data = load_data(config.CLEANED_DIR)
    reviews = data["reviews"]
    reviews = reviews.drop(["text", "scores", "link"], axis=1)
    rating_pivot_table = reviews.pivot_table(
        values="score", index="user_id", columns="anime_id", fill_value=0
    )
    X = rating_pivot_table.T
    column_names = rating_pivot_table.columns.tolist()
    logger.log("Done", verbose, 1)

    logger.log("Computing SVD...", verbose, 1)
    svd = TruncatedSVD(n_components=12, random_state=17)
    reduced = svd.fit_transform(X)
    logger.log("Done", verbose, 1)

    corr_matrix = np.corrcoef(reduced)

    evaluate(corr_matrix, reviews, column_names, verbose)

    if timed:
        print()
        print(f"Finished in {time.time() - start_time:.4f} seconds.")


if __name__ == "__main__":
    benchmark(verbose=True, timed=True)
