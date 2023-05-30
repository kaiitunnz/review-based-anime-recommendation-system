import os
import pickle
from pickle import PickleError
from typing import Optional

import nltk
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import Rears, logger
from .vectorizer import Vectorizer


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn  # Disable Scikit-learn's warnings


def load_model(model_path: Optional[str], log_level: int = 0) -> Optional[Rears]:
    """
    Parameters
    ----------
    model_path : Optional[str]
        The path to the model file to be loaded. The file should be saved by\
        the save_model() function.
    
    Returns
    -------
    Optional[Rears]
        the saved model if successful; None, otherwise
    """
    if model_path is None or not os.path.exists(model_path):
        return None
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except (OSError, PickleError) as exc:
        logger.log(
            f'Unable to load the model at "{model_path}": {exc}', True, log_level + 1
        )
    return None


def save_model(model: Rears, model_path: str, log_level: int = 0):
    """
    Parameters
    ----------
    model : Rears
        The model to be saved to disk.

    model_path : str
        The path to which the model will be saved.
    """
    try:
        with open(model_path, "wb") as file:
            pickle.dump(model, file)
    except (OSError, PickleError) as exc:
        logger.log(
            f'Unable to save the model at "{model_path}": {exc}', True, log_level + 1
        )


def save_vectorizer(vectorizer: Vectorizer, vectorizer_path: str, log_level: int = 0):
    """
    Parameters
    ----------
    vectorizer : Vectorizer
        The vectorizer to be saved to disk.

    vectorizer_path : str
        The path to which the vectorizer will be saved.
    """
    try:
        with open(vectorizer_path, "wb") as file:
            pickle.dump(vectorizer, file)
    except (OSError, PickleError) as exc:
        logger.log(
            f'Unable to save the model at "{vectorizer_path}": {exc}',
            True,
            log_level + 1,
        )


def load_vectorizer(
    vectorizer_path: Optional[str], log_level: int = 0
) -> Optional[Vectorizer]:
    """
    Parameters
    ----------
    vectorizer_path : Optional[str]
        The path to the model file to be loaded. The file should be saved by\
        the save_model() function.
    
    Returns
    -------
    Optional[Vectorizer]
        the saved model if successful; None, otherwise
    """
    if vectorizer_path is None or not os.path.exists(vectorizer_path):
        return None
    try:
        with open(vectorizer_path, "rb") as file:
            return pickle.load(file)
    except (OSError, PickleError) as exc:
        logger.log(
            f'Unable to load the model at "{vectorizer_path}": {exc}',
            True,
            log_level + 1,
        )
    return None


def preprocess_review_data(
    model: Rears,
    anime_reviews: pd.DataFrame,
    vectorized_path: Optional[str] = None,
    sentiment_path: Optional[str] = None,
    save_only: bool = False,
    verbose: bool = False,
    log_level: int = 0,
) -> Optional[pd.DataFrame]:
    """
    Parameters
    ----------
    model : Rears
        The trained model.

    anime_reviews : pd.DataFrame
        A Pandas DataFrame that contains the reviews written by different\
        users. It is expected to have the "user_id", "anime_id", "title",\
        "review", and "score" columns.

    vectorized_path : Optional[str]
        The path to which the review vectors will be saved.
    
    sentiment_path : Optional[str]
        The path to which the sentiment scores will be saved.
    
    save_only : bool
        Whether to not return the resulting DataFrame. Setting this to True\
        helps reduce the running time.

    verbose : bool
        Whether to print out the log at each step.
    
    Returns
    -------
    pd.DataFrame
        a Pandas DataFrame that has the following columns: "user_id",\
        "anime_id", "title", "score", "review_vector", and "sentiment_score"

    Raises
    ------
    ValueError
        If some of the arguments are invalid.
    """
    anime_reviews = anime_reviews[
        ["user_id", "anime_id", "title", "review", "score"]
    ].reset_index(drop=True)
    logger.log("Preprocessing data...", verbose, log_level + 1)

    review_vectors = None
    sentiment_scores = None

    if save_only:
        if vectorized_path is not None and not os.path.exists(vectorized_path):
            review_vectors = model.vectorize(
                anime_reviews["review"].values.tolist(), verbose, log_level + 1
            )
            preprocessed = anime_reviews.assign(review_vector=review_vectors)
            try:
                with open(vectorized_path, "wb") as file:
                    pickle.dump(
                        preprocessed[["user_id", "anime_id", "review_vector"]], file
                    )
            except (OSError, PickleError) as exc:
                logger.log(
                    f'Unable to save the review vectors to "{vectorized_path}": {exc}',
                    True,
                    log_level + 1,
                )
        if sentiment_path is not None and not os.path.exists(sentiment_path):
            sentiment_scores = model.sentiment_scores(
                anime_reviews["review"].values.tolist(),
                anime_reviews["score"].values.tolist(),
                verbose,
                log_level + 1,
            )
            preprocessed = anime_reviews.assign(sentiment_score=sentiment_scores)
            try:
                with open(sentiment_path, "wb") as file:
                    pickle.dump(
                        preprocessed[["user_id", "anime_id", "sentiment_score"]], file
                    )
            except (OSError, PickleError) as exc:
                logger.log(
                    f'Unable to save the sentiment scores to "{sentiment_path}": {exc}',
                    True,
                    log_level + 1,
                )
        logger.log("Done", verbose, log_level + 1)
        return None

    if vectorized_path is not None and os.path.exists(vectorized_path):
        try:
            with open(vectorized_path, "rb") as file:
                vectorized = pickle.load(file).set_index(["user_id", "anime_id"])
            review_vectors = (
                anime_reviews.set_index(["user_id", "anime_id"])
                .join(vectorized, how="inner")["review_vector"]
                .tolist()
            )
        except (OSError, PickleError) as exc:
            logger.log(
                f'Unable to load the review vectors at "{vectorized_path}": {exc}',
                True,
                log_level + 1,
            )
    if review_vectors is None:
        review_vectors = vectorize_review_data(
            model, anime_reviews, vectorized_path, verbose, log_level + 1
        )["review_vector"].tolist()

    if sentiment_path is not None and os.path.exists(sentiment_path):
        try:
            with open(sentiment_path, "rb") as file:
                sentiment = pickle.load(file).set_index(["user_id", "anime_id"])
            sentiment_scores = (
                anime_reviews.set_index(["user_id", "anime_id"])
                .join(sentiment, how="inner")["sentiment_score"]
                .tolist()
            )
        except (OSError, PickleError) as exc:
            logger.log(
                f'Unable to load the sentiment scores at "{sentiment_path}": {exc}',
                True,
                log_level + 1,
            )
    if sentiment_scores is None:
        sentiment_scores = sia_review_data(
            model, anime_reviews, sentiment_path, verbose, log_level + 1
        )["sentiment_score"].tolist()

    preprocessed = anime_reviews.drop(columns=["review"]).assign(
        review_vector=review_vectors, sentiment_score=sentiment_scores
    )

    logger.log("Done", verbose, log_level + 1)

    return preprocessed


def vectorize_review_data(
    model: Rears,
    anime_reviews: pd.DataFrame,
    save_to: Optional[str] = None,
    verbose: bool = False,
    log_level: int = 0,
):
    """
    Parameters
    ----------
    model : Rears
        The trained model.

    anime_reviews : pd.DataFrame
        A Pandas DataFrame that contains the reviews written by different\
        users. It is expected to have the "user_id", "anime_id", and "review"\
        columns.

    save_to : Optional[str]
        The path to which the review vectors will be saved.

    verbose : bool
        Whether to print out the log at each step.
    
    Returns
    -------
    pd.DataFrame
        a Pandas DataFrame that has the following columns: "user_id",\
        "anime_id", and "review_vector"

    Raises
    ------
    ValueError
        If some of the arguments are invalid.
    """
    anime_reviews = anime_reviews[["user_id", "anime_id", "review"]].reset_index(
        drop=True
    )
    logger.log("Vectorizing data...", verbose, log_level + 1)

    review_vectors = model.vectorize(
        anime_reviews["review"].values.tolist(), verbose, log_level + 1
    )
    preprocessed = anime_reviews.drop(columns=["review"]).assign(
        review_vector=review_vectors
    )

    if save_to is not None:
        try:
            with open(save_to, "wb") as file:
                pickle.dump(preprocessed, file)
        except (OSError, PickleError) as exc:
            logger.log(
                f'Unable to save the review vectors to "{save_to}": {exc}',
                True,
                log_level + 1,
            )

    logger.log("Done", verbose, log_level + 1)

    return preprocessed


def sia_review_data(
    model: Rears,
    anime_reviews: pd.DataFrame,
    save_to: Optional[str] = None,
    verbose: bool = False,
    log_level: int = 0,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    model : Rears
        The trained model.

    anime_reviews : pd.DataFrame
        A Pandas DataFrame that contains the reviews written by different\
        users. It is expected to have the "user_id", "anime_id", "review", and\
        "score" columns.

    save_to : Optional[str]
        The path to which the reviews with the sentiment scores will be saved.

    verbose : bool
        Whether to print out the log at each step.
    
    Returns
    -------
    pd.DataFrame
        a Pandas DataFrame that has the following columns: "user_id",\
        "anime_id", and "sentiment_score"

    Raises
    ------
    ValueError
        If some of the arguments are invalid.
    """
    anime_reviews = anime_reviews[
        ["user_id", "anime_id", "review", "score"]
    ].reset_index(drop=True)
    logger.log("Performing sentiment analysis...", verbose, log_level + 1)

    sentiment_scores = model.sentiment_scores(
        anime_reviews["review"].values.tolist(),
        anime_reviews["score"].values.tolist(),
        verbose,
    )
    preprocessed = anime_reviews.drop(columns=["review", "score"]).assign(
        sentiment_score=sentiment_scores
    )

    if save_to is not None:
        try:
            with open(save_to, "wb") as file:
                pickle.dump(preprocessed, file)
        except (OSError, PickleError) as exc:
            logger.log(
                f'Unable to save the sentiment scores to "{save_to}": {exc}',
                True,
                log_level + 1,
            )

    logger.log("Done", verbose, log_level + 1)

    return preprocessed


def load_review_data(
    anime_reviews: pd.DataFrame,
    vectorized_path: str,
    sentiment_path: str,
    log_level: int = 0,
) -> Optional[pd.DataFrame]:
    """
    Parameters
    ----------
    data_path : Optional[str]
        The path to the review data file to be loaded. The review data should\
        be preprocessed and saved by the preprocess_review_data() function
    
    Returns
    -------
    pd.DataFrame
        a Pandas DataFrame that has the following columns: "user_id",\
        "anime_id", "title", "score", "review_vector", and "sentiment_score"\
        or None if unsuccessful.
    """
    anime_reviews = anime_reviews[["user_id", "anime_id", "title", "score"]].set_index(
        ["user_id", "anime_id"]
    )
    try:
        with open(vectorized_path, "rb") as file:
            vectorized = pickle.load(file).set_index(["user_id", "anime_id"])
    except (OSError, PickleError) as exc:
        logger.log(
            f'Unable to load the review vectors at "{vectorized_path}": {exc}',
            True,
            log_level + 1,
        )
        return None

    try:
        with open(sentiment_path, "rb") as file:
            sentiment = pickle.load(file).set_index(["user_id", "anime_id"])
    except (OSError, PickleError) as exc:
        logger.log(
            f'Unable to load the sentiment scores at "{sentiment_path}": {exc}',
            True,
            log_level + 1,
        )
        return None

    return (
        anime_reviews.join(vectorized, how="inner")
        .join(sentiment, how="inner")
        .reset_index()
    )


def evaluate_model(
    model: Rears, anime_reviews: pd.DataFrame, verbose: bool = False, log_level: int = 0
):
    """
    Parameters
    ----------
    model : Rears
        The model to be evaluated.

    user_reviews : pd.DataFrame
        A Pandas DataFrame that contains the reviews written by different\
        users. The reviews must have already been preprocessed and vectorized.\
        It is expected to have the "user_id", "anime_id", "review_vector",\
        "sentiment_score", and "score" columns

    verbose : bool
        Whether to print out the log at each step.

    Raises
    ------
    ValueError
        If some of the arguments are invalid.
    """
    anime_reviews = anime_reviews[
        ["user_id", "anime_id", "review_vector", "sentiment_score", "score"]
    ].reset_index(drop=True)
    correlations = []
    average_precisions = []

    logger.log("Evaluating the model...", verbose, log_level + 1)

    for user in logger.tqdm(anime_reviews["user_id"].unique(), verbose):
        user_review_index = anime_reviews.index[anime_reviews["user_id"] == user]
        user_reviews = anime_reviews.iloc[user_review_index]
        user_review_vectors = np.array(
            [vector.tolist() for vector in user_reviews["review_vector"].values]
        )
        sentiment_scores = user_reviews["sentiment_score"].to_numpy()
        relevance_scores = model.compute_relevance_scores(
            user_review_vectors,
            sentiment_scores,
        )
        correlations.append(
            __spearman_correlation(model, user_reviews, relevance_scores)
        )
        average_precisions.append(
            __average_precision(model, user_reviews, relevance_scores)
        )

    logger.log("Evaluation summary:", True, log_level + 1)
    logger.log(
        f"Spearman correlation: {sum(correlations) / len(correlations)}",
        True,
        log_level + 1,
    )
    logger.log(
        f"Mean Average Precision (mAP): {sum(average_precisions) / len(average_precisions)}",
        True,
        log_level + 1,
    )
    logger.log("Done", verbose, log_level + 1)


def __spearman_correlation(
    model: Rears, user_reviews: pd.DataFrame, relevance_scores: np.ndarray
) -> float:
    anime_ranking = model.get_anime_ranking(relevance_scores.tolist(), show_score=True)
    actual = (
        user_reviews[["anime_id", "score"]]
        .set_index("anime_id")
        .rank(method="min")
        .to_dict()
    )["score"]
    predicted = (
        anime_ranking.filter(items=user_reviews["anime_id"].to_list(), axis=0)[
            "relevance_score"
        ]
        .rank(method="min")
        .to_dict()
    )
    return nltk.spearman_correlation(actual, predicted)


def __average_precision_(
    model: Rears, user_reviews: pd.DataFrame, relevance_scores: np.ndarray
) -> float:
    if model.anime_list is None:
        raise ValueError("The model has not been trained.")
    anime_relevance_scores = (
        pd.DataFrame(
            {"anime_id": model.anime_list.index, "relevance": list(relevance_scores)}
        )
        .set_index("anime_id")
        .filter(items=user_reviews["anime_id"], axis=0)
    )["relevance"].tolist()
    user_reviews = user_reviews[["score"]].assign(relevance=0)
    user_reviews["relevance"][user_reviews["score"] > 5] = 1
    actual = user_reviews["relevance"].values.tolist()
    return average_precision_score(actual, anime_relevance_scores)


def __average_precision(
    model: Rears, user_reviews: pd.DataFrame, relevance_scores: np.ndarray, n: int = 20
) -> float:
    if model.anime_list is None:
        raise ValueError("The model has not been trained.")
    user_reviews = user_reviews[["anime_id", "score"]]
    anime_relevance_scores = (
        pd.DataFrame(
            {
                "anime_id": model.anime_list.index,
                "pred_relevance": list(relevance_scores),
            }
        )
        .set_index("anime_id")
        .filter(items=user_reviews["anime_id"], axis=0)
    )
    user_reviews = user_reviews.set_index("anime_id").filter(
        items=anime_relevance_scores.index, axis=0
    )
    user_reviews = user_reviews.assign(actual_relevance=0)
    top_n = user_reviews.nlargest(n, columns=["score"]).index
    user_reviews["actual_relevance"][user_reviews.index.isin(top_n)] = 1
    joined = user_reviews.join(anime_relevance_scores, how="inner")
    actual = joined["actual_relevance"].values.tolist()
    predicted = joined["pred_relevance"].values.tolist()
    return average_precision_score(actual, predicted)
