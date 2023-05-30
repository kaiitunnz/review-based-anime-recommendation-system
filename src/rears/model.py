from typing import Callable, Iterable, List, NamedTuple, Optional, Union

import nltk
import numpy as np
import pandas as pd

from . import logger
from .vectorizer import Vectorizer


class RearsConfig(NamedTuple):
    vectorizer: str
    sia: Optional[str]
    similarity: str


class Rears:
    config: RearsConfig
    vectorizer: Optional[Vectorizer]
    sia: Optional[Callable[[str, int], float]]
    similarity: Optional[
        Callable[[np.ndarray, Iterable[np.ndarray]], Iterable[np.ndarray]]
    ]
    animelist: Optional[pd.DataFrame]

    def __init__(
        self,
        vectorizer: str = "doc2vec-50",
        sia: str = "vader",
        similarity: str = "cosine",
    ):
        """
        Parameters
        ----------
        vectorizer : str
            Type of vectorizer to be used to vectorize the reviews.

        sia : Optional[str]
            Type of sentiment analyzer to be used to analyze the reviews. If\
            None, the score linearly mapped to the interval [-1, 1] will be\
            used.

        similarity : str
            The similarity metric to be used.

        Raises
        ------
        ValueError
            If some of the arguments are invalid.
        """
        nltk.download("punkt", quiet=True)

        self.config = RearsConfig(
            vectorizer=vectorizer.lower(),
            sia=sia.lower(),
            similarity=similarity.lower(),
        )
        self.sia = None
        self.vectorizer = None
        self.similarity = None
        self.anime_list = None

        if self.config.vectorizer == "doc2vec-50":
            from .vectorizer import Doc2VecVectorizer

            # Hyperparameters from Gensim's tutorial
            self.vectorizer = Doc2VecVectorizer(vector_size=50, min_count=2, epochs=40)
        elif self.config.vectorizer == "doc2vec-200":
            from .vectorizer import Doc2VecVectorizer

            # Hyperparameters from Gensim's tutorial
            self.vectorizer = Doc2VecVectorizer(vector_size=200, min_count=2, epochs=40)
        elif self.config.vectorizer == "doc2vec-600":
            from .vectorizer import Doc2VecVectorizer

            # Hyperparameters from Gensim's tutorial
            self.vectorizer = Doc2VecVectorizer(vector_size=600, min_count=2, epochs=40)
        elif self.config.vectorizer == "tfidf-1":
            from .vectorizer import TfidfVectorizer

            # See sklearn.feature_extraction.text.TfidfVectorizer for more details
            self.vectorizer = TfidfVectorizer(
                stop_words="english", min_df=0.05, max_features=600
            )
        elif self.config.vectorizer == "tfidf-3":
            from .vectorizer import TfidfVectorizer

            # See sklearn.feature_extraction.text.TfidfVectorizer for more details
            self.vectorizer = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 3), min_df=0.05, max_features=600
            )
        else:
            raise ValueError("Invalid vectorizer")

        if self.config.sia == "vader":
            from .sentiment import vader_sentiment

            self.sia = vader_sentiment
        elif self.config.sia == "linear":
            from .sentiment import linear_sentiment

            self.sia = linear_sentiment
        else:
            raise ValueError("Invalid sentiment intensity analyzer")

        if self.config.similarity == "cosine":
            from .similarity import cosine_similarity

            self.similarity = cosine_similarity
        elif self.config.similarity == "euclidean":
            from .similarity import euclidean_similarity

            self.similarity = euclidean_similarity
        else:
            raise ValueError("Invalid similarity metric")

    @staticmethod
    def init_with(
        vectorized_anime_reviews: Optional[pd.DataFrame] = None,
        vectorizer: Optional[Union[str, Vectorizer]] = None,
        sia: Optional[Union[str, Callable]] = None,
        similarity: Optional[Union[str, Callable]] = None,
    ):
        """
        Parameters
        ----------
        vectorized_anime_reviews : Optional[pd.DataFrame]
            A Pandas DataFrame that contains all the vectorized reviews of all\
            the animes. It is expected to have the "anime_id", "title", and\
            "review_vector" columns. It will be used as the model's internal\
            database. If not given, the model must be trained before used.
        
        vectorizer : Optional[Union[str, Vectorizer]]
            Either a vectorizer or the type of the vectorizer to be used. If not\
            given, the default vectorizer will be initialized and used.
        
        sia : Optional[Union[str, Callable]]
            Either a callable object of a sentiment analyzer or the type of the\
            sentiment analyzer to be used. If not given, the default sentiment\
            analyzer will be initialized and used.
        
        similarity : Optional[Union[str, Callable]]
            Either a callable object that calculates a similarity core or the\
            type of the similarity score to be used. If not given, the default\
            similarity score will be used.

        Returns
        -------
        Rears
            a Rears instance initialized with the given arguments
        """
        rears_args = {
            "vectorizer": vectorizer if isinstance(vectorizer, str) else None,
            "sia": sia if isinstance(sia, str) else None,
            "similarity": similarity if isinstance(similarity, str) else None,
        }
        model = Rears(**{k: v for k, v in rears_args.items() if v is not None})
        if isinstance(vectorizer, Vectorizer):
            model.vectorizer = vectorizer
        if vectorized_anime_reviews is not None:
            vectorized_anime_reviews = vectorized_anime_reviews[
                ["anime_id", "title", "review_vector"]
            ]
            anime_list = vectorized_anime_reviews.groupby("anime_id").agg(
                title=pd.NamedAgg(column="title", aggfunc=lambda x: list(x)[0]),
                vectors=pd.NamedAgg(column="review_vector", aggfunc=list),
            )
            model.anime_list = anime_list
        if callable(sia):
            model.sia = sia
        if callable(similarity):
            model.similarity = similarity
        return model

    def train(self, anime_reviews: pd.DataFrame, verbose: bool = False, log_level=0):
        """
        Parameters
        ----------
        anime_reviews : DataFrame
            A Pandas DataFrame that contains all the reviews of all the animes.\
            It is expected to have the "anime_id", "title", and "review" columns,\
            which are the anime IDs, titles, and reviews, respectively.
        
        verbose : bool
            Whether to print out the log at each step.
        
        Raises
        ------
        ValueError
            If some of the configurations are invalid.
        """
        logger.log("Training...", verbose, log_level + 1)

        if self.vectorizer is None:
            raise ValueError("The vectorizer has not been initialized.")

        anime_reviews = anime_reviews[["anime_id", "title", "review"]]

        logger.log("Preprocessing anime reviews", verbose, log_level + 1)
        preprocessed_reviews = self.vectorizer.preprocess(
            anime_reviews["review"].values, verbose
        )
        logger.log("Done", verbose, log_level + 1)

        self.vectorizer.train(preprocessed_reviews, verbose, log_level + 1)

        logger.log("Vectorizing the reviews", verbose, log_level + 1)
        vectorized_reviews = list(
            self.vectorizer.vectorize(preprocessed_reviews, verbose)
        )
        logger.log("Done", verbose, log_level + 1)

        anime_list = (
            anime_reviews.drop(columns=["review"])
            .assign(vector=vectorized_reviews)
            .groupby("anime_id")
            .agg(
                title=pd.NamedAgg(column="title", aggfunc=lambda x: list(x)[0]),
                vectors=pd.NamedAgg(column="vector", aggfunc=list),
            )
        )
        self.anime_list = anime_list

    def generate_recommendation_list(
        self,
        user_reviews: pd.DataFrame,
        num: int = 10,
        show_score: bool = False,
        verbose: bool = False,
        log_level: int = 0,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        user_reviews : DataFrame
            A Pandas DataFrame that contains all the reviews written by a user.\
            It is expected to have the "anime_id", "review", and "score" columns.
        
        num : int
            Number of recommendations to be generated. -1 to return all the\
            animes sorted by relevance.
        
        show_score : bool
            Whether to include the relevance scores in the return result.
        
        verbose : bool
            Whether to print out the log at each step.
        
        Raises
        ------
        ValueError
            If some of the configurations are invalid.
        """
        logger.log("Generating recommendations...", verbose, log_level + 1)

        if self.anime_list is None:
            raise ValueError("The model has not been trained.")

        user_reviews = user_reviews[["anime_id", "review", "score"]]

        review_vectors = self.vectorize(
            user_reviews["review"].values.tolist(), verbose, log_level + 1
        )
        review_vectors = np.array([vector.tolist() for vector in review_vectors])
        sentiment_scores = self.sentiment_scores(
            user_reviews["review"].values.tolist(),
            user_reviews["score"].values.tolist(),
            verbose,
            log_level + 1,
        )

        relevance_scores = self.compute_relevance_scores(
            review_vectors, sentiment_scores, verbose, log_level + 1
        )

        recommendation = self.get_anime_ranking(relevance_scores, show_score)
        recommendation = recommendation[
            ~recommendation.index.isin(user_reviews["anime_id"])
        ]

        logger.log("Finish generating recommendations.", verbose, log_level + 1)

        return recommendation.iloc[:num]

    def vectorize(
        self, reviews: List[str], verbose: bool = False, log_level: int = 0
    ) -> List[np.ndarray]:
        """
        Parameters
        ----------
        reviews : List[str]
            A list of review strings. It should be of the same length as scores.

        verbose : bool
            Whether to print out the log at each step.
        """
        logger.log("Preprocessing user reviews", verbose, log_level + 1)

        if self.vectorizer is None:
            raise ValueError("The vectorizer has not been initialized.")

        preprocessed = self.vectorizer.preprocess(reviews, verbose)
        logger.log("Done", verbose, log_level + 1)

        logger.log("Vectorizing user reviews", verbose, log_level + 1)
        review_vectors = list(self.vectorizer.vectorize(preprocessed, verbose))
        logger.log("Done", verbose, log_level + 1)
        return review_vectors

    def sentiment_scores(
        self,
        reviews: List[str],
        scores: List[int],
        verbose: bool = False,
        log_level: int = 0,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        reviews : List[str]
            A list of review strings. It should be of the same length as scores.
        
        scores : List[str]
            A list of scores associated with the reviews. It should be of the\
            same length as reviews

        verbose : bool
            Whether to print out the log at each step.
        
        Raises
        ------
        ValueError
            If reviews and scores are of different lengths.
        """
        logger.log("Calculating sentiment scores", verbose, log_level + 1)

        if self.sia is None:
            raise ValueError("The sentiment analyzer has not been initialized.")
        if len(reviews) != len(scores):
            raise ValueError("reviews and scores are of different lengths.")

        sentiment_scores = np.array(
            [self.sia(review, score) for review, score in zip(reviews, scores)]
        )
        logger.log("Done", verbose, log_level + 1)
        return sentiment_scores

    def compute_relevance_scores(
        self,
        review_vectors: np.ndarray,
        sentiment_scores: np.ndarray,
        verbose: bool = False,
        log_level: int = 0,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        review_vectors : np.ndarray
            A NumPy array of shape (n, d) where n is the number of reviews and\
            d is the vectors' dimension. Each of its rows is a vectorized review.
        
        sentiment_scores : np.ndarray
            A NumPy array of shape (n,) where n is the number of reviews. Each\
            of its rows is the sentiment score of the corresponding review.
        
        verbose : bool
            Whether to print out the log at each step.
        """
        logger.log("Calculating similarity scores", verbose, log_level + 1)

        if self.anime_list is None:
            raise ValueError("The model has not been trained.")

        anime_list_reviews = [
            np.array(anime_reviews)
            for anime_reviews in self.anime_list["vectors"].values
        ]
        similarity_scores = self.similarity(
            review_vectors, logger.tqdm(anime_list_reviews, verbose)
        )
        logger.log("Done", verbose, log_level + 1)

        logger.log("Calculating relevance scores", verbose, log_level + 1)
        relevance_scores = np.array(
            [
                np.array(sentiment_scores * similarity.T).mean()
                for similarity in logger.tqdm(similarity_scores, verbose)
            ]
        )
        logger.log("Done", verbose, log_level + 1)
        return relevance_scores

    def get_anime_ranking(
        self, relevance_scores: List[float], show_score: bool = False
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        relevance_scores : List[float]
            A list of relevance scores of length equal to the number of animes\
            in the model database
        
        show_score : bool
            Whether to include the relevance scores in the return result.
        """
        if self.anime_list is None:
            raise ValueError("The model has not been trained.")

        recommendation = self.anime_list.assign(relevance_score=relevance_scores)
        recommendation.sort_values("relevance_score", inplace=True, ascending=False)
        return recommendation[["title", "relevance_score"] if show_score else ["title"]]
