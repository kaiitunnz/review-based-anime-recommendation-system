import os
import time
from typing import Optional, Union

import config
from rears import Rears
from rears.utils import (
    load_model,
    load_review_data,
    load_vectorizer,
    preprocess_review_data,
    save_model,
    save_vectorizer,
)
from rears.vectorizer import Vectorizer
from utils import (
    get_anime_reviews,
    get_model_path,
    get_sentiment_path,
    get_vectorized_path,
    get_vectorizer_path,
    init_directory,
    load_data,
)


def build_models(
    verbose: bool = False,
    timed: bool = False,
):
    start_time = time.time()

    init_directory()
    data = load_data(config.CLEANED_DIR)
    animes, reviews = data["animes"], data["reviews"]
    anime_reviews = get_anime_reviews(animes, reviews)

    for vectorizer in config.vectorizers:
        for sia in config.sias:
            for similarity in config.similarities:
                model_path = get_model_path(vectorizer, sia, similarity)
                print("Building", os.path.basename(model_path) + "...")
                model = load_model(model_path)
                vectorized_anime_reviews = load_review_data(
                    anime_reviews,
                    get_vectorized_path(vectorizer),
                    get_sentiment_path(sia),
                )
                if model is None:
                    loaded_vectorizer: Union[
                        str, Optional[Vectorizer]
                    ] = load_vectorizer(get_vectorizer_path(vectorizer))
                    if loaded_vectorizer is None:
                        loaded_vectorizer = vectorizer
                    model = Rears.init_with(
                        vectorized_anime_reviews=vectorized_anime_reviews,
                        vectorizer=loaded_vectorizer,
                        sia=sia,
                        similarity=similarity,
                    )
                    if vectorized_anime_reviews is None:
                        model.train(anime_reviews, verbose=verbose)
                    save_model(model, model_path)
                    if isinstance(model.vectorizer, Vectorizer):
                        save_vectorizer(
                            model.vectorizer, get_vectorizer_path(vectorizer)
                        )
                preprocess_review_data(
                    model,
                    anime_reviews,
                    get_vectorized_path(vectorizer),
                    get_sentiment_path(sia),
                    save_only=True,
                    verbose=verbose,
                )
                print("Done")

    if timed:
        print()
        print(f"Finished in {time.time() - start_time:.4f} seconds.")


if __name__ == "__main__":
    build_models(verbose=True, timed=True)
