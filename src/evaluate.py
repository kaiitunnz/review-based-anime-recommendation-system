import time

import config
from rears import Rears
from rears.utils import (
    evaluate_model,
    load_model,
    load_review_data,
    load_vectorizer,
    preprocess_review_data,
    save_model,
    save_vectorizer,
)
from utils import (
    get_anime_reviews,
    get_model_path,
    get_sentiment_path,
    get_users_reviews,
    get_users_with_most_reviews,
    get_vectorized_path,
    get_vectorizer_path,
    init_directory,
    load_data,
)


def evaluate(verbose=True, timed=True):
    start_time = time.time()

    init_directory()
    data = load_data(config.CLEANED_DIR)
    animes, users, reviews = data["animes"], data["users"], data["reviews"]
    anime_reviews = get_anime_reviews(animes, reviews)
    test_users = get_users_with_most_reviews(anime_reviews, config.MAX_USERS)
    train_users = users["id"][~users["id"].isin(test_users)]

    for vectorizer in config.vectorizers:
        for sia in config.sias:
            for similarity in config.similarities:
                print("Evaluating", "_".join([vectorizer, sia, similarity]) + "...")
                model_path = get_model_path(vectorizer, sia, similarity)
                model = load_model(model_path)
                vectorized_anime_reviews = load_review_data(
                    anime_reviews,
                    get_vectorized_path(vectorizer),
                    get_sentiment_path(sia),
                )
                if model is None:
                    loaded_vectorizer = load_vectorizer(get_vectorizer_path(vectorizer))
                    if loaded_vectorizer is None:
                        loaded_vectorizer = vectorizer
                    model = Rears.init_with(
                        vectorized_anime_reviews=vectorized_anime_reviews,
                        vectorizer=loaded_vectorizer,
                        sia=sia,
                        similarity=similarity,
                    )
                    if vectorized_anime_reviews is None:
                        model.train(
                            get_users_reviews(anime_reviews, train_users),
                            verbose=verbose,
                        )
                    save_model(model, model_path)
                    save_vectorizer(model.vectorizer, get_vectorizer_path(vectorizer))
                if vectorized_anime_reviews is None:
                    vectorized_anime_reviews = preprocess_review_data(
                        model,
                        anime_reviews,
                        get_vectorized_path(vectorizer),
                        get_sentiment_path(sia),
                        save_only=False,
                        verbose=verbose,
                    )
                vectorized_anime_reviews = get_users_reviews(
                    vectorized_anime_reviews, test_users
                )
                evaluate_model(model, vectorized_anime_reviews, verbose=verbose)
                print("Done")

    if timed:
        print()
        print(f"Finished in {time.time() - start_time:.4f} seconds.")


if __name__ == "__main__":
    evaluate(verbose=True, timed=True)
