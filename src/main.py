import argparse
import os
import time
from typing import Optional

import pandas as pd

import config
from benchmark import benchmark
from build_models import build_models
from clean_data import clean_data
from evaluate import evaluate
from rears import Rears
from rears.utils import load_model
from utils import (
    get_anime_reviews,
    get_model_path,
    get_users_with_most_reviews,
    load_data,
)


def run(
    model: Rears,
    reviews_file: Optional[str],
    user_name: Optional[str],
    n: int,
    verbose: bool = False,
    timed: bool = False,
):
    start_time = time.time()
    data = load_data(config.CLEANED_DIR)

    if reviews_file is None:
        users = data["users"]
        if user_name is None:
            user_id = get_users_with_most_reviews(data["reviews"])[0]
            tmp_user_name = users["name"][users["id"] == user_id]
            if len(tmp_user_name) == 0:
                raise ValueError("Invalid user found.")
            user_name = tmp_user_name.values[0]
        else:
            tmp_user_id = users["id"][users["name"] == user_name]
            if len(tmp_user_id) == 0:
                raise ValueError("Username not found.")
            user_id = tmp_user_id.values[0]

        anime_reviews = get_anime_reviews(data["animes"], data["reviews"])
        user_reviews = anime_reviews[anime_reviews["user_id"] == user_id]
    else:
        user_reviews = pd.read_csv(os.path.normpath(reviews_file))

    recommendations = model.generate_recommendation_list(
        user_reviews, n, verbose=verbose
    )
    if user_name is None:
        user_name = "you"
    print()
    print(f"Animes recommended for {user_name}:")
    print(recommendations)
    if timed:
        print()
        print(f"Finished in {time.time() - start_time:.4f} seconds.")


def clean_wrapper(args: argparse.Namespace):
    if args.raw_dir is not None:
        config.RAW_DIR = os.path.normpath(args.raw_dir)
    if args.data_dir is not None:
        config.CLEANED_DIR = os.path.normpath(args.data_dir)
    clean_data(verbose=args.verbose, timed=args.timed)


def run_wrapper(args: argparse.Namespace):
    if args.data_dir is not None:
        config.CLEANED_DIR = os.path.normpath(args.data_dir)
    if args.model_dir is not None:
        config.MODEL_DIR = os.path.normpath(args.model_dir)
    model_path = (
        get_model_path(args.vectorizer, args.sia, args.similarity)
        if args.model is None
        else os.path.normpath(args.model)
    )
    model = load_model(model_path)
    if model is None:
        raise ValueError(f"Cannot load the model at {model_path}")
    run(
        model,
        args.file,
        args.user_name,
        args.n,
        verbose=args.verbose,
        timed=args.timed,
    )


def benchmark_wrapper(args: argparse.Namespace):
    if args.data_dir is not None:
        config.CLEANED_DIR = os.path.normpath(args.data_dir)
    benchmark(verbose=args.verbose, timed=args.timed)


def build_wrapper(args: argparse.Namespace):
    if args.data_dir is not None:
        config.CLEANED_DIR = os.path.normpath(args.data_dir)
    if args.model_dir is not None:
        config.MODEL_DIR = os.path.normpath(args.model_dir)
    if args.vectorized_dir is not None:
        config.VECTORIZED_DIR = os.path.normpath(args.vectorized_dir)
    if args.sentiment_dir is not None:
        config.SENTIMENT_DIR = os.path.normpath(args.sentiment_dir)
    if args.vectorizer_dir is not None:
        config.VECTORIZER_DIR = os.path.normpath(args.vectorizer_dir)
    if not args.all:
        if isinstance(args.vectorizers, list):
            config.vectorizers = args.vectorizers
        if isinstance(args.sias, list):
            config.sias = args.sias
        if isinstance(args.similarities, list):
            config.similarities = args.similarities
    build_models(verbose=args.verbose, timed=args.timed)


def evaluate_wrapper(args: argparse.Namespace):
    if args.data_dir is not None:
        config.CLEANED_DIR = os.path.normpath(args.data_dir)
    if args.model_dir is not None:
        config.MODEL_DIR = os.path.normpath(args.model_dir)
    if args.vectorized_dir is not None:
        config.VECTORIZED_DIR = os.path.normpath(args.vectorized_dir)
    if args.sentiment_dir is not None:
        config.SENTIMENT_DIR = os.path.normpath(args.sentiment_dir)
    if args.vectorizer_dir is not None:
        config.VECTORIZER_DIR = os.path.normpath(args.vectorizer_dir)
    if not args.all:
        if isinstance(args.vectorizers, list):
            config.vectorizers = args.vectorizers
        if isinstance(args.sias, list):
            config.sias = args.sias
        if isinstance(args.similarities, list):
            config.similarities = args.similarities
    if args.max_users is not None:
        config.MAX_USERS = args.max_users
    evaluate(verbose=args.verbose, timed=args.timed)


def get_arg_parser() -> argparse.ArgumentParser:
    global_parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = global_parser.add_subparsers(
        title="Commands", help="anime recommendation", required=True
    )

    clean_parser = subparsers.add_parser("clean", help="clean the raw data from Kaggle")
    clean_parser.add_argument("-v", "--verbose", action="store_true", default=False)
    clean_parser.add_argument(
        "-t",
        "--timed",
        action="store_true",
        help="whether to time the execution",
        default=False,
    )
    clean_parser.add_argument(
        "--raw-dir", help="overwrite the path to the raw data directory"
    )
    clean_parser.add_argument(
        "--data-dir", help="overwrite the path to the cleaned data directory"
    )
    clean_parser.set_defaults(func=clean_wrapper)

    run_parser = subparsers.add_parser(
        "run", help="run the anime recommendation system"
    )
    run_parser.add_argument("-v", "--verbose", action="store_true", default=False)
    run_parser.add_argument(
        "-t",
        "--timed",
        action="store_true",
        help="whether to time the execution",
        default=False,
    )
    run_parser.add_argument("-u", "--user-name", help="the username")
    run_parser.add_argument("-f", "--file", help="path to the reviews file")
    run_parser.add_argument(
        "--vectorizer",
        help="the vectorizer to be used (default: %(default)s) (ignored if model is given)",
        default="doc2vec-200",
    )
    run_parser.add_argument(
        "--sia",
        help="the sentiment analyzer to be used (default: %(default)s) (ignored if model is given)",
        default="linear",
    )
    run_parser.add_argument(
        "--similarity",
        help="the similarity metric to be used (default: %(default)s) (ignored if model is given)",
        default="euclidean",
    )
    run_parser.add_argument(
        "--data-dir", help="overwrite the path to the cleaned data directory"
    )
    run_parser.add_argument("--model-dir", help="overwrite the default model directory")
    run_parser.add_argument("-m", "--model", help="the path to the trained model")
    run_parser.add_argument(
        "-n",
        default=10,
        type=int,
        help="the number of animes to be recommended (default: %(default)s)",
    )
    run_parser.set_defaults(func=run_wrapper)

    build_parser = subparsers.add_parser("build", help="build the REARS models")
    build_parser.add_argument("-v", "--verbose", action="store_true", default=False)
    build_parser.add_argument(
        "-t",
        "--timed",
        action="store_true",
        help="whether to time the execution",
        default=False,
    )
    build_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=False,
        help="build all the available models",
    )
    build_parser.add_argument(
        "--vectorizers",
        help="the list of vectorizers to be used",
        action="extend",
        default=[],
        nargs="+",
    )
    build_parser.add_argument(
        "--sias",
        help="the list of sentiment analyzers to be used",
        action="extend",
        default=[],
        nargs="+",
    )
    build_parser.add_argument(
        "--similarities",
        help="the list of similarity metrics to be used",
        action="extend",
        default=[],
        nargs="+",
    )
    build_parser.add_argument(
        "--model-dir", help="overwrite the default model directory"
    )
    build_parser.add_argument("--data-dir", help="overwrite the default data directory")
    build_parser.add_argument(
        "--vectorized-dir", help="overwrite the default vectorized directory"
    )
    build_parser.add_argument(
        "--sentiment-dir", help="overwrite the default sentiment directory"
    )
    build_parser.add_argument(
        "--vectorizer-dir", help="overwrite the default vectorizer directory"
    )
    build_parser.set_defaults(func=build_wrapper)

    eval_parser = subparsers.add_parser(
        "evaluate",
        aliases=["eval"],
        help="evaluate the REARS models",
    )
    eval_parser.add_argument("-v", "--verbose", action="store_true", default=False)
    eval_parser.add_argument(
        "-t",
        "--timed",
        action="store_true",
        help="whether to time the execution",
        default=False,
    )
    eval_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=False,
        help="build all the available models",
    )
    eval_parser.add_argument(
        "--vectorizers",
        help="the list of vectorizers to be used",
        action="extend",
        default=[],
        nargs="+",
    )
    eval_parser.add_argument(
        "--sias",
        help="the list of sentiment analyzers to be used",
        action="extend",
        default=[],
        nargs="+",
    )
    eval_parser.add_argument(
        "--similarities",
        help="the list of similarity metrics to be used",
        action="extend",
        default=[],
        nargs="+",
    )
    eval_parser.add_argument(
        "--model-dir", help="overwrite the default model directory"
    )
    eval_parser.add_argument("--data-dir", help="overwrite the default data directory")
    eval_parser.add_argument(
        "--vectorized-dir", help="overwrite the default vectorized directory"
    )
    eval_parser.add_argument(
        "--sentiment-dir", help="overwrite the default sentiment directory"
    )
    eval_parser.add_argument(
        "--vectorizer-dir", help="overwrite the default vectorizer directory"
    )
    eval_parser.add_argument(
        "-mu",
        "--max-users",
        type=int,
        help="maximum number of users to evaluate the models (default: %(default)s)",
        default=config.MAX_USERS,
    )
    eval_parser.set_defaults(func=evaluate_wrapper)

    bench_parser = subparsers.add_parser(
        "benchmark",
        aliases=["bench"],
        help="evaluate the benchmark model",
    )
    bench_parser.add_argument("-v", "--verbose", action="store_true", default=False)
    bench_parser.add_argument(
        "-t",
        "--timed",
        action="store_true",
        help="whether to time the execution",
        default=False,
    )
    bench_parser.add_argument(
        "--data-dir", help="overwrite the path to the cleaned data directory"
    )
    bench_parser.set_defaults(func=benchmark_wrapper)

    return global_parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
