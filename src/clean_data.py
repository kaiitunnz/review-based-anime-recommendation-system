import os
import time
from typing import List

import pandas as pd

import config


def load_unique(filepath: str, subset: List[str], keep="last") -> pd.DataFrame:
    data = pd.read_csv(filepath)
    data.drop_duplicates(subset=subset, inplace=True, keep=keep)
    data.reset_index(inplace=True, drop=True)
    return data


def clean_data(verbose: bool = False, timed: bool = False):
    start_time = time.time()

    if verbose:
        print("Cleaning raw data...")

    animes = load_unique(os.path.join(config.RAW_DIR, "animes.csv"), subset=["uid"])
    users = load_unique(
        os.path.join(config.RAW_DIR, "profiles.csv"), subset=["profile"]
    )
    reviews = load_unique(os.path.join(config.RAW_DIR, "reviews.csv"), subset=["uid"])

    animes = animes.rename(columns={"uid": "id"})
    users = users.rename(columns={"profile": "name"})
    users.insert(0, "id", users.index)
    reviews = (
        reviews.drop(columns=["uid"])
        .rename(columns={"profile": "name", "anime_uid": "anime_id"})
        .join(users[["name", "id"]].set_index("name"), on="name")
        .drop(columns="name")
    )
    reviews.insert(0, "user_id", reviews.pop("id"))

    animes.to_csv(os.path.join(config.CLEANED_DIR, "animes.csv"), index=False)
    users.to_csv(os.path.join(config.CLEANED_DIR, "users.csv"), index=False)
    reviews.to_csv(os.path.join(config.CLEANED_DIR, "reviews.csv"), index=False)

    if verbose:
        print("animes:", animes.columns)
        print("users:", users.columns)
        print("reviews:", reviews.columns)
        print("Done")

    if timed:
        print()
        print(f"Finished in {time.time() - start_time:.4f} seconds.")


if __name__ == "__main__":
    clean_data(verbose=True, timed=True)
