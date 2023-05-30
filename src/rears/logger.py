from typing import Iterable

import tqdm as tq


def log(msg: str, verbose: bool = False, log_level: int = 0):
    if verbose:
        print((" |" * log_level) + msg)


def tqdm(iterable: Iterable, verbose: bool = False):
    return tq.tqdm(iterable, disable=not verbose)
