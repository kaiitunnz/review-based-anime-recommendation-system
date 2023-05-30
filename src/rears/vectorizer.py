from typing import List, Union

import nltk
import numpy as np
import sklearn.feature_extraction.text as sktext
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from . import logger


class Document:
    inner: Union[List[str], str]

    def __init__(self, inner):
        self.inner = inner


class Vectorizer:
    def train(
        self, doc_list: List[Document], verbose: bool = False, log_level: int = 0
    ):
        raise NotImplementedError("An object of this class should not be used.")

    def vectorize(self, doc_list: List[Document], verbose: bool = False) -> np.ndarray:
        raise NotImplementedError("An object of this class should not be used.")

    def preprocess(self, doc_list: List[str], verbose: bool = False) -> List[Document]:
        raise NotImplementedError("An object of this class should not be used.")


class Doc2VecVectorizer(Vectorizer):
    def __init__(self, **kwargs):
        self.model = Doc2Vec(**kwargs)

    def train(
        self, doc_list: List[Document], verbose: bool = False, log_level: int = 0
    ):
        logger.log("Tagging anime reviews", verbose, log_level + 1)
        tagged_reviews = [
            TaggedDocument(doc.inner, [i])
            for i, doc in enumerate(logger.tqdm(doc_list, verbose))
        ]
        logger.log("Done", verbose, log_level + 1)

        logger.log("Training the vectorizer", verbose, log_level + 1)
        self.model.build_vocab(tagged_reviews)
        self.model.train(
            tagged_reviews,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs,
        )
        logger.log("Done", verbose, log_level + 1)

    def vectorize(self, doc_list: List[Document], verbose: bool = False) -> np.ndarray:
        return np.array(
            [
                self.model.infer_vector(doc.inner)
                for doc in logger.tqdm(doc_list, verbose)
            ]
        )

    def preprocess(
        self, doc_list: List[str], verbose: bool = False, min_length: int = 2
    ) -> List[Document]:
        return [
            Document(
                [w.lower() for w in nltk.word_tokenize(text) if len(w) >= min_length]
            )
            for text in logger.tqdm(doc_list, verbose)
        ]


class TfidfVectorizer(Vectorizer):
    def __init__(self, **kwargs):
        self.model = sktext.TfidfVectorizer(**kwargs)

    def train(
        self, doc_list: List[Document], verbose: bool = False, log_level: int = 0
    ):
        logger.log("Training the vectorizer", verbose, log_level + 1)
        self.model.fit_transform([doc.inner for doc in doc_list])
        logger.log("Done", verbose, log_level + 1)

    def vectorize(self, doc_list: List[Document], verbose: bool = False) -> np.ndarray:
        return self.model.transform(
            doc.inner for doc in logger.tqdm(doc_list, verbose)
        ).toarray()

    def preprocess(self, doc_list: List[str], verbose: bool = False) -> List[Document]:
        return [Document(doc) for doc in logger.tqdm(doc_list, verbose)]
