import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

VADER = None


def vader_sentiment(text: str, _: int) -> float:
    global VADER

    if VADER is None:
        nltk.download("vader_lexicon", quiet=True)
        VADER = SentimentIntensityAnalyzer()

    return VADER.polarity_scores(text)["compound"]


def linear_sentiment(_: str, score: int) -> float:
    return (score - 5.5) / 4.5
