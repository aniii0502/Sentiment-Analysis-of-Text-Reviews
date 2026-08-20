import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data is available
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()


def process_and_lemmatize_review(text: str) -> str:
    """
    Full text preprocessing pipeline:
    1. Lowercase & strip HTML tags
    2. Remove non-alphabetic characters & digits
    3. Tokenize & filter English stopwords
    4. Lemmatize tokens to root form
    5. Rejoin into a single space-separated string
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase & remove HTML
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Remove punctuation, special characters, and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # 3. Tokenize (split on whitespace)
    tokens = text.split()

    # 4. Stopword filtering + Lemmatization
    lemmatized_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]

    # 5. Rejoin tokens into a single clean string
    return " ".join(lemmatized_tokens)
