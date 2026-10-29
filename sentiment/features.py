"""Turns raw review text into the six sentiment features x1..x6."""

import math
import re
from collections.abc import Iterable

import numpy as np

from .lexicon import PRONOUNS, load_opinion_lexicon

FEATURE_NAMES = (
    "x1: positive lexicon count",
    "x2: negative lexicon count",
    "x3: contains 'no'",
    "x4: 1st/2nd-person pronoun count",
    "x5: contains '!'",
    "x6: ln(word count)",
)

# Lowercase words; internal hyphens and apostrophes are kept so that lexicon
# entries such as "well-made" and words such as "can't" stay single tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    """Remove HTML tags such as the <br /> line breaks found in IMDB reviews."""
    return _HTML_TAG_RE.sub(" ", text)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(clean_text(text).lower())


def extract_features(text: str) -> np.ndarray:
    """Return the feature vector [x1, x2, x3, x4, x5, x6] for one document."""
    positive, negative = load_opinion_lexicon()
    tokens = tokenize(text)
    return np.array(
        [
            sum(t in positive for t in tokens),
            sum(t in negative for t in tokens),
            1.0 if "no" in tokens else 0.0,
            sum(t in PRONOUNS for t in tokens),
            1.0 if "!" in clean_text(text) else 0.0,
            # Guard against ln(0) for an empty document.
            math.log(max(len(tokens), 1)),
        ],
        dtype=float,
    )


def featurize(texts: Iterable[str]) -> np.ndarray:
    """Stack feature vectors into an (n_documents, 6) matrix."""
    return np.vstack([extract_features(t) for t in texts])
