"""Sentiment lexicons and the pronoun list used by the feature extractor.

The positive and negative word lists are the opinion lexicon of Hu & Liu (2004),
loaded from the copy distributed with NLTK (corpus id "opinion_lexicon").
It is downloaded automatically the first time it is needed.
"""

from functools import lru_cache

# First-person (I, we) and second-person (you) pronouns in all their forms.
PRONOUNS = frozenset(
    """
    i me my mine myself we us our ours ourselves you your yours yourself yourselves
    """.split()
)


@lru_cache(maxsize=1)
def load_opinion_lexicon() -> tuple[frozenset[str], frozenset[str]]:
    """Return (positive_words, negative_words) from the Hu & Liu opinion lexicon."""
    import nltk
    from nltk.corpus import opinion_lexicon

    try:
        positive, negative = opinion_lexicon.positive(), opinion_lexicon.negative()
    except LookupError:
        if not nltk.download("opinion_lexicon", quiet=True):
            raise RuntimeError("Could not download the NLTK 'opinion_lexicon' corpus.")
        positive, negative = opinion_lexicon.positive(), opinion_lexicon.negative()
    return frozenset(positive), frozenset(negative)
