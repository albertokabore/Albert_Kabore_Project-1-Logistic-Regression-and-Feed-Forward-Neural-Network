import math

import numpy as np
import pytest

from sentiment import clean_text, extract_features, featurize, load_opinion_lexicon, tokenize


def test_lexicon_is_the_full_hu_liu_list():
    positive, negative = load_opinion_lexicon()
    assert len(positive) == 2006 and len(negative) == 4783
    assert {"love", "excellent"} <= positive and {"boring", "awful"} <= negative


def test_tokenize_lowercases_and_keeps_contractions_and_hyphens():
    assert tokenize("I can't STOP! A well-made film.") == ["i", "can't", "stop", "a", "well-made", "film"]


def test_html_line_breaks_are_removed():
    assert clean_text("Great.<br /><br />Really.") == "Great.  Really."
    assert "br" not in tokenize("Great.<br /><br />Really.")


def test_extract_features_counts_every_feature():
    text = "No way! I loved it and you will love it too, but the ending was boring."
    positive, negative = load_opinion_lexicon()
    tokens = tokenize(text)
    x = extract_features(text)
    np.testing.assert_allclose(
        x,
        [
            sum(t in positive for t in tokens),
            sum(t in negative for t in tokens),
            1,
            2,  # I, you
            1,
            math.log(len(tokens)),
        ],
    )
    assert x[0] >= 2  # loved, love
    assert x[1] >= 1  # boring


def test_no_feature_matches_whole_word_only():
    # "nothing" and "know" contain "no" but must not trigger x3.
    assert extract_features("I know nothing about it")[2] == 0


def test_empty_document_does_not_crash():
    np.testing.assert_allclose(extract_features(""), np.zeros(6))


@pytest.mark.parametrize("texts", [["good"], ["good", "bad", "no!"]])
def test_featurize_shape(texts):
    assert featurize(texts).shape == (len(texts), 6)
