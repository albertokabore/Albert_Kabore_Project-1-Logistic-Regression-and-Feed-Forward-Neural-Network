"""Regression tests for duplicate leakage and invalid data handling.

Short text fixtures test validation only. Experiment results use the IMDB CSV.
"""

import numpy as np
import pandas as pd
import pytest

from sentiment import Standardizer, load_imdb
from sentiment.data import review_key


def test_formatting_variants_are_removed_before_splitting(tmp_path):
    path = tmp_path / "reviews.csv"
    pd.DataFrame({
        "review": ["I liked it.<br />Very good!", " i LIKED it.  Very good! ", "I disliked it."],
        "sentiment": ["positive", "positive", "negative"],
    }).to_csv(path, index=False)
    data = load_imdb(path)
    assert len(data) == 2
    assert data["source_row"].tolist() == [0, 2]
    assert data["label"].tolist() == [1, 0]
    assert data["review"].map(review_key).is_unique


@pytest.mark.parametrize("reviews,labels", [
    (["good", None], ["positive", "negative"]),
    (["good", "bad"], ["positive", "neutral"]),
    (["good", " GOOD "], ["positive", "negative"]),
    (["good", "<br />"], ["positive", "negative"]),
])
def test_invalid_or_conflicting_records_are_rejected(tmp_path, reviews, labels):
    path = tmp_path / "reviews.csv"
    pd.DataFrame({"review": reviews, "sentiment": labels}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_imdb(path)


def test_constant_features_do_not_divide_by_zero():
    data = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    scaler = Standardizer().fit(data)
    assert np.isfinite(scaler.transform(data)).all()
    np.testing.assert_array_equal(scaler.transform(data)[:, 1], 0)
