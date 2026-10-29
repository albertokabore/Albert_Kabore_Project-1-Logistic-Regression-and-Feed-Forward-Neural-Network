"""Downloading and loading the Kaggle "IMDB Dataset of 50K Movie Reviews".

Kaggle page: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
The reviews originate from the Large Movie Review Dataset of Maas et al. (2011),
http://ai.stanford.edu/~amaas/data/sentiment/
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .features import clean_text

KAGGLE_DATASET = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
CSV_NAME = "IMDB Dataset.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_imdb() -> Path:
    """Return the path of the project copy data/IMDB Dataset.csv.

    If the copy does not exist yet, the dataset is downloaded from Kaggle with
    kagglehub and the CSV is copied into the project's data/ folder.
    """
    local_csv = DATA_DIR / CSV_NAME
    if not local_csv.exists():
        import kagglehub

        DATA_DIR.mkdir(exist_ok=True)
        shutil.copyfile(Path(kagglehub.dataset_download(KAGGLE_DATASET)) / CSV_NAME, local_csv)
    return local_csv


def review_key(text: str) -> str:
    """Ignore HTML tags, letter case and whitespace when detecting duplicates."""
    return " ".join(clean_text(text).lower().split())


def load_imdb(csv_path: Path | None = None) -> pd.DataFrame:
    """Validate labels and deduplicate normalized review texts before splitting.

    Retain the CSV row number for tracing worked examples to the original file.
    Reject conflicting labels instead of silently choosing one.
    """
    df = pd.read_csv(csv_path or download_imdb())
    if not {"review", "sentiment"} <= set(df.columns):
        raise ValueError("CSV must have review and sentiment columns")
    if df.empty or df[["review", "sentiment"]].isna().any().any():
        raise ValueError("Reviews and labels must be nonempty and nonmissing")
    if not df["review"].map(lambda text: isinstance(text, str) and bool(review_key(text))).all():
        raise ValueError("Each review must contain text")
    if not df["sentiment"].isin(["positive", "negative"]).all():
        raise ValueError("Sentiment labels must be positive or negative")
    df["source_row"] = np.arange(len(df))
    keys = df["review"].map(review_key)
    if (df.groupby(keys)["sentiment"].nunique() > 1).any():
        raise ValueError("Duplicate reviews have conflicting sentiment labels")
    df = df.loc[~keys.duplicated()].reset_index(drop=True)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    return df


def stratified_split(
    y: np.ndarray, fractions: tuple[float, ...], seed: int
) -> list[np.ndarray]:
    """Split indices into len(fractions) parts, keeping the class balance in every part.

    `fractions` must sum to 1, e.g. (0.70, 0.15, 0.15) for train / validation / test.
    """
    if not fractions or any(f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must be positive and sum to 1")
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[] for _ in fractions]
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        bounds = np.round(np.cumsum(fractions) * len(idx)).astype(int)
        for part, chunk in zip(parts, np.split(idx, bounds[:-1])):
            part.extend(chunk)
    return [np.sort(np.array(p, dtype=int)) for p in parts]
