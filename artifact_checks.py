"""Dependency-free checksum checks shared by the notebook and result auditor."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify_artifacts(result):
    """Refuse to use results against changed data, code or numerical artifacts."""
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(ROOT / result["dataset"]["csv_path"]) != result["dataset"]["csv_sha256"]:
        raise ValueError("Dataset changed: rerun python train.py")
    for name, expected in result["provenance"]["artifact_sha256"].items():
        if digest(ROOT / "results" / name) != expected:
            raise ValueError(f"Numerical artifact changed: {name}")
    for name, expected in result["provenance"]["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Source changed since training: {name}")
    cached = (Path.home() / ".cache/kagglehub/datasets/lakshmi25npathi/"
              "imdb-dataset-of-50k-movie-reviews/versions/1/IMDB Dataset.csv")
    if cached.exists() and digest(cached) != result["dataset"]["csv_sha256"]:
        raise ValueError("Local dataset differs from the cached Kaggle version 1")
