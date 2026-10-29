"""Audit saved results against source reviews and parameters, without retraining."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from artifact_checks import verify_artifacts
from sentiment import FeedForwardNetwork, featurize, load_imdb, sigmoid
from sentiment.data import review_key
from train import metrics

ROOT = Path(__file__).resolve().parent


def verify_results():
    r = json.loads((ROOT / "results/results.json").read_text(encoding="utf-8"))
    verify_artifacts(r)
    df = load_imdb(ROOT / r["dataset"]["csv_path"])
    rows = df.source_row.to_numpy()
    positions = pd.Series(np.arange(len(df)), index=rows)
    with np.load(ROOT / "results/split_indices.npz") as stored:
        parts = {name: positions.loc[stored[name]].to_numpy() for name in stored.files}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(parts.values()))), np.arange(len(df)))
    keys = df.review.map(review_key)
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not set(keys.iloc[parts[a]]) & set(keys.iloc[parts[b]])
    X = featurize(df.review)
    y = df.label.to_numpy()
    mean, std = X[parts["train"]].mean(axis=0), X[parts["train"]].std(axis=0)
    np.testing.assert_allclose(mean, r["standardizer"]["mean"], atol=1e-12)
    np.testing.assert_allclose(std, r["standardizer"]["std"], atol=1e-12)
    Z = (X - mean) / np.where(std == 0, 1.0, std)
    lr = r["logistic_regression"]
    lr_p = sigmoid(X @ np.asarray(lr["weights_raw"]) + lr["bias_raw"])
    nn = r["feed_forward_network"]
    net = FeedForwardNetwork(n_hidden=nn["hidden_units"])
    net.params = {key: np.asarray(value) for key, value in nn["params"].items()}
    nn_p = net.predict_proba(Z)
    for name, idx in parts.items():
        assert len(idx) == r["dataset"]["splits"][name]["size"]
        for model, probabilities in [(lr, lr_p), (nn, nn_p)]:
            actual = metrics(y[idx], (probabilities[idx] >= 0.5).astype(int))
            for metric, value in actual.items():
                np.testing.assert_allclose(value, model[name][metric], atol=1e-12)
    with np.load(ROOT / "results/test_predictions.npz") as predictions:
        test = parts["test"]
        np.testing.assert_array_equal(predictions["source_row"], rows[test])
        np.testing.assert_array_equal(predictions["y_true"], y[test])
        np.testing.assert_allclose(predictions["logistic_probability"], lr_p[test], atol=1e-12)
        np.testing.assert_allclose(predictions["ffnn_probability"], nn_p[test], atol=1e-12)
    for key in ["learning_example", "classification_example", "misclassified_example"]:
        example = r[key]
        row = example["row"]
        assert rows[row] == example["source_row"]
        assert row in parts[example["split"]]
        assert df.loc[row, "review"] == example["text"]
        assert int(y[row]) == example["y"]
        np.testing.assert_allclose(X[row], example["x_raw"], atol=1e-12)
    ex = r["learning_example"]
    x_std = np.asarray(ex["x_std"])
    np.testing.assert_allclose(x_std, Z[ex["row"]], atol=1e-12)
    np.testing.assert_allclose(ex["gradient_w"], (0.5-ex["y"]) * x_std, atol=1e-12)
    np.testing.assert_allclose(ex["weights_after"], -0.5*np.asarray(ex["gradient_w"]), atol=1e-12)
    np.testing.assert_allclose(sigmoid(x_std @ ex["weights_after"] + ex["bias_after"]), ex["p_after"], atol=1e-12)
    nn_step = ex["ffnn_step"]
    for name in nn_step["params_before"]:
        np.testing.assert_allclose(np.asarray(nn_step["params_before"][name]) -
                                   0.5*np.asarray(nn_step["gradients"][name]),
                                   nn_step["params_after"][name], atol=1e-12)
    new = r["classification_example"]
    np.testing.assert_allclose(new["p_positive"], lr_p[new["row"]], atol=1e-12)
    np.testing.assert_allclose(new["ffnn_p_positive"], nn_p[new["row"]], atol=1e-12)
    summary = {
        "passed": True, "retained_reviews": len(df), "audited_test_predictions": len(parts["test"]),
        "checks": ["Data/source/artifact hashes", "Split completeness and normalized-text disjointness",
                   "Training-only standardization", "Both models' metrics on every split",
                   "Every saved test probability and label", "Worked-example source rows and updates"],
    }
    (ROOT / "results/verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    verify_results()
