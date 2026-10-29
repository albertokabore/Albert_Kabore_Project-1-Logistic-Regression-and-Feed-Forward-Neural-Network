"""Reproduce the experiment from real reviews: python train.py.

No generated reviews or generated labels are used. Randomness is limited to the
stratified split and neural-network initialization, with seed 42 for both.
"""

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sentiment import (
    FEATURE_NAMES, PRONOUNS, FeedForwardNetwork, LogisticRegression, Standardizer,
    download_imdb, extract_features, featurize, load_imdb, load_opinion_lexicon,
    sigmoid, stratified_split, tokenize,
)
from sentiment.data import KAGGLE_DATASET, review_key

ROOT = Path(__file__).resolve().parent
SEED = 42


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, predictions):
    tp = int(np.sum((predictions == 1) & (y == 1)))
    tn = int(np.sum((predictions == 0) & (y == 0)))
    fp = int(np.sum((predictions == 1) & (y == 0)))
    fn = int(np.sum((predictions == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def describe_review(df, row, X, split):
    positive, negative = load_opinion_lexicon()
    text = df.loc[row, "review"]
    tokens = tokenize(text)
    return {
        "row": int(row), "source_row": int(df.loc[row, "source_row"]), "split": split,
        "text": text, "label": df.loc[row, "sentiment"], "y": int(df.loc[row, "label"]),
        "x_raw": X[row].tolist(), "token_count": len(tokens),
        "positive_words": [t for t in tokens if t in positive],
        "negative_words": [t for t in tokens if t in negative],
        "pronouns": [t for t in tokens if t in PRONOUNS],
    }


def save_curve(history, name, title, out):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(history, color="#225f9c", linewidth=1.7)
    ax.set(xlabel="Epoch (full-batch updates)", ylabel="Mean cross-entropy", title=title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "figures" / name, dpi=200)
    plt.close(fig)


def run_experiment(output_dir=None):
    """Train models, evaluate fixed choices, and write traceable result artifacts."""
    out = Path(output_dir) if output_dir else ROOT / "results"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    csv_path = download_imdb()
    raw = pd.read_csv(csv_path)
    df = load_imdb(csv_path)
    print(f"Validated {len(raw):,} source rows; retained {len(df):,} distinct normalized reviews.", flush=True)
    X = featurize(df["review"])
    y = df["label"].to_numpy()
    train, val, test = stratified_split(y, (0.7, 0.15, 0.15), SEED)
    parts = {"train": train, "validation": val, "test": test}
    keys = df["review"].map(review_key)
    overlap = {}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap[f"{a}_{b}"] = len(set(keys.iloc[parts[a]]) & set(keys.iloc[parts[b]]))
    assert not any(overlap.values()), "Normalized review text crosses splits"
    assert len(np.unique(np.concatenate(list(parts.values())))) == len(df)
    scaler = Standardizer().fit(X[train])
    Z = scaler.transform(X)
    lr = LogisticRegression(learning_rate=0.5).fit(Z[train], y[train])
    grad_w, grad_b = lr.gradient(Z[train], y[train])
    gradient_norm = float(np.linalg.norm(np.append(grad_w, grad_b)))
    assert gradient_norm < lr.tol, "Logistic regression did not converge"
    assert np.all(np.diff(lr.loss_history) <= 1e-12)
    w_raw, b_raw = scaler.to_raw_weights(lr.w, lr.b)
    np.testing.assert_allclose(sigmoid(X @ w_raw + b_raw), lr.predict_proba(Z), atol=1e-12)
    A = np.column_stack([Z[train], np.ones(len(train))])
    theta = np.zeros(7)
    for _ in range(25):
        p = sigmoid(A @ theta)
        g = A.T @ (p - y[train]) / len(train)
        hessian = A.T @ (A * (p * (1 - p))[:, None]) / len(train)
        if np.linalg.norm(g) < 1e-12:
            break
        theta -= np.linalg.solve(hessian, g)
    newton_difference = float(np.max(np.abs(theta - np.append(lr.w, lr.b))))
    assert newton_difference < 1e-4
    p = sigmoid(A @ theta)
    hessian = A.T @ (A * (p * (1 - p))[:, None]) / len(train)
    smoothness_bound = float(0.25 * np.linalg.eigvalsh(A.T @ A / len(train))[-1])
    replay = LogisticRegression(learning_rate=0.5)
    replay.init_params(6)
    epochs_run = len(lr.loss_history) - 1
    trace = []
    for epoch in range(epochs_run + 1):
        if epoch in {0, 1, 5, 10, 50, 100, epochs_run}:
            trace.append({"epoch": epoch, "loss": replay.loss(Z[train], y[train]),
                          "accuracy": metrics(y[train], replay.predict(Z[train]))["accuracy"],
                          "weights": replay.w.tolist(), "bias": replay.b})
        if epoch < epochs_run:
            replay.step(Z[train], y[train])
    print(f"Logistic regression: {epochs_run} updates, gradient norm {gradient_norm:.3g}.", flush=True)
    candidates, selection = {}, []
    for hidden in (2, 4, 8):
        start = time.perf_counter()
        net = FeedForwardNetwork(n_hidden=hidden, epochs=2000, learning_rate=0.5, seed=SEED)
        net.fit(Z[train], y[train])
        candidates[hidden] = net
        selection.append({"hidden_units": hidden, "train_loss": net.loss_history[-1],
                          "validation_loss": net.loss(Z[val], y[val]),
                          "train_accuracy": metrics(y[train], net.predict(Z[train]))["accuracy"],
                          "validation_accuracy": metrics(y[val], net.predict(Z[val]))["accuracy"],
                          "seconds": time.perf_counter() - start})
        print(f"FFNN H={hidden}: validation loss {selection[-1]['validation_loss']:.6f}.", flush=True)
    best_h = min(selection, key=lambda row: row["validation_loss"])["hidden_units"]
    ffnn = candidates[best_h]
    # Test labels are used after the architecture has been fixed by validation loss.
    lr_probability, nn_probability = lr.predict_proba(Z[test]), ffnn.predict_proba(Z[test])
    lr_pred, nn_pred = (lr_probability >= 0.5).astype(int), (nn_probability >= 0.5).astype(int)
    baseline_label = int(y[train].mean() >= 0.5)
    baseline = metrics(y[test], np.full(len(test), baseline_label))
    example_idx = next(i for i in train if len(tokenize(df.loc[i, "review"])) <= 60)
    example = describe_review(df, example_idx, X, "train")
    xb, yb = Z[example_idx][None], y[example_idx:example_idx + 1]
    one_step = LogisticRegression(learning_rate=0.5)
    one_step.init_params(6)
    before_loss = one_step.loss(xb, yb)
    gw, gb = one_step.step(xb, yb)
    example.update({"x_std": xb[0].tolist(), "p_before": 0.5,
                    "p_after": float(one_step.predict_proba(xb)[0]),
                    "loss_before": before_loss, "loss_after": one_step.loss(xb, yb),
                    "gradient_w": gw.tolist(), "gradient_b": gb,
                    "weights_after": one_step.w.tolist(), "bias_after": one_step.b})
    # A separate illustrative update shows actual hidden-layer backpropagation.
    demo = FeedForwardNetwork(n_hidden=best_h, learning_rate=0.5, seed=SEED)
    demo.init_params(6)
    nn_before = {k: v.copy() for k, v in demo.params.items()}
    nn_p_before = float(demo.predict_proba(xb)[0])
    nn_loss_before = demo.loss(xb, yb)
    nn_grads = demo.step(xb, yb)
    example["ffnn_step"] = {
        "p_before": nn_p_before, "p_after": float(demo.predict_proba(xb)[0]),
        "loss_before": nn_loss_before, "loss_after": demo.loss(xb, yb),
        "params_before": {k: v.tolist() for k, v in nn_before.items()},
        "gradients": {k: v.tolist() for k, v in nn_grads.items()},
        "params_after": {k: v.tolist() for k, v in demo.params.items()},
    }
    new_idx = next(i for i in test if len(tokenize(df.loc[i, "review"])) <= 80)
    classification = describe_review(df, new_idx, X, "test")
    z_new = float(w_raw @ X[new_idx] + b_raw)
    classification.update({"contributions": (w_raw * X[new_idx]).tolist(), "z": z_new,
                           "p_positive": float(sigmoid(z_new)),
                           "ffnn_p_positive": float(ffnn.predict_proba(Z[new_idx][None])[0])})
    wrong_idx = next(i for i, pred in zip(test, lr_pred)
                     if pred != y[i] and len(tokenize(df.loc[i, "review"])) <= 80)
    wrong = describe_review(df, wrong_idx, X, "test")
    wrong["p_positive"] = float(lr.predict_proba(Z[wrong_idx][None])[0])
    positive, negative = load_opinion_lexicon()
    lexicon_hash = hashlib.sha256(json.dumps([sorted(positive), sorted(negative)]).encode()).hexdigest()
    source_files = [Path(__file__), *sorted((ROOT / "sentiment").glob("*.py"))]
    result = {
        "configuration": {"seed": SEED, "learning_rate": 0.5, "logistic_tolerance": lr.tol,
                          "ffnn_epochs": 2000, "hidden_candidates": [2, 4, 8],
                          "regularization": 0.0, "threshold": 0.5,
                          "initialization": "Gaussian, variance 1/fan_in; zero biases"},
        "provenance": {"python": platform.python_version(),
                       "packages": {name: importlib.metadata.version(name)
                                    for name in ["numpy", "pandas", "nltk", "matplotlib", "kagglehub"]},
                       "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
                       "lexicon_sorted_word_sets_sha256": lexicon_hash},
        "dataset": {"source": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
                    "csv_path": "data/IMDB Dataset.csv", "csv_sha256": sha256(csv_path),
                    "rows_in_file": len(raw), "raw_label_counts": raw.sentiment.value_counts().to_dict(),
                    "missing_values": int(raw[["review", "sentiment"]].isna().sum().sum()),
                    "exact_duplicates_removed": int(raw.review.duplicated().sum()),
                    "normalized_duplicates_removed": len(raw) - len(df), "retained_rows": len(df),
                    "retained_label_counts": df.sentiment.value_counts().to_dict(),
                    "normalized_text_overlap": overlap, "conflicting_normalized_labels": 0,
                    "splits": {name: {"size": len(idx), "positive_fraction": float(y[idx].mean())}
                               for name, idx in parts.items()},
                    "lexicon_sizes": {"positive": len(positive), "negative": len(negative)}},
        "features": {"names": list(FEATURE_NAMES),
                     "mean_by_class": {str(label): X[y == label].mean(axis=0).tolist() for label in (0, 1)},
                     "maximum": X.max(axis=0).tolist()},
        "standardizer": {"mean": scaler.mean.tolist(), "std": scaler.scale.tolist()},
        "logistic_regression": {"epochs_run": epochs_run, "final_train_loss": lr.loss_history[-1],
                                "gradient_norm": gradient_norm, "weights_standardized": lr.w.tolist(),
                                "bias_standardized": lr.b, "weights_raw": w_raw.tolist(), "bias_raw": b_raw,
                                "newton_max_abs_difference": newton_difference,
                                "newton_weights_and_bias": theta.tolist(),
                                "newton_gradient_norm": float(np.linalg.norm(A.T @ (p - y[train]) / len(train))),
                                "augmented_design_rank": int(np.linalg.matrix_rank(A)),
                                "hessian_min_eigenvalue": float(np.linalg.eigvalsh(hessian)[0]),
                                "gradient_lipschitz_upper_bound": smoothness_bound,
                                "training_loss_monotone": bool(np.all(np.diff(lr.loss_history) <= 1e-12)),
                                "train": metrics(y[train], lr.predict(Z[train])),
                                "validation": metrics(y[val], lr.predict(Z[val])),
                                "test": metrics(y[test], lr_pred), "trace": trace},
        "feed_forward_network": {"hidden_units": best_h, "selection": selection,
                                 "params": {k: v.tolist() for k, v in ffnn.params.items()},
                                 "train": metrics(y[train], ffnn.predict(Z[train])),
                                 "validation": metrics(y[val], ffnn.predict(Z[val])),
                                 "test": metrics(y[test], nn_pred)},
        "baseline": {"label_chosen_from_training": baseline_label, "test": baseline},
        "learning_example": example, "classification_example": classification,
        "misclassified_example": wrong,
    }
    split_file = out / "split_indices.npz"
    np.savez_compressed(split_file, **{name: df.iloc[idx].source_row.to_numpy() for name, idx in parts.items()})
    prediction_file = out / "test_predictions.npz"
    np.savez_compressed(prediction_file, source_row=df.iloc[test].source_row.to_numpy(),
                        y_true=y[test], logistic_probability=lr_probability, ffnn_probability=nn_probability)
    result["provenance"]["artifact_sha256"] = {
        "split_indices.npz": sha256(split_file), "test_predictions.npz": sha256(prediction_file)}
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_curve(lr.loss_history, "lr_training_loss.png", "Logistic regression: training loss", out)
    save_curve(ffnn.loss_history, "ffnn_training_loss.png", f"Feed-forward network (H={best_h}): training loss", out)
    print(f"Saved {out / 'results.json'}", flush=True)
    return result


if __name__ == "__main__":
    run_experiment()
