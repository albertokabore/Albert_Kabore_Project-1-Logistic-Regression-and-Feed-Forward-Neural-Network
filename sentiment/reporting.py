"""Build the written submission from computed results, without hand-entered scores."""

from pathlib import Path

import numpy as np


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ])


def build_report(r, destination):
    d, lr, nn = r["dataset"], r["logistic_regression"], r["feed_forward_network"]
    ex, new, wrong = r["learning_example"], r["classification_example"], r["misclassified_example"]
    feature_labels = ["Positive words", "Negative words", "Contains no", "Pronouns", "Contains !", "ln(word count)"]
    pct = lambda value: f"{100 * value:.2f}%"
    vector = lambda values: "(" + ", ".join(f"{v:.6f}" for v in values) + ")"
    raw_weights = lr["weights_raw"]
    weights = table(["Parameter", "Feature", "Standardized", "Raw scale"], [
        [f"w{j+1}", label, f"{lr['weights_standardized'][j]:.6f}", f"{raw_weights[j]:.6f}"]
        for j, label in enumerate(feature_labels)
    ] + [["b", "Bias", f"{lr['bias_standardized']:.6f}", f"{lr['bias_raw']:.6f}"]])
    scores = table(["Model / split", "Accuracy", "Precision", "Recall", "F1"], [
        [f"{name} / {split}", pct(model[split]["accuracy"]),
         *[f"{model[split][metric]:.4f}" for metric in ["precision", "recall", "F1"]]]
        for name, model in [("Logistic regression", lr), ("Feed-forward network", nn)]
        for split in ["train", "validation", "test"]
    ])
    selection = table(["Hidden units", "Train loss", "Validation loss", "Train accuracy", "Validation accuracy"], [
        [v["hidden_units"], f"{v['train_loss']:.6f}", f"{v['validation_loss']:.6f}",
         pct(v["train_accuracy"]), pct(v["validation_accuracy"])] for v in nn["selection"]])
    params = nn["params"]
    parameters = table(["Unit", "W1:1", "W1:2", "W1:3", "W1:4", "W1:5", "W1:6", "b1", "w2"], [
        [k+1, *[f"{v:.4f}" for v in params["W1"][k]], f"{params['b1'][k]:.4f}", f"{params['w2'][k]:.4f}"]
        for k in range(nn["hidden_units"])])
    feature_means = table(["Feature", "Negative mean", "Positive mean"], [
        [label, f"{r['features']['mean_by_class']['0'][j]:.4f}", f"{r['features']['mean_by_class']['1'][j]:.4f}"]
        for j, label in enumerate(feature_labels)])
    standardizer = table(["Statistic", *[f"x{j+1}" for j in range(6)]], [
        ["Training mean", *[f"{v:.5f}" for v in r["standardizer"]["mean"]]],
        ["Training std", *[f"{v:.5f}" for v in r["standardizer"]["std"]]],
    ])
    # All printed example values originate from actual CSV rows and model updates.
    step_table = table(["Feature", "Raw", "Standardized", "Gradient", "Weight after update"], [
        [label, f"{ex['x_raw'][j]:.4f}", f"{ex['x_std'][j]:.4f}",
         f"{ex['gradient_w'][j]:.4f}", f"{ex['weights_after'][j]:.4f}"]
        for j, label in enumerate(feature_labels)])
    contribution_table = table(["Feature", "Raw value", "Raw weight", "Contribution"], [
        [label, f"{new['x_raw'][j]:.6f}", f"{raw_weights[j]:.6f}", f"{new['contributions'][j]:.6f}"]
        for j, label in enumerate(feature_labels)
    ] + [["Bias", "1", f"{lr['bias_raw']:.6f}", f"{lr['bias_raw']:.6f}"],
         ["Total logit", "", "", f"{new['z']:.6f}"]])
    trace_table = table(["Epoch", "Train loss", "Train accuracy", "w1", "w2", "b"], [
        [v["epoch"], f"{v['loss']:.6f}", pct(v["accuracy"]),
         f"{v['weights'][0]:.5f}", f"{v['weights'][1]:.5f}", f"{v['bias']:.5f}"] for v in lr["trace"]])
    nn_step = ex["ffnn_step"]
    nn_step_table = table(["Parameter", "Before", "Gradient", "After"], [
        [name, f"{np.asarray(nn_step['params_before'][name]).flat[0]:.6f}",
         f"{np.asarray(nn_step['gradients'][name]).flat[0]:.6f}",
         f"{np.asarray(nn_step['params_after'][name]).flat[0]:.6f}"] for name in ["W1", "b1", "w2", "b2"]])
    split_table = table(["Part", "Reviews", "Positive", "Purpose"], [
        [name, f"{item['size']:,}", pct(item["positive_fraction"]), purpose]
        for (name, item), purpose in zip(d["splits"].items(), ["Fit weights", "Choose hidden width", "Evaluate fixed models"])])
    confusion = table(["Model", "True positive", "True negative", "False positive", "False negative"], [
        [name, *[model["test"][key] for key in ["TP", "TN", "FP", "FN"]]]
        for name, model in [("Logistic regression", lr), ("Feed-forward network", nn)]])
    source_row = lambda row: f"CSV data row {row['source_row']} (zero based; physical CSV lines can span multiple lines)"
    values = {
        "retained": f"{d['retained_rows']:,}", "train_size": f"{d['splits']['train']['size']:,}",
        "dataset_hash": d["csv_sha256"], "duplicates": d["normalized_duplicates_removed"],
        "extra_duplicates": d["normalized_duplicates_removed"]-d["exact_duplicates_removed"],
        "splits": split_table, "class_counts": str(d["retained_label_counts"]),
        "feature_means": feature_means, "standardizer": standardizer, "weights": weights,
        "epochs": lr["epochs_run"], "loss": f"{lr['final_train_loss']:.8f}",
        "gradient": f"{lr['gradient_norm']:.3e}", "newton_diff": f"{lr['newton_max_abs_difference']:.3e}",
        "newton_gradient": f"{lr['newton_gradient_norm']:.3e}",
        "rank": lr["augmented_design_rank"], "hessian_eigen": f"{lr['hessian_min_eigenvalue']:.6f}",
        "lipschitz": f"{lr['gradient_lipschitz_upper_bound']:.6f}",
        "step_bound": f"{1/lr['gradient_lipschitz_upper_bound']:.6f}",
        "raw_vector": vector(raw_weights), "raw_bias": f"{lr['bias_raw']:.6f}",
        "h": nn["hidden_units"], "parameter_count": 8*nn["hidden_units"]+1,
        "selection": selection, "parameters": parameters, "output_bias": f"{params['b2'][0]:.6f}",
        "scores": scores, "confusion": confusion, "baseline": pct(r["baseline"]["test"]["accuracy"]),
        "lr_accuracy": pct(lr["test"]["accuracy"]), "nn_accuracy": pct(nn["test"]["accuracy"]),
        "gap": f"{100*(nn['test']['accuracy']-lr['test']['accuracy']):.2f}",
        "trace": trace_table, "example_source": source_row(ex), "example_text": ex["text"],
        "example_label": ex["label"], "example_y": ex["y"], "example_count": ex["token_count"],
        "example_positive": ", ".join(ex["positive_words"]), "example_negative": ", ".join(ex["negative_words"]),
        "step_table": step_table, "step_bias": f"{ex['bias_after']:.6f}",
        "step_p": f"{ex['p_after']:.6f}", "step_loss": f"{ex['loss_after']:.6f}",
        "nn_step_table": nn_step_table, "nn_before_p": f"{nn_step['p_before']:.6f}",
        "nn_after_p": f"{nn_step['p_after']:.6f}", "nn_before_loss": f"{nn_step['loss_before']:.6f}",
        "nn_after_loss": f"{nn_step['loss_after']:.6f}",
        "new_source": source_row(new), "new_text": new["text"], "new_label": new["label"],
        "new_count": new["token_count"], "new_positive": ", ".join(new["positive_words"]),
        "new_negative": ", ".join(new["negative_words"]), "contributions": contribution_table,
        "new_z": f"{new['z']:.6f}", "new_p": f"{new['p_positive']:.6f}",
        "new_prediction": "positive" if new["p_positive"]>=0.5 else "negative",
        "nn_new_p": f"{new['ffnn_p_positive']:.6f}",
        "nn_new_prediction": "positive" if new["ffnn_p_positive"]>=0.5 else "negative",
        "wrong_text": wrong["text"], "wrong_source": source_row(wrong), "wrong_label": wrong["label"],
        "wrong_p": f"{wrong['p_positive']:.6f}",
        "packages": ", ".join(f"{k} {v}" for k,v in r["provenance"]["packages"].items()),
    }
    template = r'''# Project 1: Logistic Regression and Feed-Forward Neural Network

**Author: Albert Kabore**

## 1. Aim and main findings

The task is to learn weights W and bias B that predict positive or negative sentiment from exactly six specified text features. Here, lowercase w and b denote those same parameters. Logistic regression and a feed-forward network are implemented in NumPy, trained on labelled IMDB reviews, and evaluated on held-out reviews. All reported scores and worked-example calculations come from `train.py` and `results/results.json`.

Logistic regression achieved **@@lr_accuracy@@** test accuracy; the selected network achieved **@@nn_accuracy@@**. The baseline, predicting the training set's majority label for every test review, achieved **@@baseline@@**. These are results for this dataset and split, not universal performance guarantees.

## 2. Dataset, provenance and evaluation design

The input is [IMDB Dataset of 50K Movie Reviews, published by Lakshmipathi N on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), associated with the Large Movie Review Dataset of [Maas et al. (2011)](https://aclanthology.org/P11-1015/). The supplied CSV contains 50,000 rows: 25,000 positive and 25,000 negative, with no missing review texts or labels. Labels are encoded as positive = 1 and negative = 0. Review texts and labels are used as supplied; no reviews or labels are generated for training or evaluation. Published labels are not independently re-annotated in this project.

The input file is `data/IMDB Dataset.csv`. Its SHA-256 is:

```text
@@dataset_hash@@
```

The local file matches the copy in the Kagglehub dataset-version-1 cache. The downloader uses the official `kagglehub` client if the local CSV is absent. The checksum identifies the exact file used; the code does not infer authenticity solely from a filename.

**Duplicate handling.** There are 418 excess exact duplicate rows. Removing HTML tags, lowercasing, and collapsing whitespace identifies @@extra_duplicates@@ further excess duplicate texts. Removing all @@duplicates@@ duplicates before splitting leaves **@@retained@@ reviews**; retained class counts are `@@class_counts@@`. The loader rejects missing texts, unknown labels, and conflicting labels on equivalent texts. It preserves the original CSV row number for each retained review. The raw CSV is not modified.

**Splitting.** Seed 42 produces a stratified 70% / 15% / 15% split, with integer sizes rounded within each class:

@@splits@@

All three pairs of splits have zero overlap under the normalized-text rule. This checks exact and formatting duplicates; it does not establish the absence of paraphrases, shared movies or authors. This CSV experiment uses a new random split, rather than Stanford's official benchmark split, so the scores should not be compared directly with its published benchmark scores.

Only training reviews determine scaling statistics and parameter updates. The neural-network width is selected by validation cross-entropy, then the fixed models are evaluated on test reviews. This submission recomputes results after correcting duplicate handling in an earlier evaluated version; it does not claim the test set has never been inspected. Subsequent feature or model development should use validation data and obtain fresh test data for a new confirmation.

## 3. The six required features

| Symbol | Feature | Exact implementation |
| --- | --- | --- |
| x1 | Positive lexicon words | Count every token occurrence in the positive word set |
| x2 | Negative lexicon words | Count every token occurrence in the negative word set |
| x3 | Contains no | 1 when the whole lowercase token `no` occurs, otherwise 0 |
| x4 | First- and second-person pronouns | Count exact tokens from the pronoun list below |
| x5 | Contains ! | 1 when `!` appears in text after HTML removal, otherwise 0 |
| x6 | Log word count | Natural logarithm of token count, with the empty-text convention below |

The opinion word lists are the Hu and Liu lexicon distributed by NLTK: 2,006 positive and 4,783 negative entries. Tokens use `[a-z0-9]+(?:[-'][a-z0-9]+)*` after HTML removal and lowercasing. Hyphenated words and apostrophe contractions remain single tokens. There is no stemming or learned vocabulary. Thus `no` differs from `know`, `nothing`, and `no-good`. Each repeated occurrence counts again.

The pronoun list is `i me my mine myself we us our ours ourselves you your yours yourself yourselves`. Matching is exact: a contraction such as `I'm` is one token and is not counted as standalone `I`. This explicit convention makes the feature reproducible. A different contraction policy would require retraining and reporting new weights.

For an empty document, x6 is defined as ln(max(token count, 1)) = 0 to avoid ln(0). For every nonempty review, it is exactly ln(token count). No seventh feature is added.

Descriptive means across the retained corpus (not used to fit scaling):

@@feature_means@@

Training uses standardized features:

$$\tilde{x}_j = \frac{x_j-\mu_j}{s_j}$$

Here mu is the training mean and s is the training population standard deviation, calculated with `ddof=0`. A zero standard deviation is replaced by 1 to prevent division by zero; that feature is then zero on training data. Scaling can improve numerical conditioning. With unregularized logistic regression, an invertible scaling preserves the set of possible fitted logits.

@@standardizer@@

## 4. Logistic regression: equations and notation

The formulation follows the logistic-regression chapter of [Jurafsky and Martin (2026)](https://web.stanford.edu/~jurafsky/slp3/4.pdf).

| Symbol | Plain-English meaning |
| --- | --- |
| m | Number of training reviews: @@train_size@@ |
| i, j | Review index and feature index; j runs from 1 to 6 |
| x, tilde x | Raw six-feature vector and its standardized version |
| y | Supplied target label: positive = 1, negative = 0 |
| w, b | Six learned coefficients and one learned bias |
| z | Weighted score, also called the logit |
| sigma | Logistic sigmoid function |
| p | Model-estimated probability of the positive class |
| L | Average training cross-entropy loss |
| eta | Learning rate, 0.5 in this experiment |
| gradient | Vector of partial derivatives, indicating how loss changes with each parameter |

### 4.1 Prediction and objective

$$z^{(i)} = \sum_{j=1}^{6}w_j\tilde{x}^{(i)}_j+b$$

$$p^{(i)} = \sigma(z^{(i)}) = \frac{1}{1+e^{-z^{(i)}}}$$

Predict positive when p is at least 0.5, otherwise negative. The sigmoid gives a model probability estimate; calibration is not established by using a sigmoid alone.

For one review:

$$\ell(p,y) = -y\ln(p)-(1-y)\ln(1-p)$$

The desired parameters minimize the mean training loss:

$$L(w,b) = \frac{1}{m}\sum_{i=1}^{m}\ell(p^{(i)},y^{(i)})$$

The best weights mean a numerical approximation to this training-loss minimizer. They do not mean weights that maximize test accuracy. No L2 penalty is used in the reported experiment.

### 4.2 Derivatives and parameter updates

Combining the sigmoid derivative with cross-entropy gives:

$$\frac{\partial\ell}{\partial z}=p-y$$

$$\frac{\partial L}{\partial w_j}=\frac{1}{m}\sum_{i=1}^{m}(p^{(i)}-y^{(i)})\tilde{x}^{(i)}_j$$

$$\frac{\partial L}{\partial b}=\frac{1}{m}\sum_{i=1}^{m}(p^{(i)}-y^{(i)})$$

For the m-by-6 standardized training matrix Z, predictions p and labels y are length-m vectors:

$$g_w=Z^T(p-y)/m$$

$$w\leftarrow w-\eta g_w,\qquad b\leftarrow b-\eta g_b$$

Initialize all seven parameters to zero. Stop when the joint gradient norm is below 0.000001, with a maximum of 100,000 updates. The stopping condition is checked before each update. The program asserts that the requested tolerance was reached.

### 4.3 Why the optimization is justified

For the augmented design A = [Z, 1], the Hessian is:

$$\nabla^2 L=A^TDA/m,\qquad D_{ii}=p^{(i)}(1-p^{(i)})$$

This matrix is positive semidefinite, so the objective is convex. Full column rank of A gives strict convexity at finite parameter values. A finite unregularized optimum also requires the absence of complete or quasi separation; convexity alone does not guarantee a finite or unique solution. These distinctions follow the convex-optimization framework of Boyd and Vandenberghe (2004).

Here A has measured rank **@@rank@@ of 7**. Newton's method yields a finite numerical solution whose Hessian has minimum eigenvalue **@@hessian_eigen@@** and gradient norm **@@newton_gradient@@**. Gradient descent agrees within **@@newton_diff@@** in maximum absolute parameter difference. These are numerical checks supporting a well-resolved solution, not a symbolic proof of exact coefficients.

Because each diagonal entry of D is at most 1/4, one upper bound on the gradient's Lipschitz constant is one quarter of the largest eigenvalue of A-transpose A divided by m. The measured bound is **@@lipschitz@@**. The learning rate 0.5 is below its reciprocal **@@step_bound@@**, giving a sufficient descent step size. The saved training loss is also checked to be nonincreasing.

## 5. Learned logistic weights and bias

Training required **@@epochs@@ full-batch updates**. Final mean loss is **@@loss@@**, compared with ln(2) = 0.693147 at initialization; the final joint gradient norm is **@@gradient@@**.

@@weights@@

The equivalent raw-scale weights are **@@raw_vector@@**, with bias **@@raw_bias@@**. Full-precision parameters are saved in the JSON. The conversion is:

$$w^{raw}_j=w_j/s_j$$

$$b^{raw}=b-\sum_{j=1}^{6}w_j\mu_j/s_j$$

This follows by expanding the standardized weighted sum. The program verifies that raw and standardized predictions agree numerically. Holding the other features fixed, increasing one raw feature by one unit multiplies predicted positive odds by exp(its raw weight). This conditional association is not a causal effect. The two lexicon counts have the largest absolute standardized coefficients in this fit; small coefficients do not prove a feature is useless in every model.

![Logistic-regression training cross-entropy](results/figures/lr_training_loss.png)

## 6. Feed-forward neural network

### 6.1 Forward propagation and symbols

The architecture is 6 standardized inputs, H tanh hidden units, and one sigmoid output. W1 is an H-by-6 matrix, b1 is a length-H bias vector, h is a length-H activation vector, w2 is a length-H output-weight vector, and b2 is the scalar output bias. Superscript T denotes transpose.

$$h=\tanh(W_1\tilde{x}+b_1)$$

$$z=w_2^Th+b_2,\qquad p=\sigma(z)$$

The objective is the same average cross-entropy. For a single example, the chain rule gives the following derivatives; batch gradients average these values across training reviews. The symbol odot means element-by-element multiplication.

$$\delta_o=p-y,\qquad g_{w_2}=\delta_o h,\qquad g_{b_2}=\delta_o$$

$$\delta_h=(\delta_o w_2)\odot(1-h^2)$$

$$g_{W_1}=\delta_h\tilde{x}^T,\qquad g_{b_1}=\delta_h$$

Every parameter is updated by subtracting eta times its batch gradient. All gradients are calculated from the same pre-update parameters. The derivative of tanh is 1 minus h squared. See [Goodfellow, Bengio and Courville (2016), Chapter 6](https://www.deeplearningbook.org/contents/mlp.html) for feed-forward networks and backpropagation.

Weights use Gaussian initialization with mean zero and variance 1/fan-in: 1/6 for W1 and 1/H for w2, with zero biases and random seed 42. This is the implemented fan-in rule; it is not labelled as exact Xavier/Glorot initialization. Random hidden weights break the symmetry that would make identical hidden units learn identical functions.

### 6.2 Model selection and learned parameters

Each width is trained for 2,000 full-batch epochs, learning rate 0.5, without regularization or early stopping:

@@selection@@

**H = @@h@@** has the lowest validation loss and is selected. This network has **@@parameter_count@@ parameters**, compared with seven for logistic regression. Its objective is nonconvex, so these are the parameters obtained from this seed and search, with no claim of a global optimum.

The selected parameters are rounded below; `results/results.json` stores full precision. W1:j is the weight from input j into that hidden unit.

@@parameters@@

Output bias b2 = **@@output_bias@@**. These weights operate on standardized inputs.

![Selected feed-forward network training cross-entropy](results/figures/ffnn_training_loss.png)

## 7. Code, pseudocode and correctness

Run `python train.py` to validate the CSV, compute features, split and scale the data, train both models, select H by validation loss, and save results. The complete gradient and backpropagation implementations are in `sentiment/logistic_regression.py` and `sentiment/neural_network.py`.

```text
read labelled reviews; reject missing/unknown/conflicting labels
remove duplicate texts using HTML/case/whitespace normalization
extract the six features from each retained review
split indices by class using seed 42 into train / validation / test
fit mean and standard deviation on training features only
standardize each partition with those training statistics

w = zeros(6); b = 0
repeat at most 100000 times:
    p = sigmoid(Z_train @ w + b)
    gw = Z_train.T @ (p - y_train) / m
    gb = mean(p - y_train)
    if sqrt(sum(gw**2) + gb**2) < 0.000001: stop
    w = w - 0.5 * gw
    b = b - 0.5 * gb
check convergence; convert weights and bias to the raw scale

for H in [2, 4, 8]:
    initialize W1 and w2 randomly with variance 1/fan_in
    initialize b1 and b2 to zero
    repeat 2000 times:
        h = tanh(Z_train @ W1.T + b1)
        p = sigmoid(h @ w2 + b2)
        do = (p - y_train) / m
        dh = outer(do, w2) * (1 - h**2)
        compute gW1=dh.T@Z_train, gb1=sum_rows(dh)
        compute gw2=h.T@do, gb2=sum(do)
        subtract 0.5 times each gradient from its parameter
    compute validation cross-entropy
select H with the lowest validation cross-entropy
evaluate the fixed models on test reviews at threshold 0.5
save weights, metrics, split membership and per-review predictions
```

`@` means matrix multiplication; `*` and `**2` mean elementwise multiplication and squaring. For example, every row of `outer(do, w2)` multiplies one review's output error by all hidden-to-output weights. Multiplying by `1-h**2` applies the tanh derivative. The transpose-and-multiply operations then sum the contribution of every training review to each parameter gradient. This is the chain rule expressed as array operations.

`sigmoid` and cross-entropy use `np.logaddexp`, avoiding explicit logarithms of rounded probabilities. The stable logit loss is log(1+exp(z)) - yz. Tests compare analytic derivatives against finite differences for both models, check the raw-scale conversion, and compare logistic gradient descent against a Newton solution. Regression tests cover duplicate normalization, invalid labels and constant features. Numerical tests use small random matrices or explicit text fixtures to test code; none of those fixtures enters the reported experiment. Run `python -m pytest -q` to reproduce these checks.

The experiment itself also checks split disjointness, logistic convergence, decreasing logistic training loss, agreement with Newton's method, and equivalent predictions after converting to raw coefficients. Passing these checks supports implementation correctness within their scope; it does not establish that the six features represent all aspects of language.

## 8. Concrete learning examples

### 8.1 One logistic-regression update on a real training review

Choose the first training review with at most 60 tokens, independently of the model prediction. It is **@@example_source@@**. Its supplied label is **@@example_label@@**, y = @@example_y@@:

> @@example_text@@

It has @@example_count@@ tokens. Positive matches: *@@example_positive@@*. Negative matches: *@@example_negative@@*. The following values come from the feature extractor and a separate one-review demonstration, initialized at zero. This illustrative update does not replace the full-batch fit used for evaluation.

@@step_table@@

At initialization, z = 0, p = 0.5, and loss = 0.693147. The error is p-y; multiply it by each standardized feature to obtain the gradient. At learning rate 0.5, each weight becomes minus 0.5 times its gradient and the bias becomes **@@step_bias@@**. Recomputing the prediction yields **p = @@step_p@@** and loss **@@step_loss@@**. This step makes the observed label more likely for this review. A single review's update is not evidence of how well the model generalizes.

Full-batch training then averages errors over all @@train_size@@ training reviews. Actual snapshots show how the parameters and loss evolve:

@@trace@@

### 8.2 One hidden-layer backpropagation update

A separate freshly initialized H = @@h@@ network receives the same real training review. Its positive probability changes from **@@nn_before_p@@** to **@@nn_after_p@@**; its one-review cross-entropy changes from **@@nn_before_loss@@** to **@@nn_after_loss@@** after one backpropagation update.

For W1 the table shows entry [0,0]; for b1 and w2 it shows entry [0]; for b2 it shows the scalar. Every row satisfies after = before - 0.5 times gradient. All parameter arrays and gradients for this example are saved under `learning_example.ffnn_step`.

@@nn_step_table@@

This example directly demonstrates a hidden-layer learning update, while Section 8.1 provides the simpler hand calculation for the logistic unit.

## 9. Concrete classification of a new input

Use the first test review with at most 80 tokens, chosen by length and order, not by whether its prediction is correct. It is **@@new_source@@** and was excluded from training and validation:

> @@new_text@@

The supplied label is **@@new_label@@**. It has @@new_count@@ tokens. Positive lexicon matches: *@@new_positive@@*. Negative matches: *@@new_negative@@*. Applying the trained raw weights gives:

@@contributions@@

The total logit is **@@new_z@@**. Applying the sigmoid gives **P(positive) = @@new_p@@**. Thresholding at 0.5 gives **@@new_prediction@@**. The selected feed-forward network, using the same review standardized with training statistics, gives **P(positive) = @@nn_new_p@@**, hence **@@nn_new_prediction@@**. These are model estimates for a real held-out input, not invented labels.

## 10. Evaluation and interpretation

Positive is the target class for precision, recall and F1. TP counts correctly predicted positive reviews, TN correctly predicted negatives, FP negatives predicted positive, and FN positives predicted negative:

$$\mathrm{accuracy}=\frac{TP+TN}{TP+TN+FP+FN}$$

$$\mathrm{precision}=\frac{TP}{TP+FP},\qquad \mathrm{recall}=\frac{TP}{TP+FN}$$

$$F_1=\frac{2\,\mathrm{precision}\,\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}$$

@@scores@@

Test confusion counts:

@@confusion@@

The majority baseline is **@@baseline@@**; the majority label is chosen from training labels. The selected network's test accuracy differs from logistic regression by **@@gap@@ percentage points**. A single seed and split do not establish a statistically reliable advantage across samples or datasets. Similar training and held-out scores are consistent with a limited generalization gap here, but do not prove the absence of overfitting or domain shift.

**A real error.** The first short misclassified logistic test review is **@@wrong_source@@**; supplied label **@@wrong_label@@**, predicted positive probability **@@wrong_p@@**:

> @@wrong_text@@

This review illustrates that a six-number summary can lose context. An example can suggest a possible failure mechanism, but does not determine the proportion or causes of all classification errors.

## 11. Limitations and justified conclusions

The six required features discard word order and most context. The `no` indicator does not associate a negator with its target, and the lexicon can count plot descriptions as opinions. Exact token matching misses some inflections and contractions. These are representation limitations, not a measured explanation for every error.

The experiment evaluates three small neural networks with one initialization per width; it does not establish the best possible network or a performance ceiling for the six features. The weights approximate the specified logistic training optimum and the validation-selected neural fit. They are specific to this preprocessing, dataset and split.

The corrected experiment uses published IMDB review records, known label values, and no fabricated training or evaluation examples. Dataset labels and the original collection process are inherited from the source. Checksum and split artifacts make the calculations auditable; they do not guarantee each original review or label is error free.

Under these conditions, both learned models exceed the majority baseline. Logistic regression offers seven interpretable parameters and a convex objective. The hidden layer changes performance by the measured amount above, with more parameters and a nonconvex objective. Future experiments could study negation handling or richer representations, but those would change this six-feature task and must be evaluated separately. No benefit from those untested changes is claimed here.

## 12. Reproducibility and assignment coverage

The run used **@@packages@@**. Exact package versions, source-code hashes and a lexicon fingerprint are in `results/results.json`.

| Artifact | Purpose |
| --- | --- |
| `train.py` | Recompute the entire experiment from the source CSV |
| `results/results.json` | Full coefficients, metrics, configuration, data checks and examples |
| `results/split_indices.npz` | Original zero-based CSV row numbers for each partition |
| `results/test_predictions.npz` | True labels and both model probabilities per test review |
| `results/figures/` | Training curves generated from actual loss histories |
| `REPORT.md` | Written submission, generated from the computed JSON |
| `Project1_Sentiment_LR_FFNN.ipynb` | Runnable notebook using the same training implementation |
| `Albert_Kabore_Project1_Report.pdf` | Printable version with no watermark overlays or browser headers |

| Assignment requirement | Where addressed |
| --- | --- |
| All six sentiment features | Section 3 |
| Equations and plain-English symbols | Sections 4 and 6 |
| Learned weights and bias | Sections 5 and 6.2; full precision in JSON |
| Code or pseudocode | Section 7 and the Python source files |
| Explanation and justification | Sections 4.3 and 7 |
| Concrete training example | Sections 8.1 and 8.2 |
| Concrete classification example | Section 9 |
| Clear organization and citations | Numbered sections and references below |

## 13. References

1. Lakshmipathi N. (n.d.). *IMDB Dataset of 50K Movie Reviews* [Dataset]. Kaggle. [Dataset page](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). Local dataset-version-1 CSV used; see checksum in Section 2.
2. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. *Proceedings of ACL-HLT*, 142-150. [Paper and bibliographic record](https://aclanthology.org/P11-1015/).
3. Hu, M., & Liu, B. (2004). Mining and summarizing customer reviews. *Proceedings of KDD*, 168-177. [DOI](https://doi.org/10.1145/1014052.1014073). The evolving opinion lexicon is documented on [Bing Liu's research page](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html).
4. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. [Book](https://www.nltk.org/book/) and [NLTK corpus documentation](https://www.nltk.org/howto/corpus.html).
5. Jurafsky, D., & Martin, J. H. (2026). *Speech and Language Processing*, 3rd edition online manuscript, August 19, 2026. [Manuscript](https://web.stanford.edu/~jurafsky/slp3/); [Logistic Regression and Text Classification](https://web.stanford.edu/~jurafsky/slp3/4.pdf).
6. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. [Author-hosted book](https://web.stanford.edu/~boyd/cvxbook/).
7. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [Chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html).
8. NumPy developers. (n.d.). [numpy.logaddexp documentation](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html). Stable computation of logarithms of sums of exponentials.
'''
    for key, value in values.items():
        template = template.replace(f"@@{key}@@", str(value))
    if "@@" in template:
        raise ValueError("Unresolved report template value")
    Path(destination).write_text(template + "\n", encoding="utf-8")
    return template
