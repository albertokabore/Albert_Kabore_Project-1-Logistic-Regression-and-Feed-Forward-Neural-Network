# Project 1: Logistic Regression and Feed-Forward Neural Network

Sentiment classification using exactly six assignment features and two models implemented in NumPy. Training and evaluation use published IMDB reviews; small test fixtures are confined to the test suite.

- **Notebook:** [Project1_Sentiment_LR_FFNN.ipynb](Project1_Sentiment_LR_FFNN.ipynb) verifies saved results and renders the report. Set `RECOMPUTE = True` to retrain.
- **Report:** [REPORT.md](REPORT.md) and [Albert_Kabore_Project1_Report.pdf](Albert_Kabore_Project1_Report.pdf) cover the equations, best weights, pseudocode, an explanation of the code, worked examples and references.
- **Word copy:** [Albert_Kabore_Project1_Report.docx](Albert_Kabore_Project1_Report.docx) duplicates the final report with editable equations, tables, figures, references and the AI disclosure in black text.
- **AI disclosure:** [AI_USE_DISCLOSURE.md](AI_USE_DISCLOSURE.md) is appended after the references in the report, PDF and notebook, identifying the assistance used, verification performed and author responsibility.

## Data

- [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle, Lakshmipathi N). It is associated with the Large Movie Review Dataset of Maas et al. (2011).
- A copy is stored at `data/IMDB Dataset.csv` (66 MB). If it is missing, the notebook downloads it again from Kaggle with `kagglehub`.
- Positive and negative word lists: the Hu & Liu (2004) opinion lexicon, loaded through NLTK.

## Quick start

```powershell
python -m pip install -r requirements.txt
python train.py
python -m pytest
python verify_results.py
```

To regenerate the report, executed notebook and PDF:

```powershell
python -m pip install -r requirements-report.txt
python -m playwright install chromium
python build_submission.py --execute-notebook
```

Run `python build_word.py` to duplicate the current report as a Word document. It uses the `MML2OMML.XSL` equation converter supplied with Microsoft Office; use `--math-xsl PATH` if it is installed elsewhere.

PDF rendering uses local mathematical SVGs and Chromium with browser headers disabled. The output has no watermark overlays. Training takes a few minutes, depending on the machine; exact numerical differences can occur across NumPy/BLAS versions. Initial dataset and NLTK lexicon downloads require network access.

The corrected experiment removes 418 exact duplicates and four formatting duplicates, leaving **49,578 reviews**. The split contains 34,704 training, 7,438 validation and 7,436 test reviews, with no overlap under HTML/case/whitespace normalization. The report explains the corrected evaluation, exact tokenization conventions, and limits of numerical optimization.

## Layout

| Path | Contents |
|---|---|
| `train.py` | Reproduce the complete experiment from real CSV records |
| `build_submission.py` | Verify artifact hashes and render Markdown, notebook and PDF |
| `verify_results.py` | Recompute predictions and metrics from the source reviews and saved parameters |
| `sentiment/data.py` | Kaggle download, label validation, normalized duplicate removal, stratified split |
| `sentiment/lexicon.py` | Hu & Liu lexicon loader and pronoun list |
| `sentiment/features.py` | Tokenizer and the six feature extractors |
| `sentiment/preprocessing.py` | Standardization and conversion of weights to the raw feature scale |
| `sentiment/logistic_regression.py` | Logistic regression trained with gradient descent |
| `sentiment/neural_network.py` | 6 → H (tanh) → 1 (sigmoid) network trained with backpropagation |
| `sentiment/reporting.py` | Generate the written report from computed results |
| `results/` | Full-precision JSON, split CSV row indices, per-review predictions, verification and training curves |
| `tests/` | Feature, gradient-check, convergence and scaling tests |

## Results (7,436 held-out test reviews)

| Model | Test accuracy |
|---|---:|
| Logistic regression | 73.57% |
| Feed-forward network (8 hidden units) | 74.70% |
| Majority-class baseline | 50.19% |

These replace the earlier numbers from exact-only deduplication. This experiment uses a random CSV split rather than Stanford's official benchmark split. Dataset provenance, the input SHA-256, worked examples, references and evaluation limitations are documented in the report.
