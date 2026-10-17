# Detecting European Portuguese fake news

[![DOI](https://zenodo.org/badge/1114744293.svg)](https://doi.org/10.5281/zenodo.18393435)

🥈 **2nd place, Machine Learning (IST) & NTT Data Challenge 2025–26** · **[Paper (Zenodo)](https://doi.org/10.5281/zenodo.18393436)** · **[▶ Live demo](https://duartepcruz.com/work/fake-news-detection/)** · [Report (PDF)](submission/report/group9-report.pdf)

Five classifiers trained on 63k European Portuguese news articles, plus a close look at *why* they decide what they decide. Off-the-shelf fake-news tooling is mature for English and thin for European Portuguese, which is the gap this project targets.

## Results

Test set: 6,324 held-out articles, macro-F1.

| Model | Test F1 |
|---|---|
| Decision tree | 0.859 |
| Gaussian naive Bayes | 0.852 |
| Logistic regression (L2) | 0.900 |
| Logistic regression (L1) | 0.901 |
| **MLP (256 → 128)** | **0.926** (ROC AUC 0.978) |
| Fine-tuned BERT (reference) | 0.948 |

**Takeaway:** a small MLP on TF-IDF features gets within 0.02 F1 of fine-tuned BERT, at a fraction of the compute.

## Method

- **Data:** FakeNews-PT ([fake-news-pt-eu](https://github.com/ro-afonso/fake-news-pt-eu)), with 63,235 articles split 50,587 / 6,324 / 6,324 (train/val/test).
- **Preprocessing ablation:** we ran the whole pipeline four times (± punctuation, ± Portuguese stop words). Keeping punctuation and dropping stop words worked best; removing punctuation hurt most, likely because clickbait relies on "!" and quotes.
- **Features:** TF-IDF, 5,000 terms (min_df=10, max_df=0.9).
- **Models:** grid search with 5-fold CV, optimising macro-F1.

## What the models learned

- **Sparsity:** L1 zeroed 2,898 of 5,000 word weights, which leaves a short, readable list of the words that separate real from fake.
- **Local explanations (LIME):** the linear model leans on the same global words every time, while the MLP weighs words by context.
- **Global view:** permutation importance over 1,000 validation articles.
- **Topics:** K-Means (k=5) plus PCA found Ukraine war, health, domestic politics, economy, and clickbait. Fake articles cluster more densely in one region but appear across every topic, so misinformation imitates whatever is in the news.

## Limitations

The models judge **style, not truth**. A fabricated story written like a wire report ("anunciou", "Lusa") is classified as real, and a true sentence in clickbait style is classified as fake. Words outside the 5,000-term vocabulary are invisible, so new names and events don't register.

## Repository

| Path | Contents |
|---|---|
| [`submission/`](submission/) | The submitted notebook and report |
| `notebooks/<variant>/` | The same pipeline for each preprocessing variant (`punctuation`, `punctuation_no_stop_words`, `no_punctuation`, `no_punctuation_no_stop_words`): `ex1` TF-IDF + model training, `ex2` interpretability (weights, L1 vs L2, LIME, permutation importance), `ex3` K-Means topic clustering |
| [`notebooks/artifacts/`](notebooks/artifacts/) | Fitted vectorizers and best models (`.joblib`) for each variant |
| [`notebooks/data/`](notebooks/data/) | Train / validation / test CSVs |
| [`notebooks/report_stuff/`](notebooks/report_stuff/) | Notebooks that generate the report's figures |
| [`docs/`](docs/) | Challenge statement and notes |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Authors
Daniel Martins Borges & Duarte Pereira da Cruz (Group 9)
