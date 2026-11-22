# SentimentSense Pro — NLP Transformer Intelligence Studio

A polished, interactive **Streamlit** dashboard for customer review sentiment classification using the pretrained `distilbert-base-uncased-finetuned-sst-2-english` model. No separate frontend/backend server is required. SQLite stores prediction history locally.

## Features (16+)
- Premium dark dashboard with responsive sidebar and interactive Plotly charts
- Dataset KPI cards, label share, review length histogram and keyword frequencies
- Live review classification, confidence indicator and example inputs
- CSV upload, selectable text column, batch predictions (up to 200 rows)
- Export classified CSV, filter/search labeled dataset, export filtered CSV
- Evaluation against actual labels: accuracy, precision, recall, F1 and confusion matrix
- Evaluation CSV upload and downloadable row-level results
- Word clouds, keyword exploration, sentiment-specific text intelligence
- Persistent SQLite history, filters, history charts, CSV export and clear history
- Built-in project architecture, limitations and viva questions

## Quick start (Windows, VS Code)

Open terminal **inside the `SentimentSense_Pro` folder**:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

**First prediction:** internet required to download model weights (~250 MB). Later runs use local cache. A normal CPU works; batch evaluation may take time. If `python` does not work, try `py` in the commands. Python 3.10–3.12 recommended.

## Dataset

`data/reviews.csv` is a **small synthetic illustrative dataset** with `review,sentiment` columns. It is **not** a real-world benchmark or training data for the model. You can upload a separate, properly labeled held-out CSV on the Model Evaluation page. Never report synthetic-demo accuracy as real-world performance.

## What the project actually does

DistilBERT is a pretrained binary English sentiment classifier fine-tuned on SST-2. The project **does not retrain** the model. The classifier always returns Positive or Negative (no Neutral class). The confidence score is not the same as evaluation accuracy.

## Monday demo sequence

1. Command Center: show dataset charts and KPIs.
2. Live AI Lab: predict one positive and one negative review.
3. Batch Analyzer: analyze `data/reviews.csv` and download results.
4. Model Evaluation: calculate metrics on labeled examples and explain the small demo dataset limitation.
5. Prediction Vault: show SQLite history.
6. Project & Viva: explain tokenization → DistilBERT → classification.

## Troubleshooting

- `ModuleNotFoundError`: install requirements using the **same Python interpreter** used to run Streamlit.
- Model download fails: connect to internet; rerun. No predictions are faked when model fails.
- Slow laptop: analyze 5–15 reviews at a time; first model load is slow.
- Windows `torch` installation problems: use Python 3.11 in a new virtual environment.
- Run from the extracted project folder, not the parent ZIP directory.

## Files

- `app.py`: dashboard, inference, evaluation and SQLite storage
- `data/reviews.csv`: demo dataset
- `requirements.txt`: dependencies
- `sentimentsense.db`: auto-created after first run
