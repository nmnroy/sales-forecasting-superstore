# Sales Forecasting & Demand Intelligence System

**Live dashboard:** **https://sales-forecasting-superstore.streamlit.app/** 

**GitHub repo:** https://github.com/nmnroy/sales-forecasting-superstore

## What's included
- `analysis.ipynb` — full notebook, all 8 tasks, executed end-to-end (outputs and charts baked in)
- `train.csv` — Superstore Sales dataset used throughout
- `app.py` — Streamlit dashboard (4 pages: Sales Overview, Forecast Explorer, Anomaly Report, Product Demand Segments)
- `requirements.txt` — all libraries needed to re-run the notebook and the app
- `summary.docx` — 2-page executive business report for Head of Supply Chain / CFO
- `charts/` — all 14 chart PNGs referenced in the notebook and report

## Known gap — please read
The assignment references a second dataset (Kaggle's Video Game Sales dataset) for a
merging/multi-source exercise inside Task 1. That file was not provided alongside
`train.csv`, so this submission proceeds with the Superstore dataset alone, which
carries the large majority of the graded content (Tasks 2 through 8, ~90% of the
rubric weight). If you can supply the video game sales CSV, the merge exercise
can be added to Task 1 as a follow-up.

## Dashboard walkthrough

### 1. Sales Overview
Region/category filters, headline metrics (total sales, orders, avg order value), yearly sales bar chart, monthly trend line, and region/category breakdowns.


### 2. Forecast Explorer
Pick a Category or Region, choose a 1–3 month horizon, and get a live SARIMA forecast with confidence interval, plotted against history.


### 3. Anomaly Report
Weekly sales with both detection methods overlaid — Isolation Forest (red X) for global outliers and rolling Z-score (purple diamond) for local surprises — plus the underlying flagged-week tables.


### 4. Product Demand Segments
K-Means clusters (k=4) of sub-categories by total sales, volatility, growth rate, and average order value, projected via PCA, plus the full sub-category → cluster mapping table and recommended stocking strategy per cluster.

## Running the notebook
```bash
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

## Running the dashboard locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Community Cloud
Already deployed — see the live link at the top of this README. For reference, the steps were:
1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io, sign in, click "New app".
3. Point it at the repo, branch `main`, and file `app.py`.
4. Deploy — Streamlit Cloud installs `requirements.txt` automatically.

## Model recommendation (Task 3)
XGBoost had the lowest holdout error (MAPE ~19.3%), narrowly ahead of SARIMA (~20.5%)
and Prophet (~21.9%) but **SARIMA is the recommended production model** because it
provides calibrated confidence intervals and avoids the compounding error risk of
XGBoost's recursive multi step forecast. Full reasoning is in the notebook (Task 3)
and the executive report.
