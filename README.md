# Sales Forecasting & Demand Intelligence System

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
1. Push this folder to a new public GitHub repository.
2. Go to https://share.streamlit.io, sign in, click "New app".
3. Point it at your repo, branch `main`, and file `app.py`.
4. Deploy — Streamlit Cloud installs `requirements.txt` automatically.
5. Copy the live `*.streamlit.app` URL it gives you and paste it into the submission form
   along with your GitHub repo link and this folder as a ZIP.

## Model recommendation (Task 3)
XGBoost had the lowest holdout error (MAPE ~19.3%), narrowly ahead of SARIMA (~20.5%)
and Prophet (~21.9%) — but **SARIMA is the recommended production model** because it
provides calibrated confidence intervals and avoids the compounding-error risk of
XGBoost's recursive multi-step forecast. Full reasoning is in the notebook (Task 3)
and the executive report.
