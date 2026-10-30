# Forecasting Colombo Tea Auction Prices with an Explainable Boosting Machine

Machine learning assignment. Forecasts next month's Colombo Tea Auction average price for
High, Medium and Low grown tea, explains every forecast, and serves the model in a web app.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py          # rebuilds dataset, models, figures, tables (~2 min)
python app/app.py          # open http://127.0.0.1:5000
```

Optional checks (Node.js 18+): `node app/test_parity.js` (browser twin = Python model)
and `node app/test_render.js` (page renders without errors).

## Deploy to Azure App Service (free F1 tier)

Needs the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli). Run from a fresh
clone of this repo (no `.venv` folder in it, so it isn't uploaded):

```bash
az login
az webapp up --name tea-outlook-269173k --resource-group tea-outlook-rg \
  --location centralindia --runtime "PYTHON:3.12" --sku F1
az webapp config set --name tea-outlook-269173k --resource-group tea-outlook-rg \
  --startup-file "gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers 1 --threads 4 application:app"
```

The site is then at `https://tea-outlook-269173k.azurewebsites.net`. Logs:
`az webapp log tail --name tea-outlook-269173k --resource-group tea-outlook-rg`.
To redeploy after changes, run the `az webapp up` line again.

## Layout

| Path | Contents |
| --- | --- |
| `data/raw/` | Compiled source data (CSV), with the source of every price row |
| `data/processed/tea_panel.csv` | Modelling panel: 279 rows (month x elevation), 19 candidate features + target |
| `src/build_dataset.py` | Merging, USD conversion (validated against published USD prices), weather anomalies, features |
| `src/ebm.py` | Explainable Boosting Machine implemented from scratch (NumPy, scikit-learn API) incl. exact SHAP |
| `src/train_evaluate.py` | Chronological split, tuning, baselines, test and rolling-origin evaluation |
| `src/explain.py` | Shape functions, importance, exact SHAP, local waterfall, PDP (EBM vs RF), permutation importance |
| `src/export_bundle.py` | Exports the model and data the web app needs |
| `models/` | Trained models (`ebm_deploy.pkl` serves the app) |
| `outputs/figures`, `outputs/tables` | Every figure and table in the report |
| `app/` | Flask app (`app.py`, `templates/`, `static/`), standalone page, JS tests |

## Data sources (all public, aggregated, no personal data)

- Elevation-wise auction averages: Forbes & Walker Tea Brokers / Sri Lanka Tea Board
  (Tea Reviews 2020-2022; monthly "National Tea Sale Averages" 2022-2026; press reports for May-Sep 2026)
- USD/LKR monthly averages: x-rates.com
- Brent crude (MCOILBRENTEU) and global tea price (PTEAUSDM): FRED, Federal Reserve Bank of St. Louis
- Weather: NASA POWER (Nuwara Eliya, Kandy, Ratnapura)

The raw source documents are not redistributed; only the extracted figures are, with citations.

## Demo video outline (3-5 minutes)

1. (0:00) The problem: monthly auction prices by elevation; who uses them.
2. (0:40) Open the app. Pick High grown: October 2026 forecast and the "Why" chart.
3. (1:30) Switch to Low grown; point out how the drivers differ.
4. (2:00) What-if: set the month to June and show the seasonal dip; open "What the model learned" for the month.
5. (2:45) What-if: raise rainfall to +6 mm/day; the forecast and the rainfall bar update.
6. (3:20) The accuracy footer: how far to trust it (vs the naive forecast).
7. (3:50) Close: data sources and what would improve it.
