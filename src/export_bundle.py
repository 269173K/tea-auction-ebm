"""
Export everything the web front-end needs into app/static/model_bundle.json:
  * the deployed EBM (bins + term tables) and SHAP background expectations,
  * feature metadata in user-facing units, with slider ranges from the training data,
  * the latest observed inputs per elevation (September 2026 -> forecast October 2026),
  * 36 months of price history per elevation, and headline evaluation metrics.
The Flask app uses the pickled model for /api/predict; the bundle lets the page draw
inputs, history and shape functions (and run standalone if no server is present).
"""
from pathlib import Path
import json, pickle, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import train_evaluate as T

ROOT = Path(__file__).resolve().parents[1]
MOD, TAB, OUT = ROOT / "models", ROOT / "outputs" / "tables", ROOT / "app" / "static"
OUT.mkdir(parents=True, exist_ok=True)

# UI metadata: label, group, how a slider value maps to the model feature
META = {
    "world_tea_chg_3m": ("World tea price, change over last 3 months", "market", "pct_log", "%"),
    "ceylon_premium": ("Colombo price premium over world average", "market", "pct_log", "%"),
    "brent_chg_3m": ("Brent crude oil, change over last 3 months", "market", "pct_log", "%"),
    "fx_chg_3m": ("Rupee against the dollar, change over 3 months (+ = weaker)", "market", "pct_log", "%"),
    "rain_anom_2m": ("Rainfall vs normal, average of last 2 months", "weather", "raw", "mm/day"),
    "temp_anom": ("Temperature vs normal, last month", "weather", "raw", "°C"),
    "solar_anom": ("Sunshine vs normal, last month", "weather", "raw", "MJ/m²/day"),
    "ret_1m": ("Price change last month", "prices", "pct_log", "%"),
    "ret_12m": ("Price change over 12 months", "prices", "pct_log", "%"),
    "dev_12m_mean": ("Price vs its 12-month average", "prices", "pct_log", "%"),
    "premium_vs_total": ("Price vs national average (all elevations)", "prices", "pct_log", "%"),
}


def to_ui(v, kind):
    return (np.exp(v) - 1) * 100 if kind == "pct_log" else v


def main():
    deploy = pickle.load(open(MOD / "ebm_deploy.pkl", "rb"))
    df = T.load_panel()
    lab = df.dropna(subset=["target"])
    bg = T.fit_rows(lab)
    F = T.FEATURES
    Xbg = bg[F].values.astype(float)
    B, P = deploy._bin_all(Xbg), deploy._bin_pairs(Xbg)

    model = deploy.to_json()
    model["feature_order"] = F
    main_means = [float(t[B[:, j]].mean()) for j, t in enumerate(deploy.term_scores_)]
    pairs_bg = []
    for (i, j), tab in deploy.pair_tables_.items():
        pairs_bg.append({"Ei_given_j": [float(tab[P[:, i], bj].mean()) for bj in range(tab.shape[1])],
                         "Ej_given_i": [float(tab[bi, P[:, j]].mean()) for bi in range(tab.shape[0])],
                         "E": float(tab[P[:, i][:, None], P[:, j][None, :]].mean())})
    model["shap_background"] = {"main_means": main_means, "pairs": pairs_bg, "n_rows": int(len(bg))}

    features = []
    for f in F:
        if f in META:
            label, group, kind, unit = META[f]
            ui = to_ui(bg[f].values, kind)
            lo, hi = np.percentile(ui, [1, 99])
            pad = (hi - lo) * 0.1
            step = 0.1 if unit in ("%", "°C") or kind == "raw" else 0.1
            features.append({"name": f, "label": label, "group": group, "transform": kind, "unit": unit,
                             "min": round(float(lo - pad), 1), "max": round(float(hi + pad), 1), "step": step})
    latest_month = df["month"].max()
    state = {}
    usd = pd.read_csv(ROOT / "data" / "processed" / "usd_prices_monthly.csv", index_col=0)
    lkr = pd.read_csv(ROOT / "data" / "raw" / "prices_monthly_lkr.csv", index_col=0)
    for _, r in df[df["month"] == latest_month].iterrows():
        e = r["elevation_name"]
        state[e] = {"origin": r["month"], "target": r["target_month"], "usd_t": float(r["usd_price_t"]),
                    "lkr_t": float(r["lkr_price_t"]), "fx_t": float(r["fx_t"]),
                    "features": {f: float(r[f]) for f in F}}
    history = {e: [{"month": m, "usd": round(float(usd.loc[m, e]), 4), "lkr": round(float(lkr.loc[m, e]), 2)}
                   for m in usd.index[-36:]] for e in ["high", "medium", "low"]}
    shapes = {}
    for f in [x["name"] for x in features]:
        s = deploy.shape_function(f)
        meta = META[f]
        shapes[f] = {"x": [float(v) for v in to_ui(np.asarray(s["x"]), meta[2])],
                     "score_pp": [float(v * 100) for v in s["score"]], "std_pp": [float(v * 100) for v in s["std"]]}
    seas = pd.read_csv(TAB / "seasonal_effect_by_elevation.csv")
    tm = pd.read_csv(TAB / "test_metrics.csv").set_index("model")
    ro = pd.read_csv(TAB / "rolling_origin_summary.csv").set_index("model")
    metrics = {"test_rmse_ebm": tm.loc["EBM", "RMSE_change_pct"], "test_rmse_naive": tm.loc["Naive (no change)", "RMSE_change_pct"],
               "test_dir_ebm": tm.loc["EBM", "Direction_acc_pct"], "test_mape_ebm": tm.loc["EBM", "MAPE_pct"],
               "rolling_rmse_ebm": ro.loc["EBM", "RMSE_change_pct"], "rolling_rmse_naive": ro.loc["Naive (no change)", "RMSE_change_pct"]}
    bundle = {"model": model, "features": features, "state": state, "history": history, "shapes": shapes,
              "seasonality": seas.to_dict(orient="records"),
              "metrics": {k: round(float(v), 2) for k, v in metrics.items()},
              "data_through": latest_month}
    json.dump(bundle, open(OUT / "model_bundle.json", "w"), separators=(",", ":"))
    print("bundle KB", round((OUT / "model_bundle.json").stat().st_size / 1024, 1), "features", len(features), "state", list(state))

    # parity fixture for the JS engine test: inputs + Python predictions and SHAP
    test = lab[lab["target_month"] >= "2025-01"].head(12)
    phi, base = deploy.shap_values(test[F].values, background=Xbg)
    fx = {"rows": [dict(zip(F, map(float, x))) for x in test[F].values],
          "pred": deploy.predict(test[F].values).tolist(), "phi": phi.tolist(), "base": float(base)}
    json.dump(fx, open(ROOT / "app" / "parity_fixture.json", "w"))


if __name__ == "__main__":
    main()
