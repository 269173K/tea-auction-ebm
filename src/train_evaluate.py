"""
Train, tune and evaluate the EBM against baselines and standard models.

Split (chronological, by the month being forecast):
  train      : target months 2019-02 .. 2023-12
  validation : two expanding-window folds, 2023-07..2023-12 and 2024-01..2024-12
               (hyperparameters chosen on the average of the two)
  test       : target months 2025-01 .. 2026-09   (held out, touched once)
Structural break: target months 2022-03 .. 2023-06 (rupee float, devaluation and
re-appreciation) are excluded from every training set. A first run that kept them,
with all 19 features, was worse than the naive forecast on validation (see report).
Plus a rolling-origin check: for each year 2023 (Jul-Dec), 2024, 2025, 2026 the models
are re-trained on everything before that period and scored on it.
"""
from pathlib import Path
import itertools, json, pickle, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent))
from ebm import EBMRegressor
from plot_style import apply_style, COLORS, ELEV_COLORS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "outputs" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
MOD = ROOT / "models"; MOD.mkdir(parents=True, exist_ok=True)

ELEV_CODE = {"high": 0, "medium": 1, "low": 2}
PRICE_FEATURES = ["ret_1m", "ret_12m", "dev_12m_mean", "premium_vs_total"]
DRIVER_FEATURES = ["elevation", "next_month", "fx_chg_3m", "brent_chg_3m", "world_tea_chg_3m",
                   "ceylon_premium", "rain_anom_2m", "temp_anom", "solar_anom", "fertilizer_ban"]
FEATURES = DRIVER_FEATURES[:2] + PRICE_FEATURES + DRIVER_FEATURES[2:]
CATEGORICAL = ["elevation", "next_month", "fertilizer_ban"]
BREAK = ("2022-03", "2023-06")


def fit_rows(df):
    """Rows usable for fitting: everything outside the 2022-23 currency-crisis break."""
    return df[~df["target_month"].between(*BREAK)]


def load_panel():
    df = pd.read_csv(DATA / "tea_panel.csv")
    df["elevation_name"] = df["elevation"]
    df["elevation"] = df["elevation"].map(ELEV_CODE)
    return df


def seasonal_naive(df):
    """Same calendar-month change one year earlier, same elevation."""
    usd = pd.read_csv(DATA / "usd_prices_monthly.csv", index_col=0)
    usd.index = pd.PeriodIndex(usd.index, freq="M")
    out = []
    for _, r in df.iterrows():
        t1 = pd.Period(r["target_month"], "M"); e = r["elevation_name"]
        a, b = t1 - 12, t1 - 13
        out.append(np.log(usd.loc[a, e] / usd.loc[b, e]) if a in usd.index and b in usd.index else 0.0)
    return np.array(out)


def linear_features(X):
    """Linear model needs cyclic month and one-hot elevation."""
    Z = X.drop(columns=["next_month", "elevation"]).copy()
    Z["month_sin"] = np.sin(2 * np.pi * X["next_month"] / 12)
    Z["month_cos"] = np.cos(2 * np.pi * X["next_month"] / 12)
    Z["elev_medium"] = (X["elevation"] == 1).astype(float)
    Z["elev_low"] = (X["elevation"] == 2).astype(float)
    return Z


def metrics(df_part, pred_change):
    y = df_part["target"].values
    p0 = df_part["usd_price_t"].values
    actual = df_part["usd_price_t1"].values
    fc = p0 * np.exp(pred_change)
    move = np.abs(y) > 0.005           # ignore near-flat months for direction accuracy
    return {
        "RMSE_change_pct": float(np.sqrt(np.mean((y - pred_change) ** 2)) * 100),
        "RMSE_usd": float(np.sqrt(np.mean((actual - fc) ** 2))),
        "MAE_usd": float(np.mean(np.abs(actual - fc))),
        "MAPE_pct": float(np.mean(np.abs(actual - fc) / actual) * 100),
        # a model that never predicts a move (naive) makes no up/down call -> not applicable
        "Direction_acc_pct": (float(np.mean(np.sign(pred_change[move]) == np.sign(y[move])) * 100)
                              if np.any(pred_change != 0) else np.nan),
    }


# -------------------------------------------------------------------- model factory
def make_model(name, params, features):
    if name == "EBM":
        cats = [c for c in CATEGORICAL if c in features]
        return EBMRegressor(feature_names=features, categorical=cats,
                            forced_interactions=[("elevation", "next_month")], **params)
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1, **params)
    if name == "HistGradBoost":
        return HistGradientBoostingRegressor(random_state=0, **params)
    raise ValueError(name)


def fit_predict(name, params, tr, te, features):
    m = make_model(name, params, features)
    Xtr, Xte = tr[features], te[features]
    if name == "Ridge":
        Xtr, Xte = linear_features(Xtr), linear_features(Xte)
    m.fit(Xtr, tr["target"].values)   # caller passes break-filtered rows
    return m, m.predict(Xte)


GRIDS = {
    # Stumps (max_leaves=2) and a fixed number of boosting rounds tuned on the time-series
    # folds; the inner random early-stopping split proved too noisy at ~150 training rows.
    "EBM": [dict(learning_rate=0.01, max_rounds=mr, max_leaves=2, max_bins=mb, min_samples_leaf=10,
                 interactions=0, outer_bags=16, early_stopping_rounds=10**9)
            for mr, mb in itertools.product([30, 60, 100, 150, 250, 400], [8, 16])],
    "Ridge": [dict(alpha=a) for a in [0.1, 1, 10, 30, 100]],
    "RandomForest": [dict(max_depth=d, min_samples_leaf=l, max_features=f)
                     for d, l, f in itertools.product([3, 6, None], [5, 10], [0.33, 1.0])],
    "HistGradBoost": [dict(learning_rate=lr, max_depth=d, max_iter=it, min_samples_leaf=10)
                      for lr, d, it in itertools.product([0.03, 0.1], [2, 3], [100, 300])],
}


def main():
    apply_style()
    df = load_panel().dropna(subset=["target"]).reset_index(drop=True)
    train = df[df["target_month"] <= "2023-12"]
    val = df[(df["target_month"] >= "2023-07") & (df["target_month"] <= "2024-12")]
    test = df[df["target_month"] >= "2025-01"]
    trval = fit_rows(df[df["target_month"] <= "2024-12"])
    folds = [(fit_rows(df[df["target_month"] <= "2023-06"]), df[df["target_month"].between("2023-07", "2023-12")]),
             (fit_rows(df[df["target_month"] <= "2023-12"]), df[df["target_month"].between("2024-01", "2024-12")])]
    split_info = {k: {"rows": len(v), "from": v["target_month"].min(), "to": v["target_month"].max()}
                  for k, v in [("train_all", train), ("train_used", fit_rows(train)), ("validation", val),
                               ("test", test)]}
    split_info["break_excluded"] = list(BREAK)
    print(json.dumps(split_info, indent=1))
    naive_val = np.mean([np.sqrt(np.mean(v["target"].values ** 2)) * 100 for _, v in folds])
    print(f"Naive validation RMSE {naive_val:.3f}%")

    # ---------------- tuning on the validation folds
    tuning, best = [], {}
    for name, grid in GRIDS.items():
        scores = []
        for params in grid:
            t0 = time.time()
            fold_rmse = []
            for tr_f, va_f in folds:
                _, p = fit_predict(name, params, tr_f, va_f, FEATURES)
                fold_rmse.append(np.sqrt(np.mean((va_f["target"].values - p) ** 2)) * 100)
            rmse = float(np.mean(fold_rmse))
            scores.append((rmse, params))
            tuning.append({"model": name, "params": json.dumps(params), "val_RMSE_change_pct": rmse,
                           "fold_2023H2": fold_rmse[0], "fold_2024": fold_rmse[1],
                           "naive_val_RMSE_pct": naive_val, "fit_seconds": time.time() - t0})
        best[name] = min(scores, key=lambda s: s[0])[1]
        print(f"{name:14s} best val RMSE {min(s[0] for s in scores):.3f}%  {best[name]}")
    pd.DataFrame(tuning).to_csv(TAB / "hyperparameter_tuning.csv", index=False)
    json.dump(best, open(TAB / "best_params.json", "w"), indent=1)

    # ---------------- held-out test (refit on train+validation)
    results, preds = [], {"Naive (no change)": np.zeros(len(test)),
                          "Seasonal naive": seasonal_naive(test)}
    for name in GRIDS:
        m, p = fit_predict(name, best[name], trval, test, FEATURES)
        preds[name] = p
        if name == "EBM":
            ebm_test_model = m
    # drivers-only EBM (no price-history features)
    _, preds["EBM drivers-only"] = fit_predict("EBM", best["EBM"], trval, test, DRIVER_FEATURES)
    # post-hoc sensitivity (added after the test result): remove the training-period drift
    preds_sens = {f"{k} zero-drift (post-hoc)": preds[k] - trval["target"].mean() for k in ["EBM", "Ridge"]}
    for k, p in {**preds, **preds_sens}.items():
        results.append({"model": k, **metrics(test, p)})
    res = pd.DataFrame(results)
    naive_rmse = res.loc[res.model == "Naive (no change)", "RMSE_change_pct"].iloc[0]
    res["Skill_vs_naive_pct"] = (1 - res["RMSE_change_pct"] / naive_rmse) * 100
    res.to_csv(TAB / "test_metrics.csv", index=False)
    print(res.round(3).to_string(index=False))

    # per-elevation EBM metrics
    rows = []
    for e in ["high", "medium", "low"]:
        mask = (test["elevation_name"] == e).values
        rows.append({"elevation": e, **{f"EBM_{k}": v for k, v in metrics(test[mask], preds["EBM"][mask]).items()},
                     **{f"Naive_{k}": v for k, v in metrics(test[mask], np.zeros(mask.sum())).items()}})
    pd.DataFrame(rows).to_csv(TAB / "test_metrics_by_elevation.csv", index=False)

    test_out = test[["month", "target_month", "elevation_name", "usd_price_t", "usd_price_t1", "lkr_price_t1", "fx_t", "target"]].copy()
    for k, p in preds.items():
        test_out[f"pred_change_{k}"] = p
        test_out[f"pred_usd_{k}"] = test["usd_price_t"].values * np.exp(p)
    test_out.to_csv(TAB / "test_predictions.csv", index=False)

    # ---------------- rolling-origin evaluation
    roll = []
    for year, start, end in [("2023H2", "2023-07", "2023-12"), ("2024", "2024-01", "2024-12"),
                             ("2025", "2025-01", "2025-12"), ("2026", "2026-01", "2026-12")]:
        tr = fit_rows(df[df["target_month"] < start])
        te = df[df["target_month"].between(start, end)]
        ps = {"Naive (no change)": np.zeros(len(te)), "Seasonal naive": seasonal_naive(te)}
        for name in GRIDS:
            ps[name] = fit_predict(name, best[name], tr, te, FEATURES)[1]
        for k, p in ps.items():
            roll.append({"test_year": year, "n": len(te), "model": k, **metrics(te, p)})
    roll = pd.DataFrame(roll)
    roll.to_csv(TAB / "rolling_origin_metrics.csv", index=False)
    summary = roll.groupby("model")[["RMSE_change_pct", "MAPE_pct", "Direction_acc_pct"]].mean().sort_values("RMSE_change_pct")
    summary.to_csv(TAB / "rolling_origin_summary.csv")
    print(summary.round(3))

    # ---------------- final models
    pickle.dump(ebm_test_model, open(MOD / "ebm_trainval.pkl", "wb"))
    allfit = fit_rows(df)
    deploy = make_model("EBM", best["EBM"], FEATURES).fit(allfit[FEATURES], allfit["target"].values)
    pickle.dump(deploy, open(MOD / "ebm_deploy.pkl", "wb"))
    json.dump(deploy.to_json(), open(MOD / "ebm_deploy.json", "w"))
    drivers = make_model("EBM", best["EBM"], DRIVER_FEATURES).fit(trval[DRIVER_FEATURES], trval["target"].values)
    pickle.dump(drivers, open(MOD / "ebm_drivers_trainval.pkl", "wb"))
    json.dump({"features": FEATURES, "driver_features": DRIVER_FEATURES, "categorical": CATEGORICAL,
               "elevation_code": ELEV_CODE, "split": split_info, "break": BREAK}, open(MOD / "feature_spec.json", "w"), indent=1)

    # ---------------- figures
    plot_test_forecasts(test_out)
    plot_model_comparison(res, summary)


def plot_test_forecasts(t):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    for ax, e in zip(axes, ["high", "medium", "low"]):
        d = t[t.elevation_name == e]
        x = pd.PeriodIndex(d.target_month, freq="M").to_timestamp()
        ax.plot(x, d.usd_price_t1, color=COLORS["ink"], lw=2, label="Actual", marker="o", ms=3)
        ax.plot(x, d["pred_usd_EBM"], color=COLORS["primary"], lw=2, label="EBM forecast", marker="o", ms=3)
        ax.plot(x, d["pred_usd_Naive (no change)"], color=COLORS["muted"], lw=1.4, ls="--", label="Naive (last month)")
        ax.set_title(f"{e.capitalize()} grown", loc="left")
        ax.set_ylabel("USD per kg")
    axes[0].legend(ncol=3, loc="upper right", frameon=False)
    fig.suptitle("One-month-ahead forecasts on the held-out test period (2025-01 to 2026-09)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "test_forecasts.png", dpi=180); plt.close(fig)


def plot_model_comparison(res, roll_summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    order = res[~res.model.str.contains("post-hoc")].sort_values("RMSE_change_pct")
    cols = [COLORS["primary"] if m.startswith("EBM") else COLORS["muted"] for m in order.model]
    axes[0].barh(order.model, order.RMSE_change_pct, color=cols)
    axes[0].invert_yaxis(); axes[0].set_xlabel("RMSE of monthly % change (lower is better)")
    axes[0].set_title("Held-out test 2025-2026", loc="left")
    for i, v in enumerate(order.RMSE_change_pct):
        axes[0].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
    r = roll_summary.reset_index().dropna(subset=["Direction_acc_pct"])
    cols = [COLORS["primary"] if m.startswith("EBM") else COLORS["muted"] for m in r.model]
    axes[1].barh(r.model, r.Direction_acc_pct, color=cols)
    axes[1].invert_yaxis(); axes[1].axvline(50, color=COLORS["ink"], lw=0.8, ls=":")
    axes[1].set_xlabel("Direction accuracy % (rolling origin, mean of 4 periods)")
    axes[1].set_title("Up/down calls, Jul 2023 - Sep 2026", loc="left")
    for i, v in enumerate(r.Direction_acc_pct):
        axes[1].text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "model_comparison.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
