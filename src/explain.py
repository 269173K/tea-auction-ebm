"""
Explainability analysis for the tea-price EBM.

  1. Global term importance (mean |contribution|)            - EBM native
  2. Shape functions with +/-1 sd bands across bags          - EBM native (exact)
  3. Seasonal effect by elevation (month + elevation x month)
  4. Exact SHAP values on the test set + one local waterfall  - SHAP (exact for GA2M)
  5. Partial dependence: EBM vs Random Forest                 - PDP (sklearn)
  6. Permutation importance on the test set                   - model-agnostic check
  7. Drivers-only EBM (no price-history features)
"""
from pathlib import Path
import json, pickle, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, partial_dependence

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_style, COLORS, ELEV_COLORS
import train_evaluate as T

ROOT = Path(__file__).resolve().parents[1]
FIG, TAB, MOD = ROOT / "outputs" / "figures", ROOT / "outputs" / "tables", ROOT / "models"

NICE = {
    "elevation": "Elevation", "next_month": "Month being forecast", "ret_1m": "Last month's change",
    "ret_12m": "Change over 12 months", "dev_12m_mean": "Price vs 12-month average",
    "premium_vs_total": "Premium vs national average", "fx_chg_3m": "Rupee change, 3 months",
    "brent_chg_3m": "Brent oil change, 3 months", "world_tea_chg_3m": "World tea price change, 3 months",
    "ceylon_premium": "Ceylon premium over world tea", "rain_anom_2m": "Rainfall anomaly, 2 months",
    "temp_anom": "Temperature anomaly", "solar_anom": "Sunshine anomaly", "fertilizer_ban": "Fertiliser-ban period",
}
UNITS = {"ret_1m": "log change", "ret_12m": "log change", "dev_12m_mean": "log ratio", "premium_vs_total": "log ratio",
         "fx_chg_3m": "log change (+ = rupee weaker)", "brent_chg_3m": "log change", "world_tea_chg_3m": "log change",
         "ceylon_premium": "log(Colombo / world price)", "rain_anom_2m": "mm/day vs normal", "temp_anom": "deg C vs normal",
         "solar_anom": "MJ/m2/day vs normal"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def nice(t):
    if " x " in t:
        a, b = t.split(" x ")
        return f"{NICE.get(a, a)} x {NICE.get(b, b)}"
    return NICE.get(t, t)


def main():
    apply_style()
    df = T.load_panel()
    lab = df.dropna(subset=["target"])
    trval = T.fit_rows(lab[lab["target_month"] <= "2024-12"])
    test = lab[lab["target_month"] >= "2025-01"]
    ebm = pickle.load(open(MOD / "ebm_trainval.pkl", "rb"))
    drivers = pickle.load(open(MOD / "ebm_drivers_trainval.pkl", "rb"))
    deploy = pickle.load(open(MOD / "ebm_deploy.pkl", "rb"))
    best = json.load(open(TAB / "best_params.json"))
    F = T.FEATURES

    # 1. global importance ---------------------------------------------------------
    imp = pd.Series(ebm.term_importances(trval[F].values)).sort_values()
    imp_d = pd.Series(drivers.term_importances(trval[T.DRIVER_FEATURES].values)).sort_values()
    pd.DataFrame({"full_model": imp, "drivers_only": imp_d}).mul(100).to_csv(TAB / "term_importance_pct.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, s, title in [(axes[0], imp, "Full EBM"), (axes[1], imp_d, "Drivers-only EBM (no price history)")]:
        s = s * 100
        col = [COLORS["primary"] if v >= s.quantile(0.7) else COLORS["muted"] for v in s]
        ax.barh([nice(i) for i in s.index], s.values, color=col)
        ax.set_xlabel("Mean |contribution| to forecast (percentage points)")
        ax.set_title(title, loc="left")
    fig.suptitle("Which terms move the forecast most (EBM global importance, training data)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "global_importance.png", dpi=180); plt.close(fig)

    # 2. shape functions ------------------------------------------------------------
    cont = [f for f in imp.sort_values(ascending=False).index if f in F and f not in T.CATEGORICAL][:8]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    for ax, f in zip(axes.ravel(), cont):
        sf = ebm.shape_function(f)
        j = F.index(f)
        x = trval[f].values
        edges = sf["edges"]
        lo, hi = np.percentile(x, [1, 99])
        grid = np.concatenate([[lo], edges[(edges > lo) & (edges < hi)], [hi]])
        # step function over bins
        xs, ys, ss = [], [], []
        for a, b in zip(grid[:-1], grid[1:]):
            k = np.searchsorted(edges, (a + b) / 2, side="right")
            xs += [a, b]; ys += [sf["score"][k] * 100] * 2; ss += [sf["std"][k] * 100] * 2
        xs, ys, ss = map(np.array, (xs, ys, ss))
        ax.fill_between(xs, ys - ss, ys + ss, color=COLORS["primary"], alpha=0.15, lw=0)
        ax.plot(xs, ys, color=COLORS["primary"], lw=2)
        ax.axhline(0, color=COLORS["ink"], lw=0.7)
        ax.plot(x, np.full_like(x, ax.get_ylim()[0]), "|", color=COLORS["muted"], ms=6, alpha=0.5)
        ax.set_title(nice(f), loc="left", fontsize=9.5)
        ax.set_xlabel(UNITS.get(f, ""), fontsize=8)
    axes[0, 0].set_ylabel("Effect on next month's\nchange (pp)"); axes[1, 0].set_ylabel("Effect on next month's\nchange (pp)")
    fig.suptitle("EBM shape functions: how each feature pushes the forecast (band = +/-1 sd across 16 bags; ticks = data)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "shape_functions.png", dpi=180); plt.close(fig)

    # 3. seasonality by elevation (month main effect + elevation x month pair) ----------
    rows = []
    for e, code in T.ELEV_CODE.items():
        for m in range(1, 13):
            X = trval[F].iloc[[0]].copy()
            X["elevation"], X["next_month"] = code, m
            C, names = ebm.term_contributions(X.values)
            keep = [i for i, n in enumerate(names) if n in ("next_month", "elevation x next_month")]
            rows.append({"elevation": e, "month": m, "effect_pp": C[0, keep].sum() * 100})
    seas = pd.DataFrame(rows)
    seas.to_csv(TAB / "seasonal_effect_by_elevation.csv", index=False)
    obs = (trval.assign(m=pd.PeriodIndex(trval["target_month"], freq="M").month)
           .groupby(["elevation_name", "m"])["target"].mean().mul(100).rename("obs").reset_index())
    obs.to_csv(TAB / "observed_mean_change_by_month.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for e in T.ELEV_CODE:
        s = seas[seas.elevation == e]
        axes[0].plot(s.month, s.effect_pp, marker="o", ms=3.5, lw=2, color=ELEV_COLORS[e], label=f"{e.capitalize()} grown")
        o = obs[obs.elevation_name == e]
        axes[1].plot(o.m, o.obs, marker="o", ms=3.5, lw=2, color=ELEV_COLORS[e], label=f"{e.capitalize()} grown")
    for ax in axes:
        ax.axhline(0, color=COLORS["ink"], lw=0.7); ax.set_xticks(range(1, 13)); ax.set_xticklabels([m[0] for m in MONTHS])
        ax.axvspan(0.5, 3.5, color=COLORS["primary"], alpha=0.06); ax.axvspan(6.5, 9.5, color=COLORS["accent"], alpha=0.06)
    axes[0].set_title("What the EBM learned (month + elevation x month terms)", loc="left")
    axes[0].set_ylabel("Effect on forecast (pp)")
    axes[1].set_title("What the data show (mean change in training years)", loc="left")
    axes[1].set_ylabel("Mean monthly change (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    axes[1].text(2, axes[1].get_ylim()[0] * 0.55, "Western\nquality", ha="center", fontsize=7.5, color=COLORS["primary"])
    axes[1].text(8, axes[1].get_ylim()[0] * 0.55, "Uva\nquality", ha="center", fontsize=7.5, color=COLORS["accent"])
    fig.suptitle("Seasonality: the forecast month is the strongest single driver; June is the low point", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "seasonality.png", dpi=180); plt.close(fig)

    # 4. exact SHAP on the test set ------------------------------------------------------
    phi, base = ebm.shap_values(test[F].values, background=trval[F].values)
    assert np.allclose(phi.sum(1) + base, ebm.predict(test[F].values), atol=1e-10)
    shap_imp = pd.Series(np.abs(phi).mean(0) * 100, index=F).sort_values()
    shap_imp.to_csv(TAB / "shap_mean_abs_test_pp.csv")
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    order = shap_imp.index
    for i, f in enumerate(order):
        j = F.index(f)
        v = test[f].values.astype(float)
        rk = (pd.Series(v).rank(pct=True).values if len(np.unique(v)) > 2 else (v - v.min()) / max(np.ptp(v), 1e-9))
        jit = (np.random.default_rng(j).random(len(v)) - 0.5) * 0.5
        sc = ax.scatter(phi[:, j] * 100, i + jit, c=rk, cmap="viridis", s=12, alpha=0.85, lw=0)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([nice(f) for f in order])
    ax.axvline(0, color=COLORS["ink"], lw=0.7)
    ax.set_xlabel("SHAP value: contribution to the forecast change (percentage points)")
    cb = fig.colorbar(sc, ax=ax, pad=0.01); cb.set_label("Feature value (low to high, rank)"); cb.outline.set_visible(False)
    ax.set_title("Exact SHAP values on the 2025-26 test set (63 forecasts)", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "shap_summary.png", dpi=180); plt.close(fig)

    # local explanation: the live forecast for October 2026 (deploy model, Sep-2026 data)
    latest = df[df["month"] == df["month"].max()]
    local = []
    for _, r in latest.iterrows():
        x = r[F].values.astype(float)[None, :]
        ph, b0 = deploy.shap_values(x, background=T.fit_rows(lab)[F].values)
        pred = float(deploy.predict(x)[0])
        local.append({"elev": r["elevation_name"], "origin": r["month"], "pred_change_pct": pred * 100,
                      "usd_t": r["usd_price_t"], "usd_forecast": r["usd_price_t"] * np.exp(pred),
                      "lkr_t": r["lkr_price_t"], "lkr_forecast_at_current_fx": r["lkr_price_t"] * np.exp(pred),
                      **{f: ph[0, i] * 100 for i, f in enumerate(F)}, "base_pct": b0 * 100})
    local = pd.DataFrame(local)
    local.to_csv(TAB / "live_forecast_oct2026.csv", index=False)
    r = local[local.elev == "high"].iloc[0]
    contrib = pd.Series({f: r[f] for f in F})
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:8]
    rest = contrib.drop(top.index).sum()
    steps = list(top.items()) + [("Other 6 features", rest)]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    cum = r["base_pct"]
    ax.barh(0, cum, color=COLORS["muted"]); ax.text(cum, 0, f"  {cum:+.2f}", va="center", fontsize=8)
    labels = ["Average change (baseline)"]
    for i, (f, v) in enumerate(steps, start=1):
        ax.barh(i, v, left=cum, color=COLORS["primary"] if v > 0 else COLORS["negative"])
        ax.text(max(cum, cum + v), i, f"  {v:+.2f}", va="center", fontsize=8)
        cum += v; labels.append(nice(f))
    ax.barh(len(steps) + 1, cum, color=COLORS["ink"]); ax.text(cum, len(steps) + 1, f"  {cum:+.2f}%", va="center", fontsize=8, fontweight="bold")
    labels.append("Forecast change")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.axvline(0, color=COLORS["ink"], lw=0.7); ax.set_xlabel("Percentage points"); ax.margins(x=0.12)
    ax.set_title(f"Why the model forecasts {cum:+.2f}% for High grown in October 2026 (from September data)", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "local_waterfall_high_oct2026.png", dpi=180); plt.close(fig)

    # 5. PDP: EBM vs random forest --------------------------------------------------------
    rf = T.make_model("RandomForest", best["RandomForest"], F).fit(trval[F], trval["target"].values)
    pdp_feats = ["ceylon_premium", "world_tea_chg_3m", "rain_anom_2m", "fx_chg_3m"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, f in zip(axes, pdp_feats):
        for model, name, col in [(ebm, "EBM", COLORS["primary"]), (rf, "Random forest", COLORS["accent"])]:
            pdres = partial_dependence(model, trval[F], [F.index(f)], grid_resolution=30, percentiles=(0.05, 0.95))
            g = pdres["grid_values"][0]; a = pdres["average"][0] * 100
            ax.plot(g, a - a.mean(), color=col, lw=2, label=name)
        ax.axhline(0, color=COLORS["ink"], lw=0.7)
        ax.set_title(nice(f), loc="left", fontsize=9.5); ax.set_xlabel(UNITS.get(f, ""), fontsize=8)
    axes[0].set_ylabel("Partial dependence (pp, centred)"); axes[0].legend(frameon=False)
    fig.suptitle("Partial dependence: two different model families learn the same direction for each driver", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "pdp_ebm_vs_rf.png", dpi=180); plt.close(fig)

    # 6. permutation importance on the test set ---------------------------------------------
    pi = permutation_importance(ebm, test[F], test["target"].values, n_repeats=30, random_state=0,
                                scoring="neg_root_mean_squared_error")
    pd.DataFrame({"feature": F, "rmse_increase_pp": pi.importances_mean * 100, "sd": pi.importances_std * 100}) \
        .sort_values("rmse_increase_pp", ascending=False).to_csv(TAB / "permutation_importance_test.csv", index=False)

    # print a compact summary for the report
    print("Top terms (full):\n", imp.sort_values(ascending=False).mul(100).round(3).head(10).to_string())
    print("Top terms (drivers):\n", imp_d.sort_values(ascending=False).mul(100).round(3).head(8).to_string())
    print("SHAP mean |phi| test:\n", shap_imp.sort_values(ascending=False).round(3).head(8).to_string())
    print(pd.read_csv(TAB / "permutation_importance_test.csv").round(3).head(8).to_string(index=False))
    print(seas.pivot(index="month", columns="elevation", values="effect_pp").round(3).T.to_string())
    print(local[["elev", "pred_change_pct", "usd_t", "usd_forecast", "lkr_t", "lkr_forecast_at_current_fx", "base_pct"]].round(3).to_string(index=False))
    print("pairs chosen:", ebm.pairs_, "intercept", ebm.intercept_)


if __name__ == "__main__":
    main()
