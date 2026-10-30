"""
Build the modelling panel for the Colombo Tea Auction price-forecasting project.

Unit of analysis : one row per (month t, elevation e), e in {High, Medium, Low} grown.
Target           : y = log( USD_price[e, t+1] / USD_price[e, t] )
                   i.e. next month's % change in the elevation's average auction
                   price, expressed in US dollars per kg.
Why USD?         : The 2022 Sri Lankan currency crisis roughly doubled rupee prices
                   in a few months. Modelling in USD separates the "tea market"
                   signal from the currency collapse; the exchange rate is kept
                   as an explanatory feature instead.

All features are computed from information available at the end of month t,
so there is no look-ahead leakage.

Raw inputs (data/raw/):
  prices_monthly_lkr.csv      Forbes & Walker / Sri Lanka Tea Board elevation averages (LKR/kg)
  prices_monthly_usd.csv      Published USD averages (used only to validate the FX conversion)
  fx_usd_lkr_xrates.csv       Monthly average USD/LKR (x-rates.com)
  fred_brent_worldtea.csv     Brent crude (FRED MCOILBRENTEU) and IMF global tea price (FRED PTEAUSDM)
  weather_nasa_power_raw.csv  NASA POWER monthly weather at Nuwara Eliya / Kandy / Ratnapura
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

ELEVATIONS = ["high", "medium", "low"]
CLIMATOLOGY_END = "2022-12"   # weather "normals" use training-period years only (no leakage)


def load_prices():
    p = pd.read_csv(RAW / "prices_monthly_lkr.csv")
    p["month"] = pd.PeriodIndex(p["month"], freq="M")
    return p.set_index("month").drop(columns="source")


def load_fx():
    fx = pd.read_csv(RAW / "fx_usd_lkr_xrates.csv")
    fx["month"] = pd.PeriodIndex(fx["month"], freq="M")
    return fx.set_index("month")["usd_lkr"]


def load_macro():
    m = pd.read_csv(RAW / "fred_brent_worldtea.csv")
    m["month"] = pd.PeriodIndex(m["month"], freq="M")
    return m.set_index("month")


def load_weather():
    w = pd.read_csv(RAW / "weather_nasa_power_raw.csv")
    # Data-quality fix: NASA POWER's near-real-time daily feed returned 2026
    # precipitation scaled by 24 (mm/hour * 24). Verified on the Dec-2025 overlap,
    # where daily/24 reproduces the official monthly value exactly.
    scaled = w["note"].str.startswith("daily_x24")
    w.loc[scaled, "prec_mm_day"] = w.loc[scaled, "prec_mm_day"] / 24.0
    w = w[~w["note"].str.contains("partial")]          # drop incomplete Oct-2026
    w["month"] = pd.PeriodIndex(w["month"], freq="M")
    return w.drop(columns="note")


def validate_fx(prices, fx):
    """Check: LKR price / FX should reproduce the USD averages published by the Tea Board."""
    usd = pd.read_csv(RAW / "prices_monthly_usd.csv")
    usd["month"] = pd.PeriodIndex(usd["month"], freq="M")
    usd = usd.set_index("month")
    rows = []
    for e in ELEVATIONS + ["total"]:
        pub = usd[e].dropna()
        conv = (prices[e] / fx).reindex(pub.index)
        err = (conv - pub) / pub
        rows.append({"series": e, "n_months": len(pub),
                     "mean_abs_pct_err": float(err.abs().mean() * 100),
                     "max_abs_pct_err": float(err.abs().max() * 100)})
    return pd.DataFrame(rows)


def build():
    prices, fx, macro, wx = load_prices(), load_fx(), load_macro(), load_weather()
    usd = prices.div(fx, axis=0).dropna()            # USD/kg for each elevation + total

    # weather anomalies vs calendar-month normals (training years only)
    wx["cal_month"] = wx["month"].dt.month
    normals = (wx[wx["month"] <= pd.Period(CLIMATOLOGY_END, "M")]
               .groupby(["loc", "cal_month"])[["prec_mm_day", "t2m", "solar"]].mean())
    wx = wx.join(normals, on=["loc", "cal_month"], rsuffix="_norm")
    wx["rain_anom"] = wx["prec_mm_day"] - wx["prec_mm_day_norm"]
    wx["temp_anom"] = wx["t2m"] - wx["t2m_norm"]
    wx["solar_anom"] = wx["solar"] - wx["solar_norm"]
    wx = wx.set_index(["loc", "month"]).sort_index()

    rows = []
    months = usd.index
    for e in ELEVATIONS:
        s = np.log(usd[e])
        tot = np.log(usd["total"])
        w = wx.loc[e]
        for t in months:
            t1 = t + 1
            need = [t - 12, t - 3, t - 1, t]
            if any(m not in s.index for m in need):
                continue
            r = {
                "month": str(t),               # information month (forecast origin)
                "target_month": str(t1),
                "elevation": e,
                # --- price dynamics (own elevation) ---
                "ret_1m": s[t] - s[t - 1],
                "ret_3m": s[t] - s[t - 3],
                "ret_12m": s[t] - s[t - 12],
                "dev_12m_mean": s[t] - np.log(np.exp(s.loc[t - 11:t]).mean()),
                "premium_vs_total": s[t] - tot[t],
                # --- seasonality ---
                "next_month": t1.month,
                # --- currency ---
                "fx_chg_1m": np.log(fx[t] / fx[t - 1]),
                "fx_chg_3m": np.log(fx[t] / fx[t - 3]),
                # --- global demand / substitutes ---
                "brent_usd": macro.loc[t, "brent_usd_bbl"],
                "brent_chg_3m": np.log(macro.loc[t, "brent_usd_bbl"] / macro.loc[t - 3, "brent_usd_bbl"]),
                "world_tea_chg_3m": np.log(macro.loc[t, "tea_world_usc_kg"] / macro.loc[t - 3, "tea_world_usc_kg"]),
                "ceylon_premium": np.log(usd.loc[t, "total"] * 100 / macro.loc[t, "tea_world_usc_kg"]),
                # --- weather in the elevation's growing district ---
                "rain_anom": w.loc[t, "rain_anom"],
                "rain_anom_2m": (w.loc[t, "rain_anom"] + w.loc[t - 1, "rain_anom"]) / 2,
                "temp_anom": w.loc[t, "temp_anom"],
                "solar_anom": w.loc[t, "solar_anom"],
                # --- documented shocks ---
                "fertilizer_ban": int(pd.Period("2021-05", "M") <= t <= pd.Period("2021-11", "M")),
                "econ_crisis": int(pd.Period("2022-03", "M") <= t <= pd.Period("2022-12", "M")),
                # --- bookkeeping (not features) ---
                "usd_price_t": usd.loc[t, e],
                "lkr_price_t": prices.loc[t, e],
                "fx_t": fx[t],
            }
            if t1 in usd.index:
                r["usd_price_t1"] = usd.loc[t1, e]
                r["lkr_price_t1"] = prices.loc[t1, e]
                r["target"] = s[t1] - s[t]
            rows.append(r)
    panel = pd.DataFrame(rows).sort_values(["month", "elevation"]).reset_index(drop=True)
    return panel, validate_fx(prices, fx), usd


if __name__ == "__main__":
    panel, fxcheck, usd = build()
    panel.to_csv(OUT / "tea_panel.csv", index=False)
    usd.to_csv(OUT / "usd_prices_monthly.csv")
    fxcheck.to_csv(OUT / "fx_validation.csv", index=False)
    print(panel.shape, "rows with target:", panel["target"].notna().sum())
    print(panel["month"].min(), "->", panel["month"].max())
    print(fxcheck.round(2).to_string(index=False))
    print(panel.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
