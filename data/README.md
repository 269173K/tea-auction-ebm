# Colombo Tea Auction Monthly Dataset (2018–2026)

**Author:** Rajapakse V.O.V (269173K) · MSc in Artificial Intelligence, Machine Learning module
**Compiled:** October 2026 · **Coverage:** January 2018 to September 2026 (monthly)

A dataset for forecasting next month's Colombo Tea Auction average price for each tea-growing
elevation (High, Medium, Low grown), with market, currency and weather drivers. It was compiled
by hand from public sources for this assignment; no single public file contains it.

## Files

| File | Rows | What it holds |
| --- | --- | --- |
| `processed/tea_panel.csv` | 279 | **The modelling dataset.** One row per month × elevation (Jan 2019 – Sep 2026), 19 features and the target |
| `raw/prices_monthly_lkr.csv` | 105 | Monthly auction averages by elevation in rupees/kg (Jan 2018 – Sep 2026), with the source of every row |
| `raw/prices_monthly_usd.csv` | 51 | USD averages published by the Tea Board, used only to validate the currency conversion |
| `raw/fx_usd_lkr_xrates.csv` | 117 | Monthly average USD/LKR exchange rate |
| `raw/fred_brent_worldtea.csv` | 117 | Brent crude oil (US$/barrel) and IMF global tea price (US cents/kg) |
| `raw/weather_nasa_power_raw.csv` | 354 | Monthly rainfall, temperature, humidity and solar radiation at one site per elevation |
| `processed/usd_prices_monthly.csv` | 105 | Auction averages converted to US$/kg |
| `processed/fx_validation.csv` | 4 | Check of converted USD prices against the Tea Board's published USD prices |

## How it was obtained

| Data | Source | Method |
| --- | --- | --- |
| Auction averages by elevation | Forbes & Walker Tea Brokers, compiling Sri Lanka Tea Board data | 2018–2022 from the monthly "Price Movement" tables in the F&W Tea Review 2020, 2021 and 2022 (forbestea.com). 2023 – Apr 2026 from 30 monthly "National Tea Sale Averages" reports, each also restating the previous month and the same month a year earlier. May–Sep 2026 from press reports of the same releases (Daily Mirror, Daily FT). Values were read from the PDFs and typed into CSV |
| USD/LKR exchange rate | x-rates.com monthly averages | Read from the yearly average tables |
| Brent crude oil | FRED series MCOILBRENTEU (Federal Reserve Bank of St. Louis) | Downloaded as CSV |
| Global tea price | FRED series PTEAUSDM (IMF Primary Commodity Prices) | Downloaded as CSV |
| Weather | NASA POWER API (power.larc.nasa.gov), community "AG" | Point data at Nuwara Eliya (6.97 N, 80.77 E) for High grown, Kandy (7.29 N, 80.63 E) for Medium grown, Ratnapura (6.68 N, 80.40 E) for Low grown. Monthly endpoint for 2017–2025; 2026 aggregated from the daily endpoint |

## Data quality checks and corrections

- **Overlapping reports agree.** Most months appear in two or three reports; later restatements differed by at most about Rs 0.05. The month's own report was used.
- **Elevation labels verified.** Two unlabelled tables in the 2021 Tea Review were identified by matching overlapping years against the fully labelled 2020 review.
- **Derived months.** May, July and August 2026 were reconstructed from the next published month and the stated month-on-month change (marked "Derived" in the `source` column).
- **Currency conversion validated.** Rupee price ÷ monthly USD/LKR rate reproduces the Tea Board's published USD prices with a mean error of 0.48% (max 1.15%) over 47 months (`fx_validation.csv`).
- **Unit error fixed.** NASA POWER's near-real-time 2026 rainfall was scaled by 24. Dividing by 24 reproduces the official December 2025 monthly value exactly; the raw file keeps the original values, flagged `daily_x24` in the `note` column, and the correction is applied in `src/build_dataset.py`. October 2026 (20 days only) is excluded.

## Column dictionary: `processed/tea_panel.csv`

All features use information available at the end of `month`, so there is no look-ahead. "log change" = log(value now / value then).

| Column | Meaning | Unit |
| --- | --- | --- |
| `month` | Month whose data is known (forecast origin) | YYYY-MM |
| `target_month` | Month being forecast (`month` + 1) | YYYY-MM |
| `elevation` | Tea segment | high / medium / low |
| `ret_1m`, `ret_3m`, `ret_12m` | Change in this elevation's USD price over 1, 3, 12 months | log change |
| `dev_12m_mean` | Price vs its 12-month average | log ratio |
| `premium_vs_total` | Price vs the national (all-elevation) average | log ratio |
| `next_month` | Calendar month of `target_month` | 1–12 |
| `fx_chg_1m`, `fx_chg_3m` | Change in USD/LKR (positive = rupee weaker) | log change |
| `brent_usd` | Brent crude price | US$/barrel |
| `brent_chg_3m` | Brent change over 3 months | log change |
| `world_tea_chg_3m` | IMF global tea price change over 3 months | log change |
| `ceylon_premium` | Colombo national average vs IMF global tea price | log ratio |
| `rain_anom`, `rain_anom_2m` | Rainfall vs the 2017–2022 average for that calendar month; last month, and mean of last 2 months | mm/day |
| `temp_anom` | Temperature vs calendar-month average | °C |
| `solar_anom` | Solar radiation vs calendar-month average | MJ/m²/day |
| `fertilizer_ban` | 1 for May–Nov 2021 (chemical fertiliser ban) | 0/1 |
| `econ_crisis` | 1 for Mar–Dec 2022 (currency crisis) | 0/1 |
| `usd_price_t`, `lkr_price_t` | This elevation's auction average in `month` | US$/kg, Rs/kg |
| `fx_t` | USD/LKR in `month` | Rs per US$ |
| `usd_price_t1`, `lkr_price_t1` | Actual auction average in `target_month` (blank for the latest month) | US$/kg, Rs/kg |
| `target` | **Prediction target:** log(`usd_price_t1` / `usd_price_t`) | log change |

## Use and attribution

All sources are public and aggregated; the dataset contains no personal, estate-level or
confidential information. The original reports and data remain the property of their publishers
(Forbes & Walker Tea Brokers / Sri Lanka Tea Board, Federal Reserve Bank of St. Louis / IMF,
x-rates.com, NASA Langley Research Center POWER Project); this dataset redistributes only the
extracted monthly figures, for academic use, with those sources credited.

To rebuild the processed files from the raw ones: `python src/build_dataset.py`.
