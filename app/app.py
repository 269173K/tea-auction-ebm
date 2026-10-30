"""
Tea auction price outlook - Flask front-end for the trained EBM.

Run:   python app/app.py        then open http://127.0.0.1:5000
API:
  GET  /api/bundle    feature metadata, latest inputs, price history, shapes, metrics
  POST /api/predict   {"elevation": "high", "features": {...model-unit values...}, "fx": 329.1}
                      -> forecast in USD and LKR + exact SHAP contribution of every feature
"""
from pathlib import Path
import json, pickle, sys
import numpy as np
from flask import Flask, jsonify, request, render_template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ebm  # noqa: F401  (needed to unpickle the EBMRegressor class)
import train_evaluate as T

app = Flask(__name__, root_path=str(Path(__file__).resolve().parent),
            static_folder="static", template_folder="templates")
BUNDLE = json.load(open(ROOT / "app" / "static" / "model_bundle.json"))
_lab = T.load_panel().dropna(subset=["target"])


def _load_model():
    """Load the trained model. If the pickle can't be read on this machine (e.g. a cloud
    host with different NumPy/Python versions), refit the identical deployment model from
    the committed data and tuned hyperparameters; the fit is deterministic and takes seconds."""
    try:
        return pickle.load(open(ROOT / "models" / "ebm_deploy.pkl", "rb"))
    except Exception as exc:  # noqa: BLE001
        print(f"Model pickle not loadable here ({exc!r}); refitting from data.", flush=True)
        best = json.load(open(ROOT / "outputs" / "tables" / "best_params.json"))["EBM"]
        rows = T.fit_rows(_lab)
        return T.make_model("EBM", best, T.FEATURES).fit(rows[T.FEATURES], rows["target"].values)


MODEL = _load_model()
BACKGROUND = T.fit_rows(_lab)[T.FEATURES].values.astype(float)
FEATURES = T.FEATURES
ELEV_CODE = T.ELEV_CODE
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/bundle")
def bundle():
    return jsonify(BUNDLE)


@app.post("/api/predict")
def predict():
    body = request.get_json(force=True)
    elev = body.get("elevation")
    if elev not in ELEV_CODE:
        return jsonify({"error": "elevation must be one of high, medium, low"}), 400
    feats = dict(BUNDLE["state"][elev]["features"])          # start from the latest observed values
    feats.update({k: float(v) for k, v in body.get("features", {}).items() if k in FEATURES})
    feats["elevation"] = ELEV_CODE[elev]
    month = int(feats["next_month"])
    if not 1 <= month <= 12:
        return jsonify({"error": "next_month must be 1-12"}), 400
    x = np.array([[feats[f] for f in FEATURES]], dtype=float)
    pred = float(MODEL.predict(x)[0])
    phi, base = MODEL.shap_values(x, background=BACKGROUND)
    st = BUNDLE["state"][elev]
    fx = float(body.get("fx") or st["fx_t"])
    usd_f = st["usd_t"] * np.exp(pred)
    return jsonify({
        "elevation": elev, "month_name": MONTHS[month - 1],
        "pred_change_pct": (np.exp(pred) - 1) * 100,
        "usd_now": st["usd_t"], "usd_forecast": usd_f,
        "lkr_now": st["lkr_t"], "lkr_forecast": usd_f * fx, "fx": fx,
        "base_pct": base * 100,
        "contributions": sorted([{"feature": f, "pp": float(phi[0, i] * 100)} for i, f in enumerate(FEATURES)],
                                key=lambda c: -abs(c["pp"])),
        "engine": "flask",
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
