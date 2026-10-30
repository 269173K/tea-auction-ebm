// Checks that the browser twin of the EBM reproduces the Python model exactly.
// Run: node app/test_parity.js
const fs = require("fs"), path = require("path");
const { Engine } = require("./static/app.js");
const bundle = JSON.parse(fs.readFileSync(path.join(__dirname, "static/model_bundle.json")));
const fx = JSON.parse(fs.readFileSync(path.join(__dirname, "parity_fixture.json")));
const eng = Engine(bundle.model), F = bundle.model.feature_order;
let maxPred = 0, maxPhi = 0, maxBase = 0;
fx.rows.forEach((row, i) => {
  const r = eng.explain(row);
  maxPred = Math.max(maxPred, Math.abs(r.pred - fx.pred[i]));
  maxBase = Math.max(maxBase, Math.abs(r.base - fx.base));
  F.forEach((f, j) => { maxPhi = Math.max(maxPhi, Math.abs(r.phi[f] - fx.phi[i][j])); });
});
console.log(`rows ${fx.rows.length}  max |pred diff| ${maxPred.toExponential(2)}  max |SHAP diff| ${maxPhi.toExponential(2)}  max |base diff| ${maxBase.toExponential(2)}`);
process.exit(maxPred < 1e-9 && maxPhi < 1e-9 ? 0 : 1);
