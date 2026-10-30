// Smoke test: runs the page script against a minimal mock DOM (no server, embedded bundle)
// and checks every panel renders without errors. Run: node app/test_render.js
const fs = require("fs"), path = require("path"), vm = require("vm");
class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.listeners = {}; this.style = { setProperty() {} };
    this.dataset = {}; this.classList = { toggle() {}, add() {} }; this._text = ""; this.value = ""; this.options = []; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k]; }
  appendChild(c) { this.children.push(c); return c; }
  addEventListener(t, f) { this.listeners[t] = f; }
  add(o) { this.options.push(o); }
  set textContent(t) { this._text = String(t); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  set innerHTML(h) { this._html = h; this.children = []; }
  get innerHTML() { return this._html || ""; }
  querySelector() { return new Node("input"); }
  focus() {}
  count(tag) { return (this.tag === tag ? 1 : 0) + this.children.reduce((s, c) => s + c.count(tag), 0); }
}
const nodes = {};
const document = {
  querySelector: (s) => (nodes[s] ||= new Node(s)),
  querySelectorAll: () => [],
  createElement: (t) => new Node(t),
  createElementNS: (_, t) => new Node(t),
};
const bundle = JSON.parse(fs.readFileSync(path.join(__dirname, "static/model_bundle.json")));
const errors = [];
const ctx = { document, window: { EMBEDDED_BUNDLE: bundle, addEventListener() {} }, console,
  fetch: () => Promise.reject(new Error("no server")), Option: function (t, v) { this.text = t; this.value = v; },
  setTimeout, clearTimeout, Math, Number, Object, JSON, Promise };
ctx.window.document = document;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, "static/app.js"), "utf8") + "\n;globalThis.__boot = boot; globalThis.__state = () => ({ elev, ui }); globalThis.__shape = renderShape; globalThis.__upd = update;", ctx);
(async () => {
  try { await ctx.__boot(); } catch (e) { errors.push(e.stack); }
  const show = (id) => nodes[id];
  console.log("title     :", show("#fc-title").textContent);
  console.log("price Rs  :", show("#fc-lkr").textContent, "|", show("#fc-change").textContent);
  console.log("sub       :", show("#fc-sub").textContent);
  console.log("why rects :", show("#why").count("rect"), " history paths:", show("#history").count("path"), " shape rects:", show("#shape").count("rect"));
  console.log("why labels:", show("#why").children[0].children.filter((c) => c.tag === "text").map((c) => c._text).slice(0, 20).join(" | "));
  // switch the shape view to a continuous feature and a what-if month
  const st = ctx.__state(); st.ui.next_month = 6;
  for (const f of ["rain_anom_2m", "ceylon_premium", "brent_chg_3m", "ret_1m", "next_month"]) {
    nodes["#shape-feature"].value = f;
    try { await ctx.__upd(); } catch (e) { errors.push(f + ": " + e.stack); }
    console.log("shape", f.padEnd(15), "paths", nodes["#shape"].count("path"), "circles", nodes["#shape"].count("circle"), "|", nodes["#shape-note"].textContent.slice(0, 70));
  }
  console.log("June what-if:", nodes["#fc-title"].textContent, nodes["#fc-change"].textContent);
  console.log("errors    :", errors.length ? errors : "none");
  process.exit(errors.length ? 1 : 0);
})();
