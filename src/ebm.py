"""
A from-scratch Explainable Boosting Machine (EBM) regressor in NumPy.

EBM (Lou, Caruana & Gehrke 2012; Lou et al. 2013; Nori et al. 2019, InterpretML)
is a Generalised Additive Model with pairwise interactions (GA2M):

    y_hat = b0 + sum_j f_j(x_j) + sum_{(i,j) in S} f_ij(x_i, x_j)

Each shape function f_j is learned by *cyclic gradient boosting*: in every round,
the algorithm visits the features one at a time and fits a very small tree
(a few leaves) on ONE feature to the current residuals, with a tiny learning rate.
Because each tree only ever sees one feature, the order in which features are
visited does not matter much and the learned function stays additive, so every
f_j can be plotted and read directly: the model *is* its own explanation.

Implementation details (mirroring InterpretML's defaults where sensible):
  * features are quantile-binned (categoricals get one bin per level)
  * per-feature trees with up to `max_leaves` leaves, chosen greedily on bins
  * early stopping on an inner validation split
  * `outer_bags` bootstrap-style bags; terms are averaged across bags, and the
    spread across bags gives an uncertainty band for every shape function
  * interaction detection with the FAST heuristic (Lou et al. 2013): for each
    pair, the best single (cut_i, cut_j) 2x2 split of the main-effect residuals
  * pair terms are boosted with 2x2 "FAST trees" after the main effects
  * every term is centred to mean zero on the training data; the intercept
    carries the mean.

The class follows the scikit-learn estimator API (fit / predict / get_params),
so sklearn tools such as permutation_importance and partial_dependence work.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


# ---------------------------------------------------------------- binning
class _Binner:
    def __init__(self, max_bins: int, categorical: bool):
        self.max_bins, self.categorical = max_bins, categorical

    def fit(self, x):
        x = np.asarray(x, float)
        if self.categorical:
            self.levels_ = np.unique(x)
            self.edges_ = None
        else:
            qs = np.quantile(x, np.linspace(0, 1, self.max_bins + 1)[1:-1])
            self.edges_ = np.unique(qs)
        return self

    @property
    def n_bins(self):
        return len(self.levels_) if self.categorical else len(self.edges_) + 1

    def transform(self, x):
        x = np.asarray(x, float)
        if self.categorical:
            idx = np.searchsorted(self.levels_, x)
            idx = np.clip(idx, 0, len(self.levels_) - 1)
            return idx
        return np.searchsorted(self.edges_, x, side="right")

    def bin_labels(self):
        if self.categorical:
            return [f"{v:g}" for v in self.levels_]
        e = self.edges_
        lab = [f"< {e[0]:.3g}"] if len(e) else ["all"]
        lab += [f"[{e[k]:.3g}, {e[k+1]:.3g})" for k in range(len(e) - 1)]
        if len(e):
            lab.append(f">= {e[-1]:.3g}")
        return lab

    def bin_centers(self, x_train):
        """Representative x value of each bin (median of training points in it)."""
        b = self.transform(x_train)
        out = np.zeros(self.n_bins)
        for k in range(self.n_bins):
            m = b == k
            if m.any():
                out[k] = np.median(np.asarray(x_train, float)[m])
            elif self.categorical:
                out[k] = self.levels_[k]
            else:
                lo = self.edges_[k - 1] if k > 0 else self.edges_[0]
                hi = self.edges_[k] if k < len(self.edges_) else self.edges_[-1]
                out[k] = (lo + hi) / 2
        return out


# ---------------------------------------------------------------- small trees on bins
def _best_split(g, h, lo, hi, min_leaf):
    """Best single cut in bins [lo, hi) for residual sums g and counts h.
    Returns (gain, cut) where the left leaf is [lo, cut)."""
    G, H = g[lo:hi].sum(), h[lo:hi].sum()
    if H < 2 * min_leaf:
        return 0.0, None
    cg, ch = np.cumsum(g[lo:hi])[:-1], np.cumsum(h[lo:hi])[:-1]
    ok = (ch >= min_leaf) & (H - ch >= min_leaf)
    if not ok.any():
        return 0.0, None
    gain = np.where(ok, cg**2 / np.maximum(ch, 1e-12) + (G - cg)**2 / np.maximum(H - ch, 1e-12) - G**2 / H, -np.inf)
    k = int(np.argmax(gain))
    return float(gain[k]), lo + k + 1


def _fit_bin_tree(g, h, max_leaves, min_leaf):
    """Greedy tree on ordered bins with up to max_leaves leaves.
    Returns per-bin update values (mean residual of the leaf each bin falls in)."""
    nb = len(g)
    segments = [(0, nb)]
    while len(segments) < max_leaves:
        best = (0.0, None, None)
        for si, (lo, hi) in enumerate(segments):
            gain, cut = _best_split(g, h, lo, hi, min_leaf)
            if cut is not None and gain > best[0]:
                best = (gain, si, cut)
        if best[1] is None:
            break
        lo, hi = segments.pop(best[1])
        segments += [(lo, best[2]), (best[2], hi)]
    upd = np.zeros(nb)
    for lo, hi in segments:
        H = h[lo:hi].sum()
        if H > 0:
            upd[lo:hi] = g[lo:hi].sum() / H
    return upd


def _fit_pair_tree(r, bi, bj, ni, nj, min_leaf):
    """FAST 2x2 tree for a pair: best cut on i and cut on j (4 leaves).
    Returns (gain, ni x nj update table)."""
    G = np.zeros((ni, nj)); H = np.zeros((ni, nj))
    np.add.at(G, (bi, bj), r); np.add.at(H, (bi, bj), 1.0)
    cG = G.cumsum(0).cumsum(1); cH = H.cumsum(0).cumsum(1)
    Gt, Ht = cG[-1, -1], cH[-1, -1]
    best, arg = 0.0, None
    for a in range(ni - 1):
        for b in range(nj - 1):
            g00, h00 = cG[a, b], cH[a, b]
            g0_, h0_ = cG[a, -1], cH[a, -1]
            g_0, h_0 = cG[-1, b], cH[-1, b]
            q = [(g00, h00), (g0_ - g00, h0_ - h00), (g_0 - g00, h_0 - h00),
                 (Gt - g0_ - g_0 + g00, Ht - h0_ - h_0 + h00)]
            if min(hh for _, hh in q) < min_leaf:
                continue
            gain = sum(gg**2 / hh for gg, hh in q) - Gt**2 / Ht
            if gain > best:
                best, arg = gain, (a, b, q)
    upd = np.zeros((ni, nj))
    if arg is None:
        return 0.0, upd
    a, b, q = arg
    upd[:a+1, :b+1] = q[0][0] / q[0][1]
    upd[:a+1, b+1:] = q[1][0] / q[1][1]
    upd[a+1:, :b+1] = q[2][0] / q[2][1]
    upd[a+1:, b+1:] = q[3][0] / q[3][1]
    return best, upd


# ---------------------------------------------------------------- the model
class EBMRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, feature_names=None, categorical=(), max_bins=16, max_pair_bins=6,
                 learning_rate=0.02, max_rounds=3000, max_leaves=3, min_samples_leaf=5,
                 early_stopping_rounds=60, validation_size=0.15, outer_bags=8,
                 interactions=3, forced_interactions=(), random_state=0):
        self.feature_names = feature_names
        self.categorical = categorical
        self.max_bins = max_bins
        self.max_pair_bins = max_pair_bins
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.max_leaves = max_leaves
        self.min_samples_leaf = min_samples_leaf
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_size = validation_size
        self.outer_bags = outer_bags
        self.interactions = interactions
        self.forced_interactions = forced_interactions
        self.random_state = random_state

    # ---- helpers
    def _names(self, X):
        if self.feature_names is not None:
            return list(self.feature_names)
        if hasattr(X, "columns"):
            return list(X.columns)
        return [f"x{j}" for j in range(np.asarray(X).shape[1])]

    def _bin_all(self, X):
        X = np.asarray(X, float)
        return np.column_stack([b.transform(X[:, j]) for j, b in enumerate(self.binners_)])

    def _bin_pairs(self, X):
        X = np.asarray(X, float)
        return np.column_stack([b.transform(X[:, j]) for j, b in enumerate(self.pair_binners_)])

    def _boost_mains(self, B, y, tr, va, rng):
        nf = B.shape[1]
        terms = [np.zeros(b.n_bins) for b in self.binners_]
        F = np.full(len(y), y[tr].mean())
        best_loss, best_terms, best_F, stall = np.inf, None, None, 0
        for _ in range(self.max_rounds):
            for j in range(nf):
                r = y - F
                nb = self.binners_[j].n_bins
                g = np.bincount(B[tr, j], weights=r[tr], minlength=nb)
                h = np.bincount(B[tr, j], minlength=nb).astype(float)
                if self.binners_[j].categorical:
                    # nominal feature: order levels by mean residual before splitting
                    # (as InterpretML / LightGBM do), so a split can isolate any group
                    # of levels, e.g. June on its own, not only calendar-contiguous runs
                    order = np.argsort(np.where(h > 0, g / np.maximum(h, 1), 0.0), kind="stable")
                    upd_sorted = _fit_bin_tree(g[order], h[order], self.max_leaves, self.min_samples_leaf)
                    upd = np.empty(nb); upd[order] = upd_sorted
                    upd *= self.learning_rate
                else:
                    upd = self.learning_rate * _fit_bin_tree(g, h, self.max_leaves, self.min_samples_leaf)
                terms[j] += upd
                F += upd[B[:, j]]
            loss = np.mean((y[va] - F[va])**2)
            if loss < best_loss - 1e-12:
                best_loss, best_terms, best_F, stall = loss, [t.copy() for t in terms], F.copy(), 0
            else:
                stall += 1
                if stall >= self.early_stopping_rounds:
                    break
        return best_terms, best_F, y[tr].mean()

    def _boost_pairs(self, P, y, F0, tr, va):
        tables = {p: np.zeros((self.pair_binners_[p[0]].n_bins, self.pair_binners_[p[1]].n_bins))
                  for p in self.pairs_}
        F = F0.copy()
        best_loss, best_tables, stall = np.mean((y[va] - F[va])**2), {p: t.copy() for p, t in tables.items()}, 0
        for _ in range(self.max_rounds):
            for (i, j) in self.pairs_:
                r = y - F
                ni, nj = tables[(i, j)].shape
                _, upd = _fit_pair_tree(r[tr], P[tr, i], P[tr, j], ni, nj, self.min_samples_leaf)
                upd *= self.learning_rate
                tables[(i, j)] += upd
                F += upd[P[:, i], P[:, j]]
            loss = np.mean((y[va] - F[va])**2)
            if loss < best_loss - 1e-12:
                best_loss, best_tables, stall = loss, {p: t.copy() for p, t in tables.items()}, 0
            else:
                stall += 1
                if stall >= self.early_stopping_rounds:
                    break
        return best_tables

    def _detect_pairs(self, P, r):
        """FAST: rank every pair by the gain of its best 2x2 split on the residuals."""
        nf = P.shape[1]
        scores = []
        for i, j in combinations(range(nf), 2):
            gain, _ = _fit_pair_tree(r, P[:, i], P[:, j], self.pair_binners_[i].n_bins,
                                     self.pair_binners_[j].n_bins, self.min_samples_leaf)
            scores.append(((i, j), gain))
        scores.sort(key=lambda s: -s[1])
        self.pair_scores_ = [((self.feature_names_[i], self.feature_names_[j]), g) for (i, j), g in scores]
        forced = [tuple(sorted((self.feature_names_.index(a), self.feature_names_.index(b))))
                  for a, b in self.forced_interactions]
        chosen = list(dict.fromkeys(forced))
        for p, _ in scores:
            if len(chosen) >= self.interactions + len(forced):
                break
            if p not in chosen:
                chosen.append(p)
        return chosen

    # ---- public API
    def fit(self, X, y):
        names = self._names(X)
        self.feature_names_ = names
        X = np.asarray(X, float); y = np.asarray(y, float)
        cat = set(self.categorical)
        self.binners_ = [_Binner(self.max_bins, n in cat).fit(X[:, j]) for j, n in enumerate(names)]
        self.pair_binners_ = [_Binner(self.max_pair_bins, n in cat).fit(X[:, j]) for j, n in enumerate(names)]
        B, P = self._bin_all(X), self._bin_pairs(X)
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        bag_terms, bag_icpt, bag_splits = [], [], []
        for _ in range(self.outer_bags):
            perm = rng.permutation(n)
            nva = max(1, int(round(self.validation_size * n)))
            va, tr = perm[:nva], perm[nva:]
            terms, F, icpt = self._boost_mains(B, y, tr, va, rng)
            bag_terms.append(terms); bag_icpt.append(icpt); bag_splits.append((tr, va, F))
        # average main effects across bags
        self.term_scores_ = [np.mean([bt[j] for bt in bag_terms], axis=0) for j in range(len(names))]
        self.term_std_ = [np.std([bt[j] for bt in bag_terms], axis=0) for j in range(len(names))]
        self.intercept_ = float(np.mean(bag_icpt))
        # pairs
        self.pairs_, self.pair_scores_ = [], []
        self.pair_tables_ = {}
        if self.interactions > 0 or self.forced_interactions:
            r = y - self._predict_mains(B)
            self.pairs_ = self._detect_pairs(P, r)
            bag_tables = []
            for tr, va, _ in bag_splits:
                F0 = self._predict_mains(B)
                bag_tables.append(self._boost_pairs(P, y, F0, tr, va))
            self.pair_tables_ = {p: np.mean([bt[p] for bt in bag_tables], axis=0) for p in self.pairs_}
        self._center(B, P)
        self.X_train_ = X
        return self

    def _center(self, B, P):
        """Shift each term to have mean 0 over the training data (intercept absorbs it)."""
        for j, t in enumerate(self.term_scores_):
            m = t[B[:, j]].mean(); t -= m; self.intercept_ += m
        for p, tab in self.pair_tables_.items():
            m = tab[P[:, p[0]], P[:, p[1]]].mean(); tab -= m; self.intercept_ += m

    def _predict_mains(self, B):
        return self.intercept_ + sum(t[B[:, j]] for j, t in enumerate(self.term_scores_))

    def term_contributions(self, X):
        """Matrix of per-term contributions (n_samples x n_terms) and term names."""
        B, P = self._bin_all(X), self._bin_pairs(X)
        cols = [t[B[:, j]] for j, t in enumerate(self.term_scores_)]
        names = list(self.feature_names_)
        for (i, j), tab in self.pair_tables_.items():
            cols.append(tab[P[:, i], P[:, j]])
            names.append(f"{self.feature_names_[i]} x {self.feature_names_[j]}")
        return np.column_stack(cols), names

    def predict(self, X):
        C, _ = self.term_contributions(X)
        return self.intercept_ + C.sum(1)

    def term_importances(self, X=None):
        """InterpretML-style importance: mean |contribution| of each term."""
        C, names = self.term_contributions(self.X_train_ if X is None else X)
        return dict(zip(names, np.abs(C).mean(0)))

    def shape_function(self, feature):
        j = self.feature_names_.index(feature)
        b = self.binners_[j]
        return {"x": b.bin_centers(self.X_train_[:, j]), "labels": b.bin_labels(),
                "score": self.term_scores_[j].copy(), "std": self.term_std_[j].copy(),
                "categorical": b.categorical, "edges": None if b.categorical else b.edges_.copy()}

    def shap_values(self, X, background=None):
        """Exact interventional SHAP values for a GA2M.
        Main effect f_j: phi_j = f_j(x_j) - E_bg[f_j].
        Pair f_ij      : phi_i += 1/2 [f_ij(xi,xj) - E f_ij(Xi,xj)] + 1/2 [E f_ij(xi,Xj) - E f_ij(Xi,Xj)]
                         (and symmetrically for j) - the closed form of Shapley values for a
                         function of two variables under the interventional value function."""
        X = np.asarray(X, float)
        bg = self.X_train_ if background is None else np.asarray(background, float)
        B, P = self._bin_all(X), self._bin_pairs(X)
        Bb, Pb = self._bin_all(bg), self._bin_pairs(bg)
        phi = np.zeros(X.shape)
        base = self.intercept_
        for j, t in enumerate(self.term_scores_):
            mj = t[Bb[:, j]].mean()
            phi[:, j] = t[B[:, j]] - mj
            base += mj
        for (i, j), tab in self.pair_tables_.items():
            full = tab[P[:, i], P[:, j]]
            Ei_xj = np.array([tab[Pb[:, i], P[k, j]].mean() for k in range(len(X))])  # vary i
            Ej_xi = np.array([tab[P[k, i], Pb[:, j]].mean() for k in range(len(X))])  # vary j
            E = tab[Pb[:, i][:, None], Pb[:, j][None, :]].mean()
            phi[:, i] += 0.5 * (full - Ei_xj) + 0.5 * (Ej_xi - E)
            phi[:, j] += 0.5 * (full - Ej_xi) + 0.5 * (Ei_xj - E)
            base += E   # interventional baseline: features drawn independently from background
        return phi, base

    def to_json(self):
        """Portable export (used by the web front-end)."""
        out = {"intercept": self.intercept_, "features": [], "pairs": []}
        for j, n in enumerate(self.feature_names_):
            b = self.binners_[j]
            out["features"].append({"name": n, "categorical": b.categorical,
                                    "edges": None if b.categorical else b.edges_.tolist(),
                                    "levels": b.levels_.tolist() if b.categorical else None,
                                    "scores": self.term_scores_[j].tolist(),
                                    "std": self.term_std_[j].tolist()})
        for (i, j), tab in self.pair_tables_.items():
            bi, bj = self.pair_binners_[i], self.pair_binners_[j]
            out["pairs"].append({"features": [self.feature_names_[i], self.feature_names_[j]],
                                 "edges": [None if bi.categorical else bi.edges_.tolist(),
                                           None if bj.categorical else bj.edges_.tolist()],
                                 "levels": [bi.levels_.tolist() if bi.categorical else None,
                                            bj.levels_.tolist() if bj.categorical else None],
                                 "table": tab.tolist()})
        return out
