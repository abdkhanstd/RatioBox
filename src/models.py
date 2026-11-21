"""Breach forecasters.

All models answer the same question: for a firm that passes a screen with thresholds tau at quarter t,
what is the probability that it fails the screen at quarter t+h?

  clim        base rate of the training window
  margin      logistic regression on the signed margins tau_k - r_k (threshold-aware heuristic)
  proximity   logistic regression on boundary proximity exp(-5 d), rulebook disagreement and size
              (monitoring logit of Qadi et al., adapted to out-of-sample forecasting)
  logit       L2 logistic regression on all features plus margins
  gbm         gradient-boosted trees (LightGBM) trained on one rulebook's labels
  surface     baseline: one monotone boosted classifier of the breach surface P(breach | x, tau),
              trained on randomly drawn threshold vectors (threshold-augmented)
  qcop        RatioBox (the method of the paper): quantile boosting of each future ratio change and a
              Gaussian copula; the breach probability is the forecast mass outside the rulebook's box
  qcop-point  ablation of RatioBox with a mean forecast and pooled residuals instead of quantiles
"""
import numpy as np, pandas as pd, lightgbm as lgb, time
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from screens import RATIOS, OFF, RULEBOOKS, tau_vec

BASE_FEATS = (RATIOS + [f'{c}_l{L}' for c in RATIOS for L in [1, 2, 3, 4, 8]] +
              [f'{c}_{d}' for c in RATIOS + ['log_ta'] for d in ['d1', 'd4', 'd8']] +
              [f'{c}_max4' for c in RATIOS] + [f'{c}_sd4' for c in RATIOS] +
              ['log_ta', 'r_liab', 'r_curliab', 'r_curassets', 'r_equity', 'r_retained', 'r_ppe', 'r_goodwill',
               'r_intang', 'r_inventory', 'r_oplease', 'r_ltd_current', 'cash_missing', 'ar_missing',
               'debt_tagged', 'hist', 'fq', 'sic2'])
CURRENT_ONLY = RATIOS + ['log_ta', 'sic2', 'fq']   # ablation: no history and no balance-sheet structure

GBM_PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  num_threads=32)


def margin_feats(R, tau):
    tau = np.broadcast_to(tau, R.shape)
    m = tau - R
    active = tau < OFF - 1e-9
    m_act = np.where(active, m, np.nan)
    out = {f'm_{k}': np.where(active[:, i], m[:, i], 5.0) for i, k in enumerate(RATIOS)}
    out['m_min'] = np.nanmin(np.where(active, m, np.inf), axis=1)
    out.update({f'tau_{k}': np.where(active[:, i], tau[:, i], 5.0) for i, k in enumerate(RATIOS)})
    out['n_active'] = active.sum(axis=1)
    return pd.DataFrame(out)


def label(Rt, Rf, tau):
    tau = np.broadcast_to(tau, Rt.shape)
    pass_t = np.all(Rt < tau, axis=1)
    fail_f = ~np.all(Rf < tau, axis=1)
    return pass_t, fail_f


# ----------------------------------------------------------------------------- simple baselines
class Clim:
    def fit(self, D, y, tau): self.p = y.mean(); return self
    def predict(self, D, tau): return np.full(len(D), self.p)


class LogitModel:
    def __init__(self, cols_fn, C=1.0): self.cols_fn, self.C = cols_fn, C
    def _X(self, D, tau):
        X = self.cols_fn(D, tau)
        return X
    def fit(self, D, y, tau):
        X = self._X(D, tau)
        self.cols = X.columns
        self.med = X.median().fillna(0.0)   # a column with no observed value in the window is set to 0
        Xf = X.fillna(self.med)
        self.miss = [c for c in X.columns if X[c].isna().mean() > 0.01]
        Xf = pd.concat([Xf, X[self.miss].isna().astype(float).add_suffix('_na')], axis=1)
        self.sc = StandardScaler().fit(Xf.values)
        self.m = LogisticRegression(C=self.C, max_iter=2000).fit(np.clip(self.sc.transform(Xf.values), -8, 8), y)
        return self
    def predict(self, D, tau):
        X = self._X(D, tau)[self.cols]
        Xf = pd.concat([X.fillna(self.med), X[self.miss].isna().astype(float).add_suffix('_na')], axis=1)
        return self.m.predict_proba(np.clip(self.sc.transform(Xf.values), -8, 8))[:, 1]


def cols_margin(D, tau):
    M = margin_feats(D[RATIOS].values, tau)
    return M[['m_r_debt', 'm_r_cash', 'm_r_rec', 'm_min']].clip(-1, 1)


def nearest_boundary(R):
    """distance to the nearest threshold of any published rulebook (Qadi et al. proximity)."""
    taus = np.array([tau_vec(rb) for rb in RULEBOOKS.values()])  # (B,3)
    d = np.abs(R[:, None, :] - taus[None, :, :])
    d = np.where(taus[None, :, :] < OFF - 1e-9, d, np.inf)
    return d.min(axis=(1, 2))


def cols_proximity(D, tau):
    R = D[RATIOS].values
    return pd.DataFrame({'prox': np.exp(-5 * nearest_boundary(R)), 'disagree': (D['n_fail_rb'].values > 0).astype(float),
                         'n_fail': D['n_fail_rb'].values, 'log_ta': D['log_ta'].values})


def cols_full(D, tau):
    X = D[[c for c in BASE_FEATS if c != 'sic2']].reset_index(drop=True)
    return pd.concat([X, margin_feats(D[RATIOS].values, tau).clip(-5, 5)], axis=1)


# ----------------------------------------------------------------------------- boosted direct classifier
def _split_inner(D, frac_q=8):
    """temporal inner split: last `frac_q` decision quarters of the training window for early stopping."""
    qmax = D.qi.max()
    va = (D.qi > qmax - frac_q).values
    return ~va, va


class GBMDirect:
    def __init__(self, feats=BASE_FEATS, seed=0, margins=True):
        self.feats, self.seed, self.margins = feats, seed, margins
    def _X(self, D, tau):
        X = D[self.feats].reset_index(drop=True)
        if self.margins:
            X = pd.concat([X, margin_feats(D[RATIOS].values, tau)], axis=1)
        return X
    def fit(self, D, y, tau, inner=None):
        X = self._X(D, tau)
        tr, va = _split_inner(D) if inner is None else inner
        p = dict(GBM_PARAMS, seed=self.seed, bagging_seed=self.seed, feature_fraction_seed=self.seed)
        cat = ['sic2'] if 'sic2' in X.columns else []
        dtr = lgb.Dataset(X[tr], y[tr], categorical_feature=cat, free_raw_data=False)
        dva = lgb.Dataset(X[va], y[va], categorical_feature=cat, reference=dtr)
        self.m = lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
        # refit on the whole window with the selected number of rounds
        self.m = lgb.train(p, lgb.Dataset(X, y, categorical_feature=cat), num_boost_round=max(50, self.m.best_iteration))
        return self
    def predict(self, D, tau):
        return self.m.predict(self._X(D, tau))


# ----------------------------------------------------------------------------- baseline: threshold-augmented breach surface
def draw_thresholds(n, rng, p_off=0.2, uniform_active=True):
    """random screens: thresholds uniform on a broad range; the number of active ratios is uniform on {1,2,3}
    (or, if uniform_active is False, each ratio is switched off independently with probability p_off)."""
    lo = np.array([0.10, 0.10, 0.15]); hi = np.array([0.60, 0.60, 0.80])
    T = lo + (hi - lo) * rng.random((n, 3))
    if uniform_active:
        k = rng.integers(1, 4, size=n)                      # number of active ratios
        order = np.argsort(rng.random((n, 3)), axis=1)      # random subset of that size
        rank = np.argsort(order, axis=1)
        off = rank >= k[:, None]
    else:
        off = rng.random((n, 3)) < p_off
        off[off.all(axis=1), rng.integers(0, 3, size=off.all(axis=1).sum())] = False
    return np.where(off, OFF, T)


class Surface:
    """One monotone boosted model of P(fail at t+h | features at t, thresholds tau, pass at t)."""
    def __init__(self, feats=BASE_FEATS, seed=0, draws=4, include_published=True, monotone=True, p_off=0.2, uniform_active=True):
        self.feats, self.seed, self.draws, self.incl, self.monotone, self.p_off = feats, seed, draws, include_published, monotone, p_off
        self.uniform_active = uniform_active
    def _X(self, D, T):
        X = D[self.feats].reset_index(drop=True)
        return pd.concat([X, margin_feats(D[RATIOS].values, T)], axis=1)
    def augment(self, D, Rf, rng):
        Rt = D[RATIOS].values
        blocks = []
        taus = [draw_thresholds(len(D), rng, self.p_off, self.uniform_active) for _ in range(self.draws)]
        if self.incl:
            taus += [np.tile(tau_vec(rb), (len(D), 1)) for rb in RULEBOOKS.values()]
        for T in taus:
            ok_t, fail = label(Rt, Rf, T)
            ok = ok_t & ~np.isnan(Rf).any(axis=1)
            blocks.append((np.where(ok)[0], T[ok], fail[ok].astype(int)))
        idx = np.concatenate([b[0] for b in blocks]); T = np.concatenate([b[1] for b in blocks]); y = np.concatenate([b[2] for b in blocks])
        return idx, T, y
    def fit(self, D, Rf):
        rng = np.random.default_rng(self.seed)
        D = D.reset_index(drop=True)
        idx, T, y = self.augment(D, Rf, rng)
        DD = D.iloc[idx].reset_index(drop=True)
        X = self._X(DD, T)
        mono = [0] * len(X.columns)
        if self.monotone:
            for i, c in enumerate(X.columns):
                if c.startswith('m_') or c.startswith('tau_'):
                    mono[i] = -1
        p = dict(GBM_PARAMS, seed=self.seed, bagging_seed=self.seed, feature_fraction_seed=self.seed,
                 monotone_constraints=mono, monotone_constraints_method='advanced')
        tr, va = _split_inner(DD)
        cat = ['sic2'] if 'sic2' in X.columns else []
        dtr = lgb.Dataset(X[tr], y[tr], categorical_feature=cat, free_raw_data=False)
        dva = lgb.Dataset(X[va], y[va], categorical_feature=cat, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        self.m = lgb.train(p, lgb.Dataset(X, y, categorical_feature=cat), num_boost_round=max(50, m.best_iteration))
        self.n_train = len(y)
        return self
    def predict(self, D, tau):
        T = np.tile(np.asarray(tau, float), (len(D), 1)) if np.ndim(tau) == 1 else tau
        return self.m.predict(self._X(D.reset_index(drop=True), T))


# ----------------------------------------------------------------------------- RatioBox: forecast the ratios, then apply the rulebook
QLEVELS = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99])


class QuantCopula:
    """Quantile boosting of each future ratio change, Gaussian copula across ratios (screen-agnostic)."""
    def __init__(self, feats=BASE_FEATS, seed=0, n_mc=4000, rounds=600, lr=0.05, leaves=63):
        self.feats, self.seed, self.n_mc, self.rounds, self.lr, self.leaves = feats, seed, n_mc, rounds, lr, leaves
    def fit(self, D, Rf):
        D = D.reset_index(drop=True)
        X = D[self.feats]
        ok = ~np.isnan(Rf).any(axis=1) & ~np.isnan(D[RATIOS].values).any(axis=1)
        X, Rt, Rf = X[ok].reset_index(drop=True), D[RATIOS].values[ok], Rf[ok]
        dY = Rf - Rt
        self.models = {}
        cat = ['sic2'] if 'sic2' in X.columns else []
        for j, k in enumerate(RATIOS):
            for a in QLEVELS:
                p = dict(GBM_PARAMS, objective='quantile', alpha=float(a), seed=self.seed, learning_rate=self.lr,
                         num_leaves=self.leaves, min_child_samples=200, max_bin=63)
                self.models[(k, a)] = lgb.train(p, lgb.Dataset(X, dY[:, j], categorical_feature=cat), num_boost_round=self.rounds)
        # PIT residuals -> copula correlation
        U = np.column_stack([self._cdf(X, j, dY[:, j]) for j in range(3)])
        Z = norm.ppf(np.clip(U, 1e-4, 1 - 1e-4))
        self.corr = np.corrcoef(Z.T)
        rng = np.random.default_rng(self.seed)
        self.Zmc = rng.multivariate_normal(np.zeros(3), self.corr, size=self.n_mc)
        return self
    def _quantiles(self, X, j):
        Q = np.column_stack([self.models[(RATIOS[j], a)].predict(X) for a in QLEVELS])
        return np.sort(Q, axis=1)
    def _cdf(self, X, j, v, tails=True):
        """conditional CDF of the change of ratio j at v: linear between the sorted quantiles, exponential tails
        fitted to the two outermost quantiles on each side (tails=False: flat mass alpha_1/2 and (1-alpha_L)/2)."""
        Q = self._quantiles(X, j)
        v = np.asarray(v, float)
        a = QLEVELS; L = len(a)
        idx = (Q < v[:, None]).sum(axis=1)
        out = np.where(idx == 0, a[0] / 2, 1 - (1 - a[-1]) / 2).astype(float)
        mid = (idx > 0) & (idx < L)
        r = np.where(mid)[0]; i1 = idx[mid]; i0 = i1 - 1
        q0, q1 = Q[r, i0], Q[r, i1]
        w = np.where(q1 > q0, (v[mid] - q0) / np.maximum(q1 - q0, 1e-12), 0.5)
        out[mid] = a[i0] + w * (a[i1] - a[i0])
        if tails:
            lo = idx == 0
            s = np.maximum(Q[lo, 1] - Q[lo, 0], 1e-4) / np.log(a[1] / a[0])
            out[lo] = a[0] * np.exp((v[lo] - Q[lo, 0]) / s)
            hi = idx == L
            s = np.maximum(Q[hi, -1] - Q[hi, -2], 1e-4) / np.log((1 - a[-2]) / (1 - a[-1]))
            out[hi] = 1 - (1 - a[-1]) * np.exp(-(v[hi] - Q[hi, -1]) / s)
        return out
    def marginal_F(self, D, tau, tails=True):
        """marginal CDF of each future ratio evaluated at its threshold (1 for switched-off ratios)."""
        D = D.reset_index(drop=True)
        X = D[self.feats]
        Rt = D[RATIOS].values
        tau = np.broadcast_to(np.asarray(tau, float), Rt.shape)
        F = np.ones_like(Rt)
        for j in range(3):
            act = tau[:, j] < OFF - 1e-9
            if act.any():
                F[act, j] = self._cdf(X[act], j, (tau[act, j] - Rt[act, j]) - 1e-12, tails=tails)
        return F

    @staticmethod
    def box(F, U):
        """1 - P(all future ratios below their thresholds) with copula draws U (common random numbers)."""
        out = np.empty(len(F))
        for s in range(0, len(F), 2000):
            Fi = F[s:s + 2000]
            out[s:s + 2000] = 1 - np.all(U[None, :, :] <= Fi[:, None, :], axis=2).mean(axis=1)
        return out

    def qmc_draws(self, m=14):
        """scrambled Sobol points (2^m) mapped to the Gaussian copula; the same points for every firm and screen."""
        if getattr(self, '_Uq', None) is None:
            from scipy.stats import qmc
            S = qmc.Sobol(d=3, scramble=True, seed=self.seed).random_base2(m)
            Z = norm.ppf(np.clip(S, 1e-12, 1 - 1e-12)) @ np.linalg.cholesky(self.corr).T
            self._Uq = norm.cdf(Z)
        return self._Uq

    def box_exact(self, F):
        """P(breach) = 1 - C(F1,F2,F3): closed form when one ratio is screened, quasi-Monte Carlo otherwise.
        The QMC value is clipped at the Frechet bound 1 - min_k F_k, which the exact value always satisfies,
        so probabilities stay ordered across nested boxes that screen different numbers of ratios."""
        n_act = (F < 1).sum(axis=1)
        out = 1 - F.prod(axis=1)                      # exact when at most one marginal is below 1
        multi = n_act >= 2
        if multi.any():
            Fm = F[multi]
            out[multi] = np.maximum(self.box(Fm, self.qmc_draws()), 1 - Fm.min(axis=1))
        return out

    def predict_variants(self, D, tau, legacy=False):
        F = self.marginal_F(D, tau)
        V = {'full': self.box_exact(F), 'indep': 1 - F.prod(axis=1)}
        if legacy:   # flat tails and 4,000 pseudo-random draws (used only to compare the two estimators)
            V['legacy'] = self.box(self.marginal_F(D, tau, tails=False), norm.cdf(self.Zmc))
        return V

    def predict(self, D, tau):
        return self.predict_variants(D, tau)['full']


class PointCopula(QuantCopula):
    """Ablation of RatioBox without conditional quantiles: a boosted mean forecast of each ratio change plus the
    pooled distribution of its cross-fitted residuals (5 folds grouped by firm). Tails, copula and box scoring
    are those of RatioBox, so only the conditional spread and shape are removed."""
    def fit(self, D, Rf, folds=5):
        D = D.reset_index(drop=True)
        X = D[self.feats]
        ok = ~np.isnan(Rf).any(axis=1) & ~np.isnan(D[RATIOS].values).any(axis=1)
        firm = D.cik.values[ok]
        X, Rt, Rf = X[ok].reset_index(drop=True), D[RATIOS].values[ok], Rf[ok]
        dY = Rf - Rt
        cat = ['sic2'] if 'sic2' in X.columns else []
        p = dict(GBM_PARAMS, objective='regression', seed=self.seed, learning_rate=self.lr, num_leaves=self.leaves,
                 min_child_samples=200, max_bin=63)
        fold = np.random.default_rng(self.seed).integers(0, folds, len(np.unique(firm)))[np.unique(firm, return_inverse=True)[1]]
        self.models, self.rq = {}, {}
        for j, k in enumerate(RATIOS):
            res = np.empty(len(X))
            for f in range(folds):
                tr = fold != f
                b = lgb.train(p, lgb.Dataset(X[tr], dY[tr, j], categorical_feature=cat), num_boost_round=self.rounds)
                res[~tr] = dY[~tr, j] - b.predict(X[~tr])
            self.models[k] = lgb.train(p, lgb.Dataset(X, dY[:, j], categorical_feature=cat), num_boost_round=self.rounds)
            self.rq[k] = np.quantile(res, QLEVELS)
        U = np.column_stack([self._cdf(X, j, dY[:, j]) for j in range(3)])
        Z = norm.ppf(np.clip(U, 1e-4, 1 - 1e-4))
        self.corr = np.corrcoef(Z.T)
        rng = np.random.default_rng(self.seed)
        self.Zmc = rng.multivariate_normal(np.zeros(3), self.corr, size=self.n_mc)
        return self
    def _quantiles(self, X, j):
        k = RATIOS[j]
        return self.models[k].predict(X)[:, None] + self.rq[k][None, :]
