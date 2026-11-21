"""Statistics behind Proposition 1, Lemma 1, Corollaries 1-2 and Remark 1 (Table II)."""

import numpy as np


def closed_form_kappa(a: float, b: float) -> float:
    """kappa = CV[lambda] / CV[lambda^2] for lambda ~ U[a, b], Eq. (9)."""
    m1 = (a + b) / 2
    m2 = (a * a + a * b + b * b) / 3
    m4 = (b ** 5 - a ** 5) / (5 * (b - a))
    cv1 = np.sqrt(m2 - m1 ** 2) / m1
    cv2 = np.sqrt(m4 - m2 ** 2) / m2
    return float(cv1 / cv2)


def _cv(x):
    return np.std(x) / np.mean(x)


def factorization_stats(delta: np.ndarray, S: np.ndarray) -> dict:
    """Fit quality of delta_j ~ S_j and stability of the residual r_j = delta_j / S_j.

    r2_pooled: R^2 of the linear fit over all (sample, bus) pairs;
    r2_per_bus_min: smallest R^2 among the per-bus fits;
    kappa_hat: CV[r] / CV[|delta|];
    corr_S_abs_delta: Corr(eta S_j, |delta_j|), Corollary 1.
    """
    ok = np.isfinite(S) & (S > 1e-10)
    d, s = delta[ok], S[ok]
    r = d / s
    per_bus = []
    for j in range(S.shape[1]):
        m = ok[:, j]
        if m.sum() > 2:
            per_bus.append(np.corrcoef(S[m, j], delta[m, j])[0, 1] ** 2)
    return {
        "r2_pooled": float(np.corrcoef(s, d)[0, 1] ** 2),
        "r2_per_bus_mean": float(np.mean(per_bus)),
        "r2_per_bus_min": float(np.min(per_bus)),
        "kappa_hat": float(_cv(r) / _cv(np.abs(d))),
        "corr_S_r": float(np.corrcoef(s, r)[0, 1]),
        "corr_S_abs_delta": float(np.corrcoef(s, np.abs(d))[0, 1]),
    }


def equal_budget_comparison(delta: np.ndarray, S: np.ndarray, eta: float) -> dict:
    """Residual safety m = eps - |delta| of PI-SM and of a static margin with the same mean.

    cv_reduction_pct: 1 - CV[m_PI] / CV[m_static], Corollary 2;
    worst_case_impr_pct: gain in min m relative to the static margin, Remark 1;
    rho_over_eta: rho / eta with rho = 2 Cov(S, S r) / Var(S), the condition of Corollary 2.
    """
    ok = np.isfinite(S) & (S > 1e-10)
    abs_d, s = np.abs(delta[ok]), S[ok]
    eps_pi = eta * s
    m_pi = eps_pi - abs_d
    m_st = eps_pi.mean() - abs_d
    rho = 2 * np.cov(s, abs_d, bias=True)[0, 1] / np.var(s)
    return {
        "cv_reduction_pct": float((1 - _cv(m_pi) / _cv(m_st)) * 100),
        "worst_case_impr_pct": float((m_pi.min() - m_st.min()) / abs(m_st.min()) * 100),
        "rho_over_eta": float(rho / eta),
    }
