"""Offline calibration of eta (Method A, Algorithm 1) and the impedance-error repair."""

import numpy as np


def _valid(S):
    return np.isfinite(S) & (S > 1e-10)


def worst_bus_scores(delta: np.ndarray, S: np.ndarray, signed: bool = False) -> np.ndarray:
    """alpha_m = max_j |delta_j| / S_j for every sample (rows of delta, S).

    With signed=True the score is max_j delta_j / S_j: a margin on the lower
    voltage bound is only at risk where the LDF over-estimates the voltage.
    """
    ok = _valid(S)
    d = np.nan_to_num(delta)
    ratio = np.where(ok, (d if signed else np.abs(d)) / np.where(ok, S, 1.0), -np.inf)
    return ratio.max(axis=1)


def calibrate_eta(delta: np.ndarray, S: np.ndarray, coverage: float = 0.95,
                  signed: bool = False) -> float:
    """eta at the `coverage` quantile of the worst-bus scores (Algorithm 1)."""
    return float(np.quantile(worst_bus_scores(delta, S, signed), coverage))


def deterministic_eta(delta: np.ndarray, S: np.ndarray) -> float:
    """eta_det = max_m alpha_m, the sample-maximum calibration."""
    return float(worst_bus_scores(delta, S).max())


def joint_coverage(delta: np.ndarray, S: np.ndarray, eta: float, signed: bool = False) -> float:
    """Fraction of samples with eta * S_j >= |delta_j| (or delta_j if signed) at every bus."""
    return float((worst_bus_scores(delta, S, signed) <= eta).mean())


def budget_at_equal_violation(delta: np.ndarray, S: np.ndarray, quantile: float) -> dict:
    """Average PI-SM margin against a static margin with the same violation rate (Table III)."""
    ok = _valid(S)
    eta = calibrate_eta(delta, S, quantile)
    violated = ((eta * np.where(ok, S, 0.0) < np.abs(np.nan_to_num(delta))) & ok).any(axis=1)
    viol_rate = float(violated.mean())
    worst_abs = np.where(ok, np.abs(np.nan_to_num(delta)), -np.inf).max(axis=1)
    eps_static = float(np.quantile(worst_abs, 1 - viol_rate)) if viol_rate > 0 else float(worst_abs.max())
    budget_pi = float(np.mean(eta * S[ok]))
    return {"eta": eta, "violation_rate": viol_rate, "budget_pi": budget_pi,
            "budget_static": eps_static, "saving_pct": (1 - budget_pi / eps_static) * 100}


def impedance_repair_eta(eta: float, delta_tol: float) -> float:
    """Inflated scalar of Eq. (19): eta (1 + Delta) + Delta covers any Delta-bounded impedance error."""
    return eta * (1 + delta_tol) + delta_tol
