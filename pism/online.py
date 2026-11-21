"""Online recalibration of eta (Method B, Algorithm 2)."""

from collections import deque

import numpy as np


class OnlineRecalibrator:
    """Quantile tracker of Eq. (15) with the drift fallback of Algorithm 2.

    Args:
        eta0: starting value, normally the offline eta_95.
        eta_det: deterministic calibration max_m alpha_m, deployed during fallback.
        alpha: target miscoverage.
        gamma: step size; defaults to 0.05 * eta0.
        eta_min, eta_max: clip interval; eta_max defaults to 3 * eta_det.
        window: length W of the rolling coverage window.
        c_enter, c_exit: windowed coverage that enters / leaves fallback.
        metered: boolean mask of metered buses (None = all buses).
        beta: offset for unmetered buses.
        omega: delay compensation omega_k (0 without delay).
    """

    def __init__(self, eta0: float, eta_det: float, alpha: float = 0.05,
                 gamma: float | None = None, eta_min: float = 0.0,
                 eta_max: float | None = None, window: int = 16,
                 c_enter: float = 0.80, c_exit: float = 0.90,
                 metered: np.ndarray | None = None, beta: float = 0.0,
                 omega: float = 0.0):
        self.eta = float(eta0)
        self.eta_det = float(eta_det)
        self.alpha = alpha
        self.gamma = 0.05 * eta0 if gamma is None else gamma
        self.eta_min = eta_min
        self.eta_max = 3.0 * eta_det if eta_max is None else eta_max
        self.c_enter, self.c_exit = c_enter, c_exit
        self.metered = None if metered is None else np.asarray(metered, dtype=bool)
        self.beta, self.omega = beta, omega
        self.fallback = False
        self._covered = deque(maxlen=window)
        self._covered_cf = deque(maxlen=window)

    def margin(self, S: np.ndarray) -> np.ndarray:
        """Per-bus margin to deploy: (eta_bar + beta * 1[j not metered]) * S_j."""
        eta_bar = self.eta_det if self.fallback else self.eta + self.omega
        coef = np.full(S.shape, eta_bar)
        if self.metered is not None:
            coef[~self.metered] += self.beta
        return coef * S

    def update(self, delta: np.ndarray, S: np.ndarray) -> float:
        """Observe the realized errors of one step and advance eta; returns the score s_t."""
        ok = np.isfinite(S) & (S > 1e-10)
        if self.metered is not None:
            ok &= self.metered
        score = float(np.max(delta[ok] / S[ok])) if ok.any() else 0.0

        exceed = score > self.eta
        self.eta = float(np.clip(self.eta + self.gamma * (exceed - self.alpha),
                                 self.eta_min, self.eta_max))
        self._covered.append(not exceed)
        self._covered_cf.append(score <= self.eta)

        if len(self._covered) == self._covered.maxlen:
            if not self.fallback and np.mean(self._covered) < self.c_enter:
                self.fallback = True
            elif self.fallback and np.mean(self._covered_cf) >= self.c_exit:
                self.fallback = False
        return score
