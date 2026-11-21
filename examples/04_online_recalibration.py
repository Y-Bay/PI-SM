"""Method B (Algorithm 2) on a stream of operating points with a grid-parameter shift.

Eight days at 15-min resolution on the 33-bus feeder. After day 2 every line
impedance changes by an independent factor within +/-20%, while the operator
keeps the nominal LDF matrices and the offline eta_95. The stream is open loop:
all arms see the same operating points, so the comparison isolates the margin
rule. Coverage is the fraction of steps on which the deployed margin covers
delta_j at every bus.
"""

import argparse

import numpy as np

from pism import (OnlineRecalibrator, calibrate_eta, deterministic_eta, get_network,
                  ldf_errors, ldf_matrices, perturb_line_impedances, sample_operating_points,
                  worst_bus_scores)

STEPS_PER_DAY = 96
HOURLY_LOAD = [0.55, 0.50, 0.48, 0.47, 0.48, 0.55, 0.70, 0.85, 0.90, 0.88, 0.85, 0.85,
               0.83, 0.80, 0.80, 0.85, 0.95, 1.10, 1.20, 1.15, 1.05, 0.90, 0.75, 0.62]


def operating_points(net, n_days, seed):
    rng = np.random.default_rng(seed)
    pts = []
    for day in range(n_days):
        level = rng.uniform(0.9, 1.1)
        for t in range(STEPS_PER_DAY):
            lam = level * HOURLY_LOAD[t // 4]
            pts += sample_operating_points(net, n_samples=1, load_scale_range=(lam, lam),
                                           seed=seed * 10000 + day * STEPS_PER_DAY + t)
    return pts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--shift-day", type=int, default=2)
    ap.add_argument("--shift-delta", type=float, default=0.20)
    ap.add_argument("--shift-seed", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    net = get_network("33")
    R, X = ldf_matrices(net)
    cal = ldf_errors(net, R, X, sample_operating_points(net, n_samples=500, seed=0))
    eta95, eta_det = calibrate_eta(*cal), deterministic_eta(*cal)

    pts = operating_points(net, args.days, args.seed)
    k = args.shift_day * STEPS_PER_DAY
    shifted = perturb_line_impedances(net, args.shift_delta, np.random.default_rng(args.shift_seed))
    d0, S0 = ldf_errors(net, R, X, pts[:k])
    d1, S1 = ldf_errors(shifted, R, X, pts[k:])
    delta, S = np.vstack([d0, d1]), np.vstack([S0, S1])
    s = worst_bus_scores(delta, S, signed=True)
    ok = np.isfinite(S) & (S > 1e-10)

    print(f"offline eta_95 = {eta95:.4f}, eta_det = {eta_det:.4f}")
    print(f"{'arm':>22s} {'cov before':>11s} {'cov after':>10s} {'mean eta after':>15s} {'fallback steps':>15s}")
    for eta0 in (eta95, eta95 / 4):
        tag = "eta_95" if eta0 == eta95 else "eta_95/4"
        fixed = s <= eta0
        print(f"{'fixed ' + tag:>22s} {fixed[:k].mean():11.1%} {fixed[k:].mean():10.1%} "
              f"{eta0:15.4f} {0:15d}")
        rc = OnlineRecalibrator(eta0, eta_det, gamma=0.05 * eta95)
        eta_t, covered, fb = [], [], 0
        for t in range(len(s)):
            eta_t.append(rc.eta)
            fb += rc.fallback
            covered.append(np.all(delta[t, ok[t]] <= rc.margin(S[t])[ok[t]]))
            rc.update(delta[t], S[t])
        covered = np.array(covered)
        print(f"{'Method B from ' + tag:>22s} {covered[:k].mean():11.1%} {covered[k:].mean():10.1%} "
              f"{np.mean(eta_t[k:]):15.4f} {fb:15d}")
