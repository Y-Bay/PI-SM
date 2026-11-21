"""Impedance-parameter error (Section IV-H) and the repair of Eq. (19).

Every line's r and x are perturbed by independent factors in [1-Delta, 1+Delta].
consistent: eta calibrated and evaluated with the perturbed grid and its matrices;
mismatch:   nominal eta and matrices, while the physical grid is perturbed;
repair:     mismatch with eta inflated by Eq. (19);
bound:      max of |(R'-R)p + (X'-X)q| / (Delta S_j), at most 1 by construction.
Coverage is one-sided (delta_j <= eta S_j), as the margin protects the lower bound.
"""

import argparse

import numpy as np

from pism import (bus_injections, calibrate_eta, get_network, impedance_repair_eta,
                  joint_coverage, ldf_errors, ldf_matrices, perturb_line_impedances,
                  sample_operating_points, sensitivity_scale)

CLOSED_LOOP_RANGE = {"33": (0.3, 2.0), "69": (0.3, 2.0), "123": (0.21, 1.4)}


def bound_ratio(net, samples, R, X, R2, X2, delta_tol):
    worst = 0.0
    for s in samples:
        p, q = bus_injections(net, s)
        err = np.abs((R2 - R) @ p + (X2 - X) @ q)
        S = sensitivity_scale(R, X, p, q)
        ok = S > 1e-10
        worst = max(worst, float(np.max(err[ok] / (delta_tol * S[ok]))))
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="33")
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--draws", type=int, default=3)
    args = ap.parse_args()

    lr = CLOSED_LOOP_RANGE[args.network]
    net = get_network(args.network)
    R, X = ldf_matrices(net)
    cal = sample_operating_points(net, n_samples=500, seed=0, load_scale_range=lr)
    val = sample_operating_points(net, n_samples=500, seed=7, load_scale_range=lr)
    eta = calibrate_eta(*ldf_errors(net, R, X, cal), signed=True)
    print(f"{args.network}-bus, nominal eta_95 = {eta:.4f}")
    print(f"{'Delta':>6s} {'draw':>5s} {'consistent':>11s} {'mismatch':>9s} {'repair':>7s} {'bound':>6s}")
    for d in args.deltas:
        for k in range(args.draws):
            net2 = perturb_line_impedances(net, d, np.random.default_rng(1000 + k))
            R2, X2 = ldf_matrices(net2)
            eta2 = calibrate_eta(*ldf_errors(net2, R2, X2, cal), signed=True)
            consistent = joint_coverage(*ldf_errors(net2, R2, X2, val), eta2, signed=True)
            mism = ldf_errors(net2, R, X, val)
            print(f"{d:6.0%} {k:5d} {consistent:11.3f} {joint_coverage(*mism, eta, signed=True):9.3f} "
                  f"{joint_coverage(*mism, impedance_repair_eta(eta, d), signed=True):7.3f} "
                  f"{bound_ratio(net, val[:200], R, X, R2, X2, d):6.2f}")
