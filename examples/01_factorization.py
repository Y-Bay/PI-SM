"""Table II: error factorization on the IEEE 33/69/123-bus feeders.

Five seeds x 1000 operating points per feeder (load scaling U[0.3, 2.0],
10% per-load noise). Corollary 2 and Remark 1 use the pooled eta_95; the last
row is eta_95 from the first seed alone (Section III-B).
"""

import argparse

import numpy as np

from pism import (calibrate_eta, closed_form_kappa, equal_budget_comparison,
                  factorization_stats, get_network, ldf_errors, ldf_matrices,
                  sample_non_proportional, sample_operating_points)

SEEDS = [42, 43, 44, 45, 46]


def pooled_errors(net, R, X, sampler, seeds, n_samples):
    runs = [ldf_errors(net, R, X, sampler(net, n_samples=n_samples, seed=s)) for s in seeds]
    return np.vstack([d for d, _ in runs]), np.vstack([s for _, s in runs]), runs[0]


def run(name, n_samples, n_seeds):
    net = get_network(name)
    R, X = ldf_matrices(net)
    seeds = SEEDS[:n_seeds]
    delta, S, first = pooled_errors(net, R, X, sample_operating_points, seeds, n_samples)
    out = factorization_stats(delta, S)
    out.update(equal_budget_comparison(delta, S, calibrate_eta(delta, S, 0.95)))
    delta_np, S_np, _ = pooled_errors(net, R, X, sample_non_proportional, seeds[:3], n_samples // 2)
    out["r2_non_proportional"] = factorization_stats(delta_np, S_np)["r2_pooled"]
    out["eta95_first_seed"] = calibrate_eta(*first, 0.95)
    return out


ROWS = [
    ("R2(delta, S_j) pooled", "r2_pooled", "{:.3f}"),
    ("R2 per-bus min", "r2_per_bus_min", "{:.3f}"),
    ("kappa_hat", "kappa_hat", "{:.3f}"),
    ("Corr(S_j, r_j)", "corr_S_r", "{:.3f}"),
    ("Cor. 1: Corr(eps_PI, |delta|)", "corr_S_abs_delta", "{:.3f}"),
    ("Cor. 2: CV reduction (%)", "cv_reduction_pct", "{:.0f}"),
    ("Cor. 2: rho / eta", "rho_over_eta", "{:.2f}"),
    ("Rem. 1: worst-case impr. (%)", "worst_case_impr_pct", "{:.0f}"),
    ("R2, non-proportional PV/EV", "r2_non_proportional", "{:.2f}"),
    ("eta_95, seed 42", "eta95_first_seed", "{:.2f}"),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=["33", "69", "123"])
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    res = {n: run(n, args.samples, args.seeds) for n in args.networks}
    print(f"{'':32s}" + "".join(f"{n + '-bus':>10s}" for n in args.networks))
    for label, key, fmt in ROWS:
        print(f"{label:32s}" + "".join(f"{fmt.format(res[n][key]):>10s}" for n in args.networks))
    print(f"\nLemma 1, Eq. (9): kappa = {closed_form_kappa(0.3, 2.0):.3f} for U[0.3, 2.0], "
          f"R2 bound 1 - kappa^2 = {1 - closed_form_kappa(0.3, 2.0) ** 2:.2f}")
