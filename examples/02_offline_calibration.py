"""Method A (Algorithm 1): offline calibration of eta.

Part 1 reproduces Table III: average PI-SM margin against a static margin with
the same violation rate (500 samples, load scaling U[0.3, 2.0]).
Part 2 calibrates eta on the load range used in the closed-loop experiments
and checks joint coverage on an independent sample set.
"""

import argparse

from pism import (budget_at_equal_violation, calibrate_eta, deterministic_eta,
                  get_network, joint_coverage, ldf_errors, ldf_matrices,
                  sample_operating_points)

CLOSED_LOOP_RANGE = {"33": (0.3, 2.0), "69": (0.3, 2.0), "123": (0.21, 1.4)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=["33", "69", "123"])
    args = ap.parse_args()

    print("Table III: budget saving at equal violation rate")
    print(f"{'quantile':>9s}" + "".join(f"{n + '-bus':>10s}" for n in args.networks))
    errors, models = {}, {}
    for n in args.networks:
        net = get_network(n)
        models[n] = (net, *ldf_matrices(net))
        errors[n] = ldf_errors(*models[n], sample_operating_points(net, n_samples=500))
    for q in (0.90, 0.95, 0.99):
        row = [budget_at_equal_violation(*errors[n], q) for n in args.networks]
        print(f"{q:9.0%}" + "".join(f"{r['saving_pct']:9.1f}%" for r in row))

    print("\nClosed-loop calibration range, held-out joint coverage")
    print(f"{'network':>8s} {'range':>12s} {'eta_95':>8s} {'eta_det':>8s} {'coverage':>9s}")
    for n in args.networks:
        lr = CLOSED_LOOP_RANGE[n]
        net, R, X = models[n]
        cal = ldf_errors(net, R, X, sample_operating_points(net, n_samples=500, seed=0, load_scale_range=lr))
        val = ldf_errors(net, R, X, sample_operating_points(net, n_samples=500, seed=1, load_scale_range=lr))
        eta = calibrate_eta(*cal)
        print(f"{n + '-bus':>8s} {str(lr):>12s} {eta:8.4f} {deterministic_eta(*cal):8.4f} "
              f"{joint_coverage(*val, eta):8.1%}")
