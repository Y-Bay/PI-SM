# PI-SM: Physics-Informed Safety Margins

Reference implementation for

> Y. Bai, J. Zhang, Y. Dai, T. Gao, P. Xu, and H. Jiang, "Sensitivity-Scale Factorization of LinDistFlow Errors for Adaptive Voltage Safety Margins in Distribution Networks," *IEEE Transactions on Smart Grid*, 2026.

The LinDistFlow (LDF) voltage error at bus *j* factorizes approximately as
δ<sub>j</sub> ≈ S<sub>j</sub> r<sub>j</sub>, where the sensitivity scale
S<sub>j</sub> = Σ<sub>k</sub> |R̃<sub>jk</sub>||p<sub>k</sub>| + Σ<sub>k</sub> |X̃<sub>jk</sub>||q<sub>k</sub>|
is computable from the current withdrawals and r<sub>j</sub> varies slowly.
PI-SM tightens the lower voltage bound of any LDF-based dispatch by the per-bus
margin ε<sub>j</sub> = η S<sub>j</sub>, with a single scalar η calibrated
offline (Method A) or tracked online (Method B).

This package contains the LDF model, the margin, both calibration methods and
the open-loop studies of the paper. The closed-loop MPC and RL experiments and
the three-phase study are not included.

## Installation

```bash
pip install .
```

Requires Python 3.10 or later, NumPy and pandapower. Installing `numba` speeds
up the AC power flows used for calibration.

## Usage

```python
from pism import (get_network, ldf_matrices, sample_operating_points, ldf_errors,
                  calibrate_eta, sensitivity_scale)

net = get_network("33")
R, X = ldf_matrices(net)
delta, S = ldf_errors(net, R, X, sample_operating_points(net, n_samples=500, seed=0))
eta = calibrate_eta(delta, S, coverage=0.95)          # Algorithm 1

# at dispatch time, with the bus withdrawals p, q in p.u.:
eps = eta * sensitivity_scale(R, X, p, q)             # V_min + eps_j <= V_LDF_j
```

For online recalibration, `OnlineRecalibrator` implements Algorithm 2:
`margin(S)` returns the per-bus margin to deploy and `update(delta, S)` feeds
the realized errors of the step. A metered-bus mask, the offset β for unmetered
buses and the delay compensation ω<sub>k</sub> are optional arguments.

| Module | Content | Paper |
|---|---|---|
| `pism/feeders.py` | IEEE 33/69/123-bus feeders, impedance perturbation | Table I, Section IV-H |
| `pism/ldf.py` | LDF matrices R̃, X̃, LDF voltage, sensitivity scale S<sub>j</sub> | Eqs. (2), (4) |
| `pism/sampling.py` | Monte Carlo operating points, LDF errors against AC power flow | Algorithm 1 |
| `pism/calibration.py` | Method A, deterministic η, impedance repair | Algorithm 1, Eq. (19), Table III |
| `pism/online.py` | Method B quantile tracker with fallback | Eq. (15), Algorithm 2 |
| `pism/verification.py` | Statistics of Proposition 1, Lemma 1, Corollaries 1-2, Remark 1 | Eq. (9), Table II |

## Examples

| Script | Reproduces | Runtime |
|---|---|---|
| `examples/01_factorization.py` | Table II, Eq. (9), η<sub>95</sub> of Section III-B | ~10 min |
| `examples/02_offline_calibration.py` | Table III, η values of the closed-loop experiments | ~2 min |
| `examples/03_impedance_error.py` | Impedance-parameter noise and repair, Section IV-H | ~6 min |
| `examples/04_online_recalibration.py` | Method B on an open-loop stream (illustration) | ~1 min |

Runtimes are for a laptop without `numba`. Expected output:

```
$ python examples/01_factorization.py
                                    33-bus    69-bus   123-bus
R2(delta, S_j) pooled                0.895     0.846     0.865
R2 per-bus min                       0.938     0.879     0.910
kappa_hat                            0.514     0.484     0.581
Corr(S_j, r_j)                       0.853     0.804     0.783
Cor. 1: Corr(eps_PI, |delta|)        0.946     0.920     0.930
Cor. 2: CV reduction (%)                66        46        63
Cor. 2: rho / eta                     1.80      1.42      1.91
Rem. 1: worst-case impr. (%)            80        63        77
R2, non-proportional PV/EV            0.84      0.75      0.71
eta_95, seed 42                       0.18      0.25      0.38

Lemma 1, Eq. (9): kappa = 0.581 for U[0.3, 2.0], R2 bound 1 - kappa^2 = 0.66
```

```
$ python examples/02_offline_calibration.py
Table III: budget saving at equal violation rate
 quantile    33-bus    69-bus   123-bus
      90%     61.5%     81.5%     53.8%
      95%     62.6%     82.4%     54.9%
      99%     64.7%     83.8%     57.0%

Closed-loop calibration range, held-out joint coverage
 network        range   eta_95  eta_det  coverage
  33-bus   (0.3, 2.0)   0.1833   0.2096    94.0%
  69-bus   (0.3, 2.0)   0.2379   0.3254    94.2%
 123-bus  (0.21, 1.4)   0.2050   0.2287    93.4%
```

```
$ python examples/03_impedance_error.py
33-bus, nominal eta_95 = 0.1833
 Delta  draw  consistent  mismatch  repair  bound
    5%     0       0.926     0.906   1.000   0.24
    5%     1       0.926     0.996   1.000   0.46
    5%     2       0.926     0.978   1.000   0.33
   10%     0       0.926     0.864   1.000   0.24
   10%     1       0.926     1.000   1.000   0.46
   10%     2       0.926     0.980   1.000   0.33
   20%     0       0.928     0.748   1.000   0.24
   20%     1       0.926     1.000   1.000   0.46
   20%     2       0.926     0.886   1.000   0.33
```

Example 04 runs the tracker on an open-loop stream with a ±20% impedance shift
after day 2. Its numbers illustrate Algorithm 2 and are not reported in the
paper, whose Method B results come from the closed loop.

```
$ python examples/04_online_recalibration.py
offline eta_95 = 0.1833, eta_det = 0.2096
                   arm  cov before  cov after  mean eta after  fallback steps
          fixed eta_95      100.0%     100.0%          0.1833               0
  Method B from eta_95      100.0%      94.4%          0.1262              47
        fixed eta_95/4       27.1%       0.0%          0.0458               0
Method B from eta_95/4       92.2%      94.4%          0.1262              65
```

## Conventions

- Buses are 0-indexed; the substation is bus 0 on the 33- and 69-bus feeders and bus 113 on the 123-bus feeder.
- p and q are bus withdrawals in p.u. on the system base (loads positive, generation negative).
- δ<sub>j</sub> = V<sub>j</sub><sup>LDF</sup> − V<sub>j</sub><sup>AC</sup>; it is positive on the three feeders, so only the lower voltage bound needs a margin.
- The 33-bus feeder is `pandapower.networks.case33bw`. The 69-bus data follow Baran and Wu; the 123-bus single-phase equivalent follows Bobo, Venzke and Chatzivasileiadis (2020).

## Citation

```bibtex
@article{bai2026pism,
  author  = {Bai, Yuyang and Zhang, Jun and Dai, Yuxin and Gao, Tianlu and Xu, Peidong and Jiang, Huaiguang},
  title   = {Sensitivity-Scale Factorization of {LinDistFlow} Errors for Adaptive Voltage Safety Margins in Distribution Networks},
  journal = {IEEE Transactions on Smart Grid},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`.
