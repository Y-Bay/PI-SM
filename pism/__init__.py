"""Physics-Informed Safety Margins (PI-SM) for LinDistFlow-based voltage constraints."""

from .calibration import (budget_at_equal_violation, calibrate_eta, deterministic_eta,
                          impedance_repair_eta, joint_coverage, worst_bus_scores)
from .feeders import get_network, perturb_line_impedances
from .ldf import build_tree, ldf_matrices, ldf_voltage, sensitivity_scale
from .online import OnlineRecalibrator
from .sampling import (bus_injections, ldf_errors, sample_non_proportional,
                       sample_operating_points)
from .verification import closed_form_kappa, equal_budget_comparison, factorization_stats

__version__ = "1.0.0"
