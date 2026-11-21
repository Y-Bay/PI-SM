"""Monte Carlo operating points and LDF errors against AC power flow."""

import copy
import importlib.util

import numpy as np
import pandapower as pp
from pandapower.powerflow import LoadflowNotConverged

from .feeders import get_base_mva
from .ldf import ldf_voltage, sensitivity_scale

_NUMBA = importlib.util.find_spec("numba") is not None


def _pv_ev_buses(net):
    load_buses = sorted(set(net.load["bus"].values.astype(int)))
    return load_buses[::3], load_buses[1::3]


def sample_operating_points(net: pp.pandapowerNet, n_samples: int = 1000,
                            load_scale_range: tuple = (0.3, 2.0), perturb_std: float = 0.1,
                            pv_penetration: float = 0.3, ev_penetration: float = 0.3,
                            seed: int = 42) -> list[dict]:
    """Operating points of Algorithm 1.

    Loads are scaled by a common factor lambda ~ U[load_scale_range] with
    lognormal per-load noise. PV (one third of the load buses) and EV charging
    (another third) are proportional to the local base load.
    """
    rng = np.random.default_rng(seed)
    base_p = net.load["p_mw"].values.copy()
    base_q = net.load["q_mvar"].values.copy()
    load_buses = net.load["bus"].values.astype(int)
    pv_buses, ev_buses = _pv_ev_buses(net)
    n_bus = len(net.bus)

    samples = []
    for _ in range(n_samples):
        scale = rng.uniform(*load_scale_range)
        perturb = rng.lognormal(0, perturb_std, size=len(base_p))
        extra = np.zeros(n_bus)
        pv_factor = rng.uniform(0, pv_penetration)
        for bus in pv_buses:
            bus_load = base_p[load_buses == bus].sum()
            if bus_load > 1e-6:
                extra[bus] -= bus_load * pv_factor * rng.uniform(0.5, 1.5)
        ev_factor = rng.uniform(0, ev_penetration)
        for bus in ev_buses:
            bus_load = base_p[load_buses == bus].sum()
            if bus_load > 1e-6:
                extra[bus] += bus_load * ev_factor * rng.uniform(0.5, 1.5)
        samples.append({"p_mw": base_p * scale * perturb, "q_mvar": base_q * scale * perturb,
                        "p_bus_extra_mw": extra})
    return samples


def sample_non_proportional(net: pp.pandapowerNet, n_samples: int = 1000,
                            load_scale_range: tuple = (0.3, 2.0), pv_mw: float = 0.15,
                            ev_mw: float = 0.25, seed: int = 42) -> list[dict]:
    """As sample_operating_points, but PV and EV injections have fixed MW sizes."""
    rng = np.random.default_rng(seed)
    base_p = net.load["p_mw"].values.copy()
    base_q = net.load["q_mvar"].values.copy()
    pv_buses, ev_buses = _pv_ev_buses(net)
    n_bus = len(net.bus)

    samples = []
    for _ in range(n_samples):
        scale = rng.uniform(*load_scale_range)
        perturb = rng.lognormal(0, 0.1, size=len(base_p))
        extra = np.zeros(n_bus)
        for bus in pv_buses:
            extra[bus] -= pv_mw * rng.uniform(0, 1)
        for bus in ev_buses:
            extra[bus] += ev_mw * rng.uniform(0, 1)
        samples.append({"p_mw": base_p * scale * perturb, "q_mvar": base_q * scale * perturb,
                        "p_bus_extra_mw": extra})
    return samples


def bus_injections(net: pp.pandapowerNet, sample: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-bus withdrawals (p.u.) of one sample, PV/EV included."""
    n_bus = len(net.bus)
    base_mva = get_base_mva(net)
    buses = net.load["bus"].values.astype(int)
    p = np.bincount(buses, weights=sample["p_mw"], minlength=n_bus) / base_mva
    q = np.bincount(buses, weights=sample["q_mvar"], minlength=n_bus) / base_mva
    extra = sample.get("p_bus_extra_mw")
    if extra is not None:
        p = p + extra / base_mva
    return p, q


def ldf_errors(net: pp.pandapowerNet, R: np.ndarray, X: np.ndarray,
               samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """LDF error delta = V_LDF - V_AC and sensitivity scale S for every sample.

    `net` is the physical grid solved by AC power flow; R and X are the LDF
    matrices the operator uses, which may come from different (nominal)
    parameters. Returns two (n_samples, n_bus) arrays with NaN at the slack
    bus; samples whose AC power flow fails are dropped.
    """
    work = copy.deepcopy(net)
    buses = work.load["bus"].values.astype(int)
    first_load = {b: i for i, b in reversed(list(enumerate(buses)))}
    slack = int(work.ext_grid.bus.iloc[0])

    deltas, scales = [], []
    for sample in samples:
        p_mw = sample["p_mw"].copy()
        extra = sample.get("p_bus_extra_mw")
        if extra is not None:
            for bus in np.flatnonzero(np.abs(extra) >= 1e-9):
                if bus in first_load:
                    p_mw[first_load[bus]] += extra[bus]
        work.load["p_mw"] = p_mw
        work.load["q_mvar"] = sample["q_mvar"]
        try:
            pp.runpp(work, algorithm="nr", init="auto", max_iteration=100, numba=_NUMBA)
        except LoadflowNotConverged:
            continue
        p, q = bus_injections(net, sample)
        delta = ldf_voltage(R, X, p, q) - work.res_bus["vm_pu"].values
        S = sensitivity_scale(R, X, p, q)
        delta[slack] = np.nan
        S[slack] = np.nan
        deltas.append(delta)
        scales.append(S)
    return np.array(deltas), np.array(scales)
