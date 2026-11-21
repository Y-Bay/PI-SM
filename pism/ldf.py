"""LinDistFlow (LDF) sensitivity matrices, voltage prediction and sensitivity scale."""

from collections import deque

import numpy as np
import pandapower as pp

from .feeders import get_base_mva, get_slack_bus


def build_tree(net: pp.pandapowerNet) -> dict:
    """Radial tree rooted at the slack bus.

    Returns parent, children, depth and, for every non-slack bus, the p.u.
    resistance and reactance of the line that feeds it.
    """
    n_bus = len(net.bus)
    slack = get_slack_bus(net)
    base_mva = get_base_mva(net)

    adj = {i: [] for i in range(n_bus)}
    for _, row in net.line.iterrows():
        if not row.get("in_service", True):
            continue
        fb, tb = int(row["from_bus"]), int(row["to_bus"])
        z_base = net.bus.at[fb, "vn_kv"] ** 2 / base_mva
        r_pu = row["r_ohm_per_km"] * row["length_km"] / z_base
        x_pu = row["x_ohm_per_km"] * row["length_km"] / z_base
        adj[fb].append((tb, r_pu, x_pu))
        adj[tb].append((fb, r_pu, x_pu))

    parent = np.full(n_bus, -1, dtype=int)
    depth = np.full(n_bus, -1, dtype=int)
    children = {i: [] for i in range(n_bus)}
    feeder_rx = {}
    depth[slack] = 0
    queue = deque([slack])
    visited = {slack}
    while queue:
        bus = queue.popleft()
        for nb, r_pu, x_pu in adj[bus]:
            if nb not in visited:
                visited.add(nb)
                parent[nb] = bus
                children[bus].append(nb)
                depth[nb] = depth[bus] + 1
                feeder_rx[nb] = (r_pu, x_pu)
                queue.append(nb)
    return {"parent": parent, "children": children, "depth": depth, "feeder_rx": feeder_rx}


def _downstream(children: dict, bus: int) -> list:
    out, stack = [], [bus]
    while stack:
        b = stack.pop()
        out.append(b)
        stack.extend(children[b])
    return out


def ldf_matrices(net: pp.pandapowerNet) -> tuple[np.ndarray, np.ndarray]:
    """LDF matrices of Eq. (2).

    R[j, k] is the total resistance of the path shared by buses j and k from
    the substation (likewise X), so each line adds its r to every pair of
    buses downstream of it.
    """
    n_bus = len(net.bus)
    tree = build_tree(net)
    R = np.zeros((n_bus, n_bus))
    X = np.zeros((n_bus, n_bus))
    for bus in range(n_bus):
        if bus not in tree["feeder_rx"]:
            continue
        r, x = tree["feeder_rx"][bus]
        d = _downstream(tree["children"], bus)
        R[np.ix_(d, d)] += r
        X[np.ix_(d, d)] += x
    return R, X


def ldf_voltage(R: np.ndarray, X: np.ndarray, p_pu: np.ndarray, q_pu: np.ndarray,
                v0: float = 1.0) -> np.ndarray:
    """LDF voltage magnitude V = V0 - (R p + X q), with p, q the bus withdrawals."""
    return v0 - (R @ p_pu + X @ q_pu)


def sensitivity_scale(R: np.ndarray, X: np.ndarray, p_pu: np.ndarray,
                      q_pu: np.ndarray) -> np.ndarray:
    """S_j = sum_k |R_jk||p_k| + sum_k |X_jk||q_k|, Eq. (4)."""
    return np.abs(R) @ np.abs(p_pu) + np.abs(X) @ np.abs(q_pu)
