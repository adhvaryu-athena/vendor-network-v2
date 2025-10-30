# vendor_choice_model.py
# Physics-grounded customer–vendor choice with Ising–Boltzmann mean-field,
# information masks, and phase/hysteresis analysis.
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple, Dict, List, Optional
import math
from utility import compute_utility, softmax_stable, build_consideration_set_K

# -----------------------------
# Data classes
# -----------------------------
@dataclass
class Vendor:
    id: int
    s: int                 # +1 digital, -1 local
    price: float
    rating: float
    assortment: float
    xy: Tuple[float, float]  # position (for locals); for digital it's optional but kept for uniformity
    exposure: float = 1.0    # exposure weight for masks


@dataclass
class Customer:
    id: int
    xy: Tuple[float, float]
    # Add heterogeneity knobs if needed later


# -----------------------------
# Utility + features
# -----------------------------
def pairwise_features(i: Customer, j: Vendor) -> Dict[str, float]:
    """
    Construct X_ij. Keep it simple but realistic:
    - local travel time ~ Euclidean distance * 25 minutes/km (scaled)
    - digital delivery time ~ base
    """
    xi, yi = i.xy
    xj, yj = j.xy
    dist = math.hypot(xi - xj, yi - yj)  # [0, sqrt(2)]
    km_per_unit = 3.0
    minutes_per_km = 25.0
    local_travel_min = dist * km_per_unit * minutes_per_km
    digital_delivery_min = 180.0  # 3 hours baseline

    return {
        "local_travel_min": local_travel_min,
        "digital_delivery_min": digital_delivery_min,
    }


def baseline_utility(
    Xij: Dict[str, float],
    vendor: Vendor,
    beta: Dict[str, float],
    alpha: Dict[str, float],
    delta: float
) -> float:
    """
    U_ij^(0) = beta^T X_ij + alpha^T A_j + delta * s_j
    """
    if vendor.s == +1:  # digital
        time_term = beta["digital_minutes"] * Xij["digital_delivery_min"]
    else:               # local
        time_term = beta["local_minutes"] * Xij["local_travel_min"]

    attr_term = (
        alpha["price"] * vendor.price +
        alpha["rating"] * vendor.rating +
        alpha["assort"] * vendor.assortment
    )
    return time_term + attr_term + delta * vendor.s


# -----------------------------
# Information masks
# -----------------------------
def build_consideration_set_K(
    rng: np.random.Generator,
    vendor_ids: np.ndarray,
    K: int,
    exposure_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    K = max(1, min(K, vendor_ids.size))
    if exposure_weights is None:
        return rng.choice(vendor_ids, size=K, replace=False)
    else:
        w = exposure_weights / (exposure_weights.sum() + 1e-12)
        return rng.choice(vendor_ids, size=K, replace=False, p=w)


def build_consideration_set_p(
    rng: np.random.Generator,
    vendor_ids: np.ndarray,
    p: float
) -> np.ndarray:
    p = float(np.clip(p, 0.0, 1.0))
    mask = rng.random(vendor_ids.size) < p
    C = vendor_ids[mask]
    if C.size == 0:
        C = rng.choice(vendor_ids, size=1, replace=False)
    return C


# -----------------------------
# Stable softmax
# -----------------------------
def softmax_stable(u: np.ndarray, T: float) -> np.ndarray:
    if T <= 0:
        T = 1e-6
    z = u / T
    z -= z.max()
    ex = np.exp(z)
    return ex / (ex.sum() + 1e-12)


# -----------------------------
# Generators
# -----------------------------
def generate_customers(N: int, rng: np.random.Generator) -> List[Customer]:
    xy = rng.random((N, 2))
    return [Customer(id=i, xy=(xy[i, 0], xy[i, 1])) for i in range(N)]


def generate_vendors(
    M: int,
    frac_digital: float,
    rng: np.random.Generator
) -> List[Vendor]:
    s = (rng.random(M) < frac_digital).astype(int)
    s[s == 0] = -1
    xy = rng.random((M, 2))

    price = rng.normal(loc=100.0, scale=10.0, size=M)
    rating = np.clip(rng.normal(loc=4.2, scale=0.4, size=M), 1.0, 5.0)
    assort = rng.normal(loc=1.0, scale=0.2, size=M)
    exposure = np.ones(M)

    vendors = []
    for j in range(M):
        vendors.append(Vendor(
            id=j, s=int(s[j]),
            price=float(price[j]),
            rating=float(rating[j]),
            assortment=float(assort[j]),
            xy=(xy[j, 0], xy[j, 1]),
            exposure=float(exposure[j])
        ))
    return vendors


# -----------------------------
# Precompute baseline utilities
# -----------------------------
def precompute_base_utilities(
    customers: List[Customer],
    vendors: List[Vendor],
    beta: Dict[str, float],
    alpha: Dict[str, float],
    delta: float
) -> np.ndarray:
    N, M = len(customers), len(vendors)
    U0 = np.zeros((N, M), dtype=float)
    # All features/terms now come from compute_utility
    for i, cust in enumerate(customers):
        for j, vend in enumerate(vendors):
            U0[i, j] = compute_utility(
                cust, vend,
                affinity=0,
                prev_vendor_count=0,
                phi=0,
                J=0, m=0,
                noise=None, H=None,
                vendor_type=getattr(vend, 's', None)
            )
    return U0


# -----------------------------
# Fixed-point solver (mean-field)
# -----------------------------
def solve_fixed_point(
    rng: np.random.Generator,
    BASE_U: np.ndarray,
    vendors: List[Vendor],
    T: float,
    J: float,
    mask_mode: Literal["K", "p"],
    mask_value: float,
    exposure_weights: Optional[np.ndarray] = None,
    max_iters: int = 500,
    eps: float = 1e-6,
    damping: float = 0.5,
    mean_field: bool = True,
    # NEW: allow reuse of prebuilt masks and custom initial m
    consideration_sets: Optional[List[np.ndarray]] = None,
    m_init: float = 0.0
) -> Dict[str, object]:
    """
    Compute equilibrium m and vendor shares.
    If 'consideration_sets' is provided, it is used as-is (must be length N).
    """
    N, M = BASE_U.shape
    vendor_ids = np.arange(M)
    s = np.array([v.s for v in vendors], dtype=int)

    # Build or reuse consideration sets
    if consideration_sets is None:
        C_list: List[np.ndarray] = []
        for _i in range(N):
            if mask_mode == "K":
                C = build_consideration_set_K(rng, vendor_ids, K=int(mask_value), exposure_weights=exposure_weights)
            else:
                C = build_consideration_set_p(rng, vendor_ids, p=float(mask_value))
            C_list.append(C)
    else:
        C_list = consideration_sets

    # Initialize m (can be warm-started)
    m_prev = 2.0
    m = float(np.clip(m_init, -1.0, 1.0))

    shares = np.zeros(M, dtype=float)
    it = 0

    for it in range(max_iters):
        m_prev = m
        signed_sum = 0.0
        shares.fill(0.0)

        if mean_field:
            for i in range(N):
                C = C_list[i]
                u = BASE_U[i, C] + J * s[C] * m
                p = softmax_stable(u, T)
                signed_sum += np.dot(s[C], p)
                np.add.at(shares, C, p)
        else:
            for i in range(N):
                C = C_list[i]
                u = BASE_U[i, C] + J * s[C] * m
                p = softmax_stable(u, T)
                j_idx = rng.choice(np.arange(C.size), p=p)
                j = C[j_idx]
                signed_sum += s[j]
                shares[j] += 1.0

        m_new = signed_sum / N
        m = (1.0 - damping) * m + damping * m_new

        if abs(m - m_prev) <= eps:
            break

    if not mean_field:
        shares = shares / N
    else:
        shares = shares / N

    return {
        "m": float(m),
        "iters": it + 1,
        "converged": abs(m - m_prev) <= eps,
        "shares": shares
    }


# -----------------------------
# Helpers to prebuild masks
# -----------------------------
def build_masks_bank(
    rng: np.random.Generator,
    N: int,
    M: int,
    info_grid: np.ndarray,
    mask_mode: Literal["K", "p"],
    exposure_weights: Optional[np.ndarray],
    replicas: int = 10
) -> Dict[float, List[List[np.ndarray]]]:
    """
    For each info value (K or p), prebuild R replicas of consideration sets.
    Returns dict: info_val -> [replica r -> list of C_i arrays]
    """
    vendor_ids = np.arange(M)
    bank: Dict[float, List[List[np.ndarray]]] = {}
    for info in info_grid:
        info = float(info)
        reps: List[List[np.ndarray]] = []
        for _r in range(replicas):
            C_list: List[np.ndarray] = []
            for _i in range(N):
                if mask_mode == "K":
                    C = build_consideration_set_K(rng, vendor_ids, K=int(info), exposure_weights=exposure_weights)
                else:
                    C = build_consideration_set_p(rng, vendor_ids, p=float(info))
                C_list.append(C)
            reps.append(C_list)
        bank[info] = reps
    return bank


# -----------------------------
# Phase diagram sweep (denoised)
# -----------------------------
def phase_diagram(
    rng: np.random.Generator,
    BASE_U: np.ndarray,
    vendors: List[Vendor],
    J: float,
    T_grid: np.ndarray,
    info_grid: np.ndarray,
    mask_mode: Literal["K", "p"],
    exposure_weights: Optional[np.ndarray] = None,
    replicas: int = 10,
    warmstart: bool = True,
    return_std: bool = True,
    **solver_kwargs
) -> Dict[str, np.ndarray]:
    """
    Sweep T x (K or p); reuse masks across T for each info value; average over R replicas.
    Returns mean maps; optionally std maps and iteration maps.
    """
    N, M = BASE_U.shape
    # Prebuild masks once per info value
    masks_bank = build_masks_bank(
        rng, N, M, info_grid, mask_mode, exposure_weights, replicas=replicas
    )

    m_map = np.zeros((T_grid.size, info_grid.size), dtype=float)
    smax_map = np.zeros_like(m_map)
    iters_map = np.zeros_like(m_map)

    m_std = np.zeros_like(m_map) if return_std else None
    smax_std = np.zeros_like(m_map) if return_std else None

    # Optional warm-start state across neighboring cells
    last_m_for_col = np.zeros(T_grid.size, dtype=float)

    for b, info in enumerate(info_grid):
        info = float(info)
        for a, T in enumerate(T_grid):
            T = float(T)
            m_vals, smax_vals, iters_vals = [], [], []

            # choose warm-start from previous T in this column
            m0 = last_m_for_col[a] if (warmstart and a > 0) else 0.0

            for C_list in masks_bank[info]:
                res = solve_fixed_point(
                    rng=rng,
                    BASE_U=BASE_U,
                    vendors=vendors,
                    T=T,
                    J=J,
                    mask_mode=mask_mode,
                    mask_value=info,
                    exposure_weights=exposure_weights,
                    consideration_sets=C_list,  # reuse mask
                    m_init=m0,                  # warm start
                    **solver_kwargs
                )
                m_vals.append(res["m"])
                smax_vals.append(res["shares"].max() if res["shares"].size else np.nan)
                iters_vals.append(res["iters"])

            m_mean = float(np.mean(m_vals))
            smax_mean = float(np.mean(smax_vals))
            m_map[a, b] = m_mean
            smax_map[a, b] = smax_mean
            iters_map[a, b] = float(np.mean(iters_vals))

            if return_std:
                m_std[a, b] = float(np.std(m_vals))
                smax_std[a, b] = float(np.std(smax_vals))

            # carry forward warm-start
            last_m_for_col[a] = m_mean

    out = {"m_map": m_map, "smax_map": smax_map, "iters_map": iters_map}
    if return_std:
        out.update({"m_std": m_std, "smax_std": smax_std})
    return out


# -----------------------------
# Hysteresis sweep in delta (baseline tilt)
# -----------------------------
def hysteresis_sweep(
    rng: np.random.Generator,
    customers: List[Customer],
    vendors: List[Vendor],
    beta: Dict[str, float],
    alpha: Dict[str, float],
    J: float,
    T: float,
    mask_mode: Literal["K", "p"],
    mask_value: float,
    delta_min: float,
    delta_max: float,
    delta_step: float,
    exposure_weights: Optional[np.ndarray] = None,
    # NEW: replicas + reuse masks for stability
    replicas: int = 1,
    **solver_kwargs
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Sweep delta upwards and downwards, warm-starting implicitly via damping.
    Averages over 'replicas' different masks for stability if >1.
    """
    N = len(customers)
    M = len(vendors)
    vendor_ids = np.arange(M)

    def single_mask() -> List[np.ndarray]:
        C_list = []
        for _i in range(N):
            if mask_mode == "K":
                C = build_consideration_set_K(rng, vendor_ids, int(mask_value), exposure_weights)
            else:
                C = build_consideration_set_p(rng, vendor_ids, float(mask_value))
            C_list.append(C)
        return C_list

    masks = [single_mask() for _ in range(max(1, replicas))]

    # Up-sweep
    up: List[Tuple[float, float]] = []
    deltas_up = np.arange(delta_min, delta_max + 1e-12, delta_step)
    m0 = 0.0
    for d in deltas_up:
        BASE_U = precompute_base_utilities(customers, vendors, beta, alpha, d)
        m_vals = []
        for C_list in masks:
            res = solve_fixed_point(
                rng, BASE_U, vendors, T, J,
                mask_mode=mask_mode, mask_value=mask_value,
                exposure_weights=exposure_weights,
                consideration_sets=C_list,
                m_init=m0,
                **solver_kwargs
            )
            m_vals.append(res["m"])
        m0 = float(np.mean(m_vals))
        up.append((float(d), m0))

    # Down-sweep
    down: List[Tuple[float, float]] = []
    deltas_down = np.arange(delta_max, delta_min - 1e-12, -delta_step)
    m0 = up[-1][1]
    for d in deltas_down:
        BASE_U = precompute_base_utilities(customers, vendors, beta, alpha, d)
        m_vals = []
        for C_list in masks:
            res = solve_fixed_point(
                rng, BASE_U, vendors, T, J,
                mask_mode=mask_mode, mask_value=mask_value,
                exposure_weights=exposure_weights,
                consideration_sets=C_list,
                m_init=m0,
                **solver_kwargs
            )
            m_vals.append(res["m"])
        m0 = float(np.mean(m_vals))
        down.append((float(d), m0))

    return {"up": up, "down": down}

# -----------------------------
# Phase diagram sweep over (T, J) with fixed information mask
# -----------------------------
def phase_diagram_TJ(
    rng: np.random.Generator,
    BASE_U: np.ndarray,
    vendors: List[Vendor],
    T_grid: np.ndarray,
    J_grid: np.ndarray,
    mask_mode: Literal["K", "p"],
    mask_value: float,
    exposure_weights: Optional[np.ndarray] = None,
    replicas: int = 10,
    warmstart: bool = True,
    return_std: bool = True,
    **solver_kwargs
) -> Dict[str, np.ndarray]:
    """
    Sweep T x J while keeping the information mask fixed (e.g., K=5).
    Reuses the same masks across the whole sweep and averages across `replicas`.
    Returns mean maps; optionally std maps and iteration maps.
    """
    N, M = BASE_U.shape

    # --- Prebuild masks once (fixed info mask) ---
    vendor_ids = np.arange(M)
    masks: List[List[np.ndarray]] = []
    for _r in range(replicas):
        C_list: List[np.ndarray] = []
        for _i in range(N):
            if mask_mode == "K":
                C = build_consideration_set_K(rng, vendor_ids, int(mask_value), exposure_weights)
            else:
                C = build_consideration_set_p(rng, vendor_ids, float(mask_value))
            C_list.append(C)
        masks.append(C_list)

    # --- Allocate maps ---
    m_map = np.zeros((T_grid.size, J_grid.size), dtype=float)
    smax_map = np.zeros_like(m_map)
    iters_map = np.zeros_like(m_map)

    m_std = np.zeros_like(m_map) if return_std else None
    smax_std = np.zeros_like(m_map) if return_std else None

    # Warm-start state across neighboring cells
    last_m_for_row = np.zeros(J_grid.size, dtype=float)

    for a, T in enumerate(T_grid):
        T = float(T)
        # continue from previous T row if warmstart
        for b, J in enumerate(J_grid):
            J = float(J)
            m0 = last_m_for_row[b] if (warmstart and a > 0) else 0.0

            m_vals, smax_vals, iters_vals = [], [], []
            for C_list in masks:
                res = solve_fixed_point(
                    rng=rng,
                    BASE_U=BASE_U,
                    vendors=vendors,
                    T=T,
                    J=J,
                    mask_mode=mask_mode,
                    mask_value=mask_value,
                    exposure_weights=exposure_weights,
                    consideration_sets=C_list,  # reuse mask
                    m_init=m0,                  # warm start
                    **solver_kwargs
                )
                m_vals.append(res["m"])
                smax_vals.append(res["shares"].max() if res["shares"].size else np.nan)
                iters_vals.append(res["iters"])

            m_mean = float(np.mean(m_vals))
            smax_mean = float(np.mean(smax_vals))

            m_map[a, b] = m_mean
            smax_map[a, b] = smax_mean
            iters_map[a, b] = float(np.mean(iters_vals))

            if return_std:
                m_std[a, b] = float(np.std(m_vals))
                smax_std[a, b] = float(np.std(smax_vals))

            # carry forward warm-start along the row (increasing J)
            last_m_for_row[b] = m_mean

    out = {"m_map": m_map, "smax_map": smax_map, "iters_map": iters_map}
    if return_std:
        out.update({"m_std": m_std, "smax_std": smax_std})
    return out


# -----------------------------
# Optional: two-basin bistability check at fixed (params, delta)
# -----------------------------
def bistability_check(
    rng: np.random.Generator,
    BASE_U: np.ndarray,
    vendors: List[Vendor],
    T: float,
    J: float,
    mask_mode: Literal["K", "p"],
    mask_value: float,
    exposure_weights: Optional[np.ndarray] = None,
    consideration_sets: Optional[List[np.ndarray]] = None,
    threshold: float = 0.05,
    **solver_kwargs
) -> Tuple[bool, float, float]:
    """
    Solve from m_init=+1 and m_init=-1. If the equilibria differ by > threshold, we flag bistability.
    """
    res_plus = solve_fixed_point(
        rng, BASE_U, vendors, T, J, mask_mode, mask_value,
        exposure_weights=exposure_weights,
        consideration_sets=consideration_sets,
        m_init=+1.0,
        **solver_kwargs
    )
    res_minus = solve_fixed_point(
        rng, BASE_U, vendors, T, J, mask_mode, mask_value,
        exposure_weights=exposure_weights,
        consideration_sets=consideration_sets,
        m_init=-1.0,
        **solver_kwargs
    )
    diff = abs(res_plus["m"] - res_minus["m"])
    return (diff > threshold, float(res_plus["m"]), float(res_minus["m"]))


# -----------------------------
# Example usage (CLI entry)
# -----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(25)
    N, M = 1000, 50
    customers = generate_customers(N, rng)
    vendors = generate_vendors(M, frac_digital=0.55, rng=rng)

    beta = {"local_minutes": -0.01, "digital_minutes": -0.003}
    alpha = {"price": -0.02, "rating": 0.6, "assort": 0.4}

    # Global knobs for single runs
    delta = 0.0
    J = 0.8
    T = 1.0

    BASE_U = precompute_base_utilities(customers, vendors, beta, alpha, delta)
    exposure_weights = np.array([v.exposure for v in vendors], dtype=float)

    # Solve once
    res = solve_fixed_point(
        rng, BASE_U, vendors, T=T, J=J,
        mask_mode="K", mask_value=5,
        exposure_weights=exposure_weights,
        max_iters=1200, eps=1e-7, damping=0.7, mean_field=True
    )
    print(f"m*={res['m']:.3f}, iters={res['iters']}, converged={res['converged']}, s_max={res['shares'].max():.3f}")

    # Phase diagram (denoised): T x J with fixed information mask (e.g., K=5)
    T_grid = np.linspace(0.01, 1.5, 40)
    J_grid = np.linspace(0.0, 2.0, 40)   # adjust range as needed
    fixed_K = 5.0                        # keep mask fixed for this sweep
    
    phase_TJ = phase_diagram_TJ(
        rng, BASE_U, vendors,
        T_grid=T_grid, J_grid=J_grid,
        mask_mode="K", mask_value=fixed_K,
        exposure_weights=exposure_weights,
        replicas=12,          # average across 12 mask replicas
        warmstart=True,
        return_std=True,
        max_iters=2000, eps=1e-6, damping=0.85, mean_field=True
    )
    
    # Save T×J maps
    np.savez(
        "phase_TxJ.npz",
        T_grid=T_grid,
        J_grid=J_grid,
        K_fixed=fixed_K,
        **phase_TJ
    )   
    
    # Hysteresis in delta at fixed (T, K, J)
    hyst = hysteresis_sweep(
        rng, customers, vendors, beta, alpha, J=J, T=0.8,
        mask_mode="K", mask_value=5,
        delta_min=-0.5, delta_max=0.5, delta_step=0.05,
        exposure_weights=exposure_weights,
        replicas=5,              # average across 5 masks for stability
        max_iters=2000, eps=1e-6, damping=0.85, mean_field=True
    )
    np.savez("hysteresis_delta.npz", up=np.array(hyst["up"]), down=np.array(hyst["down"]))
