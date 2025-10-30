# utility.py
# Shared utility, mask, and softmax functions for both classic and mean-field market models.

import numpy as np


def softmax_stable(u: np.ndarray, T: float) -> np.ndarray:
    if T <= 0:
        T = 1e-6
    z = u / T
    z -= z.max()
    ex = np.exp(z)
    return ex / (ex.sum() + 1e-12)


def build_consideration_set_K(rng: np.random.Generator, vendor_ids: np.ndarray, K: int, exposure_weights=None) -> np.ndarray:
    K = max(1, min(K, vendor_ids.size))
    if exposure_weights is None:
        return rng.choice(vendor_ids, size=K, replace=False)
    else:
        w = exposure_weights / (exposure_weights.sum() + 1e-12)
        return rng.choice(vendor_ids, size=K, replace=False, p=w)

def compute_utility(
    customer,
    vendor,
    affinity=0.0,
    prev_vendor_count=0.0,
    phi=0.0,
    J=0.0,
    m=0.0,
    noise=None,
    H=None,
    vendor_type=None
):
    """
    Physics-inspired customer-vendor utility function (scalar):
    U_ij = -alpha_i * price_j + rating_j - urgency_i * delay_ij
           + delta_i * log(1 + affinity_ij)
           + phi * log(1 + n_j/N)    [bandwagon/network]
           + J * s_j * m             [mean-field/social field]
           + s_i * noise             [social shock]
           + H_j * L_j               [local vendor bias]

    All fields can be attribute or dict key. Only needed params must be present.
    """
    # Allow both dict and object
    def get(obj, key, default=0.0):
        if hasattr(obj, key): return getattr(obj, key)
        if isinstance(obj, dict) and key in obj: return obj[key]
        return default

    alpha = get(customer, 'alpha')
    urgency = get(customer, 'urgency')
    delta = get(customer, 'delta')
    social_influence = get(customer, 'social_influence')
    price = get(vendor, 'price')
    rating = get(vendor, 'rating')

    # Time/delay term selection (local or digital)
    vtype = vendor_type if vendor_type is not None else get(vendor, 'type', None)
    if vtype == 'digital' or get(vendor,'is_local',0)==0:
        delay = get(vendor, 'urgency_delay', 0.0)
    else:
        # Assume customer/vendor "location_x","location_y" exist
        cx, cy = get(customer, 'location_x', 0.0), get(customer, 'location_y', 0.0)
        vx, vy = get(vendor, 'location_x', 0.0), get(vendor, 'location_y', 0.0)
        delay = np.linalg.norm([cx-vx, cy-vy])

    # Bandwagon, affinity, mean-field, noise, local bias
    u = -alpha * price + rating - urgency * delay
    if delta:
        u += delta * np.log1p(affinity)
    if phi:
        u += phi * np.log1p(prev_vendor_count)
    if J:
        s_j = get(vendor, 's', None)
        if s_j is None:
            if vtype == 'digital':
                s_j = +1
            elif vtype == 'local':
                s_j = -1
            else:
                s_j = get(vendor, 'is_local', 0)*-2+1
        u += J * s_j * m
    if noise is not None:
        u += social_influence * noise
    if H is not None:
        is_local_j = get(vendor, 'is_local', vtype=='local')
        u += H * is_local_j
    return float(u)
