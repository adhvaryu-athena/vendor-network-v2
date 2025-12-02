# classic_model.py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from utility import compute_utility, softmax_stable, build_consideration_set_K

# -----------------------------------------------------------------------------
# Notation used in comments:
#   - Customers i = 1..N, Vendors j = 1..M
#   - d_ij  : Euclidean distance between customer i and vendor j
#   - η_ij  : "urgency" term seen by customer i for vendor j
#             = d_ij              if vendor j is local
#             = τ_j (a constant)  if vendor j is digital, where τ_j = vendor['urgency_delay']
#   - p_j   : price of vendor j
#   - r_j   : rating of vendor j
#   - α_i   : price sensitivity of customer i  (larger α_i → stronger dislike of high price)
#   - u_i   : urgency (delay sensitivity) of customer i (larger u_i → more penalty for η_ij)
#   - δ_i   : reinforcement strength of customer i (weights log(1 + affinity))
#   - s_i   : social-influence amplitude for customer i (scales Gaussian noise)
#   - H_j   : extra utility bias for local vendors (only applied if vendor j is local)
#   - L_j   : 1 if vendor j is local, else 0
#   - ε_ij ~ N(0,1): social noise for (i, j)
#
# Utility per (i, j) is:
#   U_ij = - α_i * p_j + r_j - u_i * η_ij + δ_i * log(1 + A_ij) + (s_i * ε_ij) + H_j * L_j + (network effects)
# where A_ij is the affinity count (number of times customer i chose vendor j).
#
# Choice model (row-wise softmax with temperature T > 0):
#   P_ij = exp(U_ij / T) / Σ_k exp(U_ik / T)
# -----------------------------------------------------------------------------

# Core utility computation
def compute_utility_matrix(customers, vendors, prev_counts=None, phi=0.5, affinities=None, J=0.0, m=0.0):
    N = len(customers)
    M = len(vendors)
    U = np.zeros((N, M), dtype=float)
    if affinities is None:
        affinities = np.zeros((N, M), dtype=float)
    if prev_counts is None:
        prev_counts = np.zeros(M, dtype=float)
    for i in range(N):
        cust = customers.iloc[i] if isinstance(customers, pd.DataFrame) else customers[i]
        for j in range(M):
            vend = vendors.iloc[j] if isinstance(vendors, pd.DataFrame) else vendors[j]
            U[i, j] = compute_utility(
                cust, vend,
                affinity=affinities[i, j],
                prev_vendor_count=prev_counts[j],
                phi=phi,
                J=J, m=m,
                # optionally add noise, H, vendor_type
            )
    return U


# Update affinity matrix based on customer choices
def update_affinities(affinities, choices):
    """
    Update the affinity matrix by incrementing counts for chosen vendors.
    
    affinities: (N x M) matrix where affinities[i, j] = number of times customer i chose vendor j
    choices: array of length N where choices[i] = vendor chosen by customer i
    
    Returns updated affinities matrix.
    """
    affinities = affinities.copy()  # Don't modify in-place
    for i, chosen_vendor in enumerate(choices):
        affinities[i, chosen_vendor] += 1
    return affinities


# Softmax selection probabilities (across vendors)
def softmax_choice(U, T, mask=None, rng=None):
    """
    Row-wise softmax over vendors at temperature T.
    If mask (N x M) is provided, entries with mask==False are set to -inf (invisible options).
    Returns probabilities P (N x M).
    """
    if rng is None:
        rng = np.random

    U_ = np.array(U, dtype=float, copy=True)
    if mask is not None:
        # block invisible options
        U_[~mask] = -np.inf

    # Numerically stable softmax
    Z = U_ / T
    row_max = np.nanmax(Z, axis=1, keepdims=True)  # safe if rows are all -inf
    Z = Z - row_max
    exps = np.exp(Z)
    exps[np.isinf(U_)] = 0.0  # exp(-inf)=0
    denom = exps.sum(axis=1, keepdims=True)
    # Avoid divide-by-zero if a row is fully masked: fall back to uniform over visible vendors
    zero_rows = (denom.squeeze() == 0)
    if zero_rows.any():
        if mask is None:
            # no visible options? uniform over all M
            exps[zero_rows, :] = 1.0
        else:
            # uniform over visible subset
            vis_counts = mask[zero_rows, :].sum(axis=1, keepdims=True)
            # where nothing visible, default to uniform over all M
            fallback = (vis_counts == 0)
            exps[zero_rows & fallback, :] = 1.0
            # for rows with some visible, set those to 1.0
            exps[zero_rows & ~fallback, :] = mask[zero_rows & ~fallback, :].astype(float)
        denom = exps.sum(axis=1, keepdims=True)

    P = exps / denom
    return P

def update_visibility_mask(customers, vendors, gamma, knowledge_mask=None, rng=None):
    """
    Accumulating knowledge model:
      - Digitals: always visible
      - Locals: per round, new discovery with P = exp(-d^2 / (2*gamma^2)); once seen, stays visible
    Returns: boolean mask (N x M), True = visible/known
    """
    if rng is None:
        rng = np.random

    N, M = len(customers), len(vendors)
    if knowledge_mask is None:
        knowledge_mask = np.zeros((N, M), dtype=bool)

    is_digital = (vendors['type'].to_numpy() == 'digital')
    is_local   = ~is_digital

    # Digitals always visible
    if is_digital.any():
        knowledge_mask[:, is_digital] = True

    # Locals: discover by distance
    if is_local.any():
        cust_xy  = customers[['location_x','location_y']].to_numpy()
        vend_xyL = vendors.loc[is_local, ['location_x','location_y']].to_numpy()
        D = cdist(cust_xy, vend_xyL)  # (N, M_local)
        # discovery probability kernel
        P = np.exp(-(D**2) / (2.0 * (gamma**2)))
        discover = rng.random(size=P.shape) < P
        knowledge_mask[:, is_local] |= discover

    return knowledge_mask


# Compute vendor market share
def compute_vendor_market_share(choices, num_vendors):
    return np.bincount(choices, minlength=num_vendors)

# Compute fraction of customers choosing local vendors
def compute_local_share(choices, vendors=None, vendor_is_local=None):
    """
    Compute fraction of customers choosing local vendors.

    Parameters
    ----------
    choices : array-like
        Indices of vendors chosen by each customer (length N)
    vendors : pandas.DataFrame, optional
        Vendor frame with a 'type' column (values include 'local'/'digital').
        Used to infer local vendors if vendor_is_local is not provided.
    vendor_is_local : array-like, optional
        Boolean/int array of shape (M,) with 1/True where vendor is local.

    Returns
    -------
    float
        Fraction of customers choosing local vendors. Returns 0.0 if choices empty.
    """
    if vendor_is_local is None:
        if vendors is None:
            raise ValueError("compute_local_share: provide either 'vendors' DataFrame or 'vendor_is_local' array")
        vendor_is_local = (vendors['type'].astype(str) == 'local').astype(int).to_numpy()

    vendor_is_local = np.asarray(vendor_is_local, dtype=int)
    local_ids = np.where(vendor_is_local == 1)[0]
    if len(choices) == 0:
        return 0.0
    local_choice_count = sum(1 for c in choices if c in local_ids)
    return local_choice_count / len(choices)

# Compute market concentration (sigma)
def compute_concentration(vendor_counts):
    total = vendor_counts.sum()
    if total == 0:
        return 0.0
    shares = vendor_counts / total
    max_share = np.max(shares)
    N = len(vendor_counts)
    return (max_share - 1/N) / (1 - 1/N) if N > 1 else 0.0