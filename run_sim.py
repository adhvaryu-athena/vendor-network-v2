import os, json, time, hashlib
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]
from itertools import product

import vendor_generator as vg
import customer_generator as cg
import classic_model as cm
import mean_field_model as mfm

from concurrent.futures import ProcessPoolExecutor, as_completed

# Inert globals at import time (so workers don't create folders)
RUN_TAG = None
BASE_OUTDIR = None
POP_CACHE_DIR = None
CACHE_POPULATIONS = True

def _init_paths():
    """Create run folders once, in the parent process."""
    global RUN_TAG, BASE_OUTDIR, POP_CACHE_DIR
    if RUN_TAG is None:
        RUN_TAG = time.strftime("run_%Y%m%d_%H%M%S")
        BASE_OUTDIR = os.path.join("out", RUN_TAG)
        POP_CACHE_DIR = os.path.join(BASE_OUTDIR, "pop_cache")
        os.makedirs(BASE_OUTDIR, exist_ok=True)
        os.makedirs(POP_CACHE_DIR, exist_ok=True)

# ------------- Top-level controls for a SIMPLE TEST RUN ------------------
REPEATS = 1
AFFINITY_DECAY = 1.0
SAVE_TIMESERIES = True
PARALLEL = False  # Easier to debug sequentially first
N_JOBS = 1        # Define N_JOBS even for sequential run
RUN_CLASSIC = True  # Toggle to disable the agent-based runs when focusing on mean-field
RUN_MEAN_FIELD = False

# ------- A minimal Sweep definition for testing ----------
SWEEP = {
    # Global choice knobs
    "T": [float(np.round(x, 4)) for x in np.linspace(0.2, 0.9, 35)],  # Temperature sweep
    "K": [7],            # Information visibility (number of vendors)
    "J": [float(np.round(x, 4)) for x in np.linspace(-0.6, 1.0, 35)],  # Social coupling (bandwagon ⇄ anti-conformity)
    
    # Other params (kept simple)
    "phi": [0.2],            # Bandwagon coupling for classic model
    "gamma": [0.05],         # Kept for classic model's visibility function
    "beta_digital": [1.0],

    # Market size/structure
    "num_customers": [200],
    "num_vendors": [40],
    "local_fraction": [0.7],

    # Dynamics
    "rounds": [12],

    # Vendor generation
    "rho": [1.1],
    "sigma": [0.15],
}

# Optional: pin the numpy seed for reproducibility at the *run* level
BASE_SEED = 42

# ----------------------------------------------------------------------

# Expected schemas (relaxed: we only assert the keys we actually use downstream)
EXPECTED_CUSTOMER_COLS = {
    "id", "location_x", "location_y", "alpha", "urgency", "delta", "social_influence"
}
EXPECTED_VENDOR_COLS = {
    "id", "type", "location_x", "location_y", "price", "rating", "urgency_delay"
    # 'is_local' is optional; market_model can infer from 'type'
}

def _coerce_df(obj, fallback_path, expected_cols, who):
    """
    Accepts either a DataFrame or None. If None, tries to read fallback_path.
    Validates required columns.
    """
    if isinstance(obj, pd.DataFrame):
        df = obj
    else:
        if not os.path.exists(fallback_path):
            raise RuntimeError(
                f"{who} returned None and fallback file not found: {fallback_path}.\n"
                f"Either: (1) make {who} return a DataFrame, or (2) ensure it writes {fallback_path}."
            )
        df = pd.read_csv(fallback_path)

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{who} missing columns {missing}. Got columns: {list(df.columns)}")
    return df


def set_global_seed(seed: int):
    np.random.seed(seed % (2**32 - 1))

def scenario_key(params: dict) -> str:
    """
    A 'scenario' is what determines the underlying population realization.
    We hold this fixed while sweeping T, gamma, phi, etc.
    """
    # Only keys that affect *generation* should go here:
    keys = ["num_customers", "num_vendors", "local_fraction", "rho", "sigma"]
    tpl = tuple((k, params[k]) for k in keys)
    return hashlib.md5(json.dumps(tpl).encode()).hexdigest()[:10]

def expand_sweep(sweep_dict):
    # Cartesian product over all lists
    keys = sorted(sweep_dict.keys())
    values = [sweep_dict[k] for k in keys]
    for combo in product(*values):
        yield {k: v for k, v in zip(keys, combo)}

def draw_choices_rowwise(P, rng=None):
    """
    Vectorized categorical draw: each row in P sums to 1.
    Returns index of chosen vendor per customer.
    """
    if rng is None: rng = np.random
    N, M = P.shape
    cdf = np.cumsum(P, axis=1)
    r = rng.random((N, 1))
    # first column where cdf >= r
    choices = (cdf >= r).argmax(axis=1)
    return choices

def metrics_from_counts(counts_vec, vendor_is_local):
    """
    counts_vec: (M,) integer choices this round
    vendor_is_local: (M,) 0/1
    Returns: dict of summary metrics
    """
    M = len(counts_vec)
    N = counts_vec.sum()
    if N == 0:
        return dict(local_share=np.nan, top_share=np.nan, hhi=np.nan, entropy=np.nan)

    p = counts_vec / N
    eps = 1e-12
    entropy = -np.sum(p * np.log(p + eps))
    hhi = np.sum(p**2)
    top_share = p.max()
    local_share = (counts_vec[vendor_is_local == 1].sum()) / N
    return dict(local_share=local_share, top_share=top_share, hhi=hhi, entropy=entropy)

def cache_population_for_scenario(params, scen_hash, seed):
    """Generate customers/vendors for a scenario and save to disk (idempotent + robust)."""
    cust_path = os.path.join(POP_CACHE_DIR, f"{scen_hash}_customers.csv")
    vend_path = os.path.join(POP_CACHE_DIR, f"{scen_hash}_vendors.csv")
    if os.path.exists(cust_path) and os.path.exists(vend_path):
        return cust_path, vend_path

    set_global_seed(seed)

    # Call generators (they may return DF or None in older versions)
    vendors_ret = vg.generate_vendors(
        params["num_vendors"],
        local_fraction=float(params["local_fraction"]),
        rho=float(params["rho"]),
        sigma=float(params["sigma"]),
    )
    customers_ret = cg.generate_customers(params["num_customers"])

    # Determine their native fallback outputs (legacy behavior)
    # Adjust these if your generators write to a different default path.
    legacy_cust = os.path.join("data", "customers.csv")
    legacy_vend = os.path.join("data", "vendors.csv")

    vendors = _coerce_df(vendors_ret, legacy_vend, EXPECTED_VENDOR_COLS, "vendor_generator.generate_vendors")
    customers = _coerce_df(customers_ret, legacy_cust, EXPECTED_CUSTOMER_COLS, "customer_generator.generate_customers")

    # Normalize minimal dtypes
    if "id" in customers: customers["id"] = customers["id"].astype(int)
    if "id" in vendors:   vendors["id"]   = vendors["id"].astype(int)
    if "type" in vendors: vendors["type"] = vendors["type"].astype(str)

    customers.to_csv(cust_path, index=False)
    vendors.to_csv(vend_path, index=False)
    return cust_path, vend_path


def run_classic_sim_wrapper(customers, vendors, params, seed):
    """
    Runs one classic (agent-based) simulation.
    """
    set_global_seed(seed)

    N, M = len(customers), len(vendors)
    vendor_is_local = (vendors["type"].to_numpy() == "local").astype(int)
    # State across rounds
    affinities = np.zeros((N, M), dtype=float)
    prev_counts = np.zeros(M, dtype=int)
    knowledge = None

    # per-round logs
    ts_records = []

    for t in range(int(params["rounds"])):
        # Correctly call the visibility mask function from the classic model itself
        knowledge = cm.update_visibility_mask(customers, vendors, params["gamma"], knowledge)

        # compute utilities and choice probabilities (masked softmax)
        U = cm.compute_utility_matrix(
            customers, vendors,
            prev_counts=prev_counts,
            phi=params["phi"],
            affinities=affinities,
        )
        P = cm.softmax_choice(U, params["T"], mask=knowledge)

        # make choices
        choices = draw_choices_rowwise(P)
        counts = np.bincount(choices, minlength=M)

        # update affinity with decay
        if AFFINITY_DECAY < 1.0:
            affinities *= AFFINITY_DECAY
        # increment chosen edges
        rows = np.arange(N)
        affinities[rows, choices] += 1.0

        # metrics & logging
        met = metrics_from_counts(counts, vendor_is_local)
        ts_records.append({
            "round": t,
            "local_share": met["local_share"],
            "top_share": met["top_share"],
            "hhi": met["hhi"],
            "entropy": met["entropy"],
        })

        # next round bandwagon uses *this* round's counts
        prev_counts = counts

    ts_df = pd.DataFrame(ts_records)

    # summaries: final and late-average (last 5 rounds or all if <5)
    k = min(5, len(ts_df))
    late = ts_df.tail(k).mean(numeric_only=True).to_dict()
    final = ts_df.tail(1).to_dict("records")[0]

    summary = {
        "final_local_share": final["local_share"],
        "final_top_share": final["top_share"],
        "final_hhi": final["hhi"],
        "final_entropy": final["entropy"],
        "late_local_share": late["local_share"],
        "late_top_share": late["top_share"],
        "late_hhi": late["hhi"],
        "late_entropy": late["entropy"],
    }

    return summary, (ts_df if SAVE_TIMESERIES else None)

def run_meanfield_sim_wrapper(customers, vendors, params, seed):
    """
    Runs one mean-field simulation.
    """
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    # Convert pandas dataframes to list of dataclasses for the mean-field model
    customer_list = [mfm.Customer(id=row.id, xy=(row.location_x, row.location_y)) for row in customers.itertuples()]
    vendor_list = [mfm.Vendor(id=row.id, s=1 if row.type == 'digital' else -1, price=row.price, rating=row.rating, assortment=0.0, xy=(row.location_x, row.location_y)) for row in vendors.itertuples()]

    # Precompute base utilities (without J-coupling)
    base_U = mfm.precompute_base_utilities(customer_list, vendor_list, beta={}, alpha={}, delta=0)

    # Solve the fixed point equation
    result = mfm.solve_fixed_point(
        rng=rng,
        BASE_U=base_U,
        vendors=vendor_list,
        T=params["T"],
        J=params["J"],
        mask_mode="K",
        mask_value=params["K"],
        return_history=SAVE_TIMESERIES,
    )

    shares = np.asarray(result["shares"], dtype=float)
    if shares.sum() > 0:
        shares = shares / shares.sum()
    eps = 1e-12
    entropy = -np.sum(shares * np.log(shares + eps))
    hhi = np.sum(shares ** 2)
    top_share = shares.max() if len(shares) else np.nan

    # Extract and return summary metrics (mean-field has no temporal dynamics, so
    # we mirror 'late_*' fields to the fixed-point outcome for downstream code parity.)
    summary = {
        "final_local_share": 1 - (result["m"] + 1) / 2,  # Approximate from magnetization
        "final_top_share": top_share,
        "final_hhi": hhi,
        "final_entropy": entropy,
        "late_local_share": 1 - (result["m"] + 1) / 2,
        "late_top_share": top_share,
        "late_hhi": hhi,
        "late_entropy": entropy,
        "converged": result["converged"],
        "iters": result["iters"],
    }
    history_df = None
    if SAVE_TIMESERIES and "history" in result:
        history_df = pd.DataFrame(result["history"])
    return summary, history_df


def _worker_run(job):
    """
    job dict contains: scen_hash, cust_path, vend_path, params, repeat, seeds, flags.
    Loads population, runs one sim, returns a summary row (and optionally writes timeseries).
    """
    set_global_seed(job["seed_for_run"])
    customers = pd.read_csv(job["cust_path"])
    vendors   = pd.read_csv(job["vend_path"])

    # Ensure dtypes we rely on (optional, safe if CSV already fine)
    customers["id"] = customers["id"].astype(int)
    vendors["id"]   = vendors["id"].astype(int)
    # If 'type' got read as category, ensure string:
    vendors["type"] = vendors["type"].astype(str)

    # Model parameter passing
    params = job["params"]
    rows = []

    if job.get("run_classic", True):
        classic_summary, classic_ts = run_classic_sim_wrapper(customers, vendors, params, job["seed_for_run"])
        classic_row = dict(job["params"])
        classic_row.update({"model_type": "classic", "repeat": job["repeat"], **classic_summary})
        rows.append(classic_row)
        if job.get("save_ts") and classic_ts is not None:
            ts_df = classic_ts.copy()
            ts_df["model_type"] = "classic"
            for key, val in job["params"].items():
                ts_df[key] = val
            ts_df["repeat"] = job["repeat"]
            ts_df["scenario_hash"] = job["scen_hash"]
            ts_df["seed_for_run"] = job["seed_for_run"]
            params_hash = hashlib.md5(json.dumps(job["params"], sort_keys=True).encode()).hexdigest()[:8]
            ts_fname = f"classic_{job['scen_hash']}_{params_hash}_rep{job['repeat']}.csv"
            ts_path = os.path.join(job["ts_outdir"], ts_fname)
            ts_df.to_csv(ts_path, index=False)

    # Run Mean-Field Model
    if job.get("run_mean_field", True):
        meanfield_summary, meanfield_ts = run_meanfield_sim_wrapper(customers, vendors, params, job["seed_for_run"])
        meanfield_row = dict(job["params"])
        meanfield_row.update({"model_type": "mean_field", "repeat": job["repeat"], **meanfield_summary})
        rows.append(meanfield_row)
        if job.get("save_ts") and meanfield_ts is not None:
            ts_df = meanfield_ts.copy()
            ts_df["model_type"] = "mean_field"
            for key, val in job["params"].items():
                ts_df[key] = val
            ts_df["repeat"] = job["repeat"]
            ts_df["scenario_hash"] = job["scen_hash"]
            ts_df["seed_for_run"] = job["seed_for_run"]
            params_hash = hashlib.md5(json.dumps(job["params"], sort_keys=True).encode()).hexdigest()[:8]
            ts_fname = f"mean_field_{job['scen_hash']}_{params_hash}_rep{job['repeat']}.csv"
            ts_path = os.path.join(job["ts_outdir"], ts_fname)
            ts_df.to_csv(ts_path, index=False)

    return rows


def main():
    set_global_seed(BASE_SEED)
    all_param_combos = list(expand_sweep(SWEEP))

    # Identify unique scenarios (things that define the underlying population)
    scenario_keys = {}
    for p in all_param_combos:
        scen_hash = scenario_key(p)
        if scen_hash not in scenario_keys:
            scenario_keys[scen_hash] = p

    # Pre-generate & cache each scenario population once
    if CACHE_POPULATIONS:
        for scen_hash, p in scenario_keys.items():
            seed_for_gen = BASE_SEED + int(hashlib.md5(f"{scen_hash}".encode()).hexdigest(), 16) % 10_000_000
            cache_population_for_scenario(p, scen_hash, seed_for_gen)

    # Build all jobs
    ts_outdir = os.path.join(BASE_OUTDIR, "timeseries")
    os.makedirs(ts_outdir, exist_ok=True)
    jobs = []
    for p in all_param_combos:
        scen_hash = scenario_key(p)
        cust_path = os.path.join(POP_CACHE_DIR, f"{scen_hash}_customers.csv")
        vend_path = os.path.join(POP_CACHE_DIR, f"{scen_hash}_vendors.csv")

        # if not caching, generate on-the-fly per repeat (falls back to your old behavior)
        if not CACHE_POPULATIONS:
            seed_for_gen = BASE_SEED + int(hashlib.md5(f"{scen_hash}".encode()).hexdigest(), 16) % 10_000_000
            cust_path, vend_path = cache_population_for_scenario(p, scen_hash, seed_for_gen)

        for rep in range(REPEATS):
            seed_for_run = BASE_SEED + int(hashlib.md5(
                f"{scen_hash}-{rep}-{p['T']}-{p['gamma']}-{p['phi']}".encode()
            ).hexdigest(), 16) % 10_000_000

            jobs.append({
                "scen_hash": scen_hash,
                "cust_path": cust_path,
                "vend_path": vend_path,
                "params": p,
                "repeat": rep,
                "seed_for_run": seed_for_run,
                "save_ts": SAVE_TIMESERIES,
                "ts_outdir": ts_outdir,
                "run_classic": RUN_CLASSIC,
                "run_mean_field": RUN_MEAN_FIELD,
            })

    # Execute (parallel or sequential)
    results = []
    if PARALLEL and N_JOBS > 1:
        with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
            futs = [ex.submit(_worker_run, job) for job in jobs]
            for fut in as_completed(futs):
                results.extend(fut.result()) # Use extend for list of rows
    else:
        for job in jobs:
            results.extend(_worker_run(job))

    # Save summary CSV (same as before)
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(BASE_OUTDIR, "results_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # ---------------- Post-processing: susceptibility wrt T ----------------
    group_keys = [k for k in SWEEP.keys() if k != "T"] + ["repeat", "model_type"]
    summary_df = summary_df.sort_values(group_keys + ["T"]).reset_index(drop=True)

    sus_records = []
    for _, g in summary_df.groupby(group_keys, dropna=False):
        Ts = g["T"].values
        Ls = g["late_local_share"].values
        if len(Ts) < 2:
            continue
        dL = np.empty_like(Ls)
        dL[1:-1] = (Ls[2:] - Ls[:-2]) / (Ts[2:] - Ts[:-2])
        dL[0]     = (Ls[1]  - Ls[0])  / (Ts[1]  - Ts[0])
        dL[-1]    = (Ls[-1] - Ls[-2]) / (Ts[-1] - Ts[-2])

        for Tval, Lval, chi in zip(Ts, Ls, dL):
            rec = {k: g.iloc[0][k] for k in group_keys}
            rec.update({"T": Tval, "late_local_share": Lval, "susceptibility_dL_dT": chi})
            sus_records.append(rec)

    sus_df = pd.DataFrame(sus_records)
    sus_path = os.path.join(BASE_OUTDIR, "results_susceptibility.csv")
    if len(sus_df):
        sus_df.to_csv(sus_path, index=False)

    # Provenance of the sweep
    with open(os.path.join(BASE_OUTDIR, "sweep_spec.json"), "w") as f:
        json.dump({"SWEEP": SWEEP, "REPEATS": REPEATS, "AFFINITY_DECAY": AFFINITY_DECAY,
                   "SAVE_TIMESERIES": SAVE_TIMESERIES, "BASE_SEED": BASE_SEED,
                   "PARALLEL": PARALLEL, "N_JOBS": N_JOBS,
                   "CACHE_POPULATIONS": CACHE_POPULATIONS}, f, indent=2)

    print(f"[done] Summary -> {summary_path}")
    if len(sus_df):
        print(f"[done] Susceptibility -> {sus_path}")
    print(f"[done] Run folder: {BASE_OUTDIR}")


if __name__ == "__main__":
    _init_paths()
    set_global_seed(BASE_SEED)
    main()
