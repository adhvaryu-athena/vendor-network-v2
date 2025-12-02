#!/usr/bin/env python3
"""
plot_phase.py
-------------
Zero-arg utility to crawl result runs and materialize phase plots if missing.

How to use:
- Edit the CONFIG section below (globs, metrics, axes, slices, etc.)
- Run: python plot_phase.py
- The script scans each matched run folder (e.g., out/run_2025...), finds
  results_summary.csv (and optional results_susceptibility.csv), and produces
  heatmaps per slice + metric into <run>/plots/. Existing figures are skipped
  unless OVERWRITE=True.

Notes:
- A "slice" is a dict of equality filters (e.g., {"phi": 0.2, "num_customers": 400}).
- Aggregation collapses any remaining dimensions via mean/median.
- Susceptibility expects column 'susceptibility_dL_dT' in results_susceptibility.csv.
"""

import os
import glob
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

# ==============================
# CONFIG — EDIT AS YOU LIKE
# ==============================

# Which run directories to scan (globs). Each run dir should contain results_summary.csv
_HERE = Path(__file__).resolve().parent
RUN_GLOBS = [
    str(_HERE / "out" / "run_*"),
    str(_HERE.parent / "out" / "run_*"),
]

# Axes and aggregation
XAXIS = "T"
YAXIS = "J"  # axis must vary across the sweep to avoid degenerate heatmaps
AGG = "mean"  # "mean" or "median"

# Which metrics from results_summary.csv to render
METRICS = [
    "late_local_share",
    "late_entropy",
    "late_hhi",
    # "final_local_share",
    # "final_entropy",
    # "final_hhi",
]

# Optional fixed vmin/vmax per metric (None => auto)
VMIN_VMAX: Dict[str, Tuple[float, float]] = {
    "late_local_share": (0.0, 1.0),
    "late_entropy": (None, None),
    "late_hhi": (None, None),
}

# Draw the per-row argmax(x) ridge (useful for locating the “critical” T for each gamma)
ARGMAX_LINE = True

# Save the pivot matrix as CSV next to the figure
SAVE_PIVOT_CSV = True

# If True, overwrite existing plots. If False, skip work when figure already exists.
OVERWRITE = False

# Produce susceptibility heatmaps if file exists in the run folder
PLOT_SUSCEPTIBILITY = True
SUSCEPTIBILITY_METRIC = "susceptibility_dL_dT"

# Define which K slices to facet when plotting mean-field runs
MEAN_FIELD_K_LEVELS = [3, 5, 7, 9, 12]

# Define slices (filters) you want heatmaps for. Leave empty to make a single "all" slice.
# Values may be scalars or lists (interpreted as membership).
SLICES: List[Dict[str, Any]] = [
    {"name": "classic", "filters": {"model_type": "classic"}},
    {"name": "mean_field_all", "filters": {"model_type": "mean_field"}},
    *(
        {"name": f"mean_field_K{k}", "filters": {"model_type": "mean_field", "K": k}}
        for k in MEAN_FIELD_K_LEVELS
    ),
    # {"name": "phi_0.0", "filters": {"phi": 0.0}},
    # {"name": "phi_0.2", "filters": {"phi": 0.2}},
    # {"name": "structure_A", "filters": {"num_customers": 400, "num_vendors": 60, "local_fraction": 0.6}},
]

# SLICES = [{
#   "name": "gamma1_phi0.2",
#   "filters": {"gamma": 1.0, "phi": 0.2, "num_customers": 400, "num_vendors": 60,
#               "rounds": 100, "rho": 1.3, "sigma": 0.30}
# }]

# ==============================
# END CONFIG
# ==============================


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _naturalize(vals: List[Any]) -> List[Any]:
    """Sort axis labels numerically when possible; else lexicographically as strings."""
    try:
        arr = np.array(vals, dtype=float)
        return list(arr[np.argsort(arr)])
    except Exception:
        return sorted(vals, key=lambda z: str(z))


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply equality or membership filters where values can be scalars or lists."""
    if not filters:
        return df
    out = df.copy()
    for k, v in filters.items():
        if k not in out.columns:
            raise KeyError(f"Filter key '{k}' not in dataframe columns")
        if isinstance(v, (list, tuple, set)):
            out = out[out[k].isin(list(v))]
        else:
            out = out[out[k] == v]
    return out


def aggregate_for_phase(
    df: pd.DataFrame,
    metric: str,
    xaxis: str,
    yaxis: str,
    agg: str = "mean",
) -> pd.DataFrame:
    """Pivot into y-by-x matrix of the chosen metric, aggregating across leftovers."""
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found in columns: {list(df.columns)}")
    if xaxis not in df.columns or yaxis not in df.columns:
        raise KeyError(f"Axes '{xaxis}', '{yaxis}' must be present in columns.")

    gb = df.groupby([yaxis, xaxis], dropna=False)[metric]
    if agg == "mean":
        g = gb.mean().reset_index()
    elif agg == "median":
        g = gb.median().reset_index()
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    pv = g.pivot(index=yaxis, columns=xaxis, values=metric)
    pv = pv.reindex(index=_naturalize(list(pv.index)))
    pv = pv.reindex(columns=_naturalize(list(pv.columns)))
    return pv


def draw_heatmap(
    pv: pd.DataFrame,
    title: str,
    outpath: str,
    vmin=None,
    vmax=None,
    argmax_line: bool = False,
):
    """Render a heatmap via imshow with axes scaled to numeric labels."""
    fig, ax = plt.subplots(figsize=(7, 5))
    # Prepare numeric axes
    xvals = list(pv.columns)
    yvals = list(pv.index)
    X = np.array(xvals, dtype=float)
    Y = np.array(yvals, dtype=float)
    Z = pv.to_numpy()

    im = ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("value")

    ax.set_xlabel(pv.columns.name if pv.columns.name else "x")
    ax.set_ylabel(pv.index.name if pv.index.name else "y")
    ax.set_title(title)

    if argmax_line:
        xs = []
        for i in range(Z.shape[0]):
            row = Z[i, :]
            if np.all(np.isnan(row)):
                xs.append(np.nan)
            else:
                j = np.nanargmax(row)
                xs.append(X[j])
        ax.plot(xs, Y, linewidth=2)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(outpath))
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def sanitize_name(s: str) -> str:
    return (
        str(s)
        .replace(" ", "")
        .replace(",", "")
        .replace("/", "-")
        .replace(":", "-")
        .replace("=", "-")
    )


def slice_name(slice_cfg: Dict[str, Any]) -> str:
    if not slice_cfg:
        return "all"
    name = slice_cfg.get("name")
    if name:
        return sanitize_name(name)
    # build from filters
    filt = slice_cfg.get("filters", {})
    if not filt:
        return "all"
    parts = []
    for k in sorted(filt.keys()):
        v = filt[k]
        if isinstance(v, (list, tuple, set)):
            vstr = "-".join(sanitize_name(x) for x in map(str, v))
        else:
            vstr = sanitize_name(v)
        parts.append(f"{k}-{vstr}")
    return "_".join(parts) if parts else "all"


def render_one_heatmap(
    df: pd.DataFrame,
    outdir: str,
    metric: str,
    xaxis: str,
    yaxis: str,
    agg: str,
    slice_cfg: Dict[str, Any],
    vmin_vmax: Tuple[float, float],
    argmax_line: bool,
    save_pivot_csv: bool,
    overwrite: bool,
) -> str:
    """Filter→pivot→plot for a single metric & slice. Returns path (or None if skipped)."""
    df_slice = apply_filters(df, slice_cfg.get("filters", {}))
    if df_slice.empty:
        return ""

    pv = aggregate_for_phase(df_slice, metric=metric, xaxis=xaxis, yaxis=yaxis, agg=agg)

    base = f"{metric}_{yaxis}-vs-{xaxis}_{slice_name(slice_cfg)}_{agg}"
    fig_path = os.path.join(outdir, f"{base}.png")
    if (not overwrite) and os.path.exists(fig_path):
        print(f"[skip] exists: {fig_path}")
        return fig_path

    vmin, vmax = vmin_vmax if vmin_vmax is not None else (None, None)
    draw_heatmap(
        pv,
        title=f"{metric} ({yaxis} vs {xaxis}) — {slice_name(slice_cfg)}",
        outpath=fig_path,
        vmin=vmin,
        vmax=vmax,
        argmax_line=argmax_line,
    )
    if save_pivot_csv:
        pv.to_csv(os.path.join(outdir, f"{base}.csv"))
    print(f"[ok] wrote: {fig_path}")
    return fig_path


def render_one_sus_heatmap(
    sus_df: pd.DataFrame,
    outdir: str,
    xaxis: str,
    yaxis: str,
    agg: str,
    slice_cfg: Dict[str, Any],
    overwrite: bool,
) -> str:
    """Susceptibility heatmap for the same slice."""
    if SUSCEPTIBILITY_METRIC not in sus_df.columns:
        return ""
    filters = slice_cfg.get("filters", {})
    usable_filters = {k: v for k, v in filters.items() if k in sus_df.columns}
    if filters and not usable_filters:
        return ""
    df_slice = apply_filters(sus_df, usable_filters)
    if df_slice.empty:
        return ""

    pv = aggregate_for_phase(
        df_slice,
        metric=SUSCEPTIBILITY_METRIC,
        xaxis=xaxis,
        yaxis=yaxis,
        agg=agg,
    )
    base = f"{SUSCEPTIBILITY_METRIC}_{yaxis}-vs-{xaxis}_{slice_name(slice_cfg)}_{agg}"
    fig_path = os.path.join(outdir, f"{base}.png")
    if (not overwrite) and os.path.exists(fig_path):
        print(f"[skip] exists: {fig_path}")
        return fig_path

    draw_heatmap(
        pv,
        title=f"{SUSCEPTIBILITY_METRIC} ({yaxis} vs {xaxis}) — {slice_name(slice_cfg)}",
        outpath=fig_path,
    )
    print(f"[ok] wrote: {fig_path}")
    return fig_path


def find_runs(run_globs: List[str]) -> List[str]:
    """Return list of run directories containing results_summary.csv."""
    found = []
    seen = set()
    for patt in run_globs:
        for path in glob.glob(patt):
            if not os.path.isdir(path):
                continue
            abs_path = os.path.abspath(path)
            if abs_path in seen:
                continue
            if os.path.exists(os.path.join(abs_path, "results_summary.csv")):
                found.append(abs_path)
                seen.add(abs_path)
    return sorted(found)


def main():
    run_dirs = find_runs(RUN_GLOBS)
    if not run_dirs:
        print("[warn] No runs found. Edit RUN_GLOBS in CONFIG.")
        return

    # If no slices specified, use a single catch-all slice
    slices = SLICES if len(SLICES) > 0 else [{"name": "all", "filters": {}}]

    for run_dir in run_dirs:
        summary_path = os.path.join(run_dir, "results_summary.csv")
        sus_path = os.path.join(run_dir, "results_susceptibility.csv")
        plots_dir = os.path.join(run_dir, "plots")
        _ensure_dir(plots_dir)

        # Load summary (and susceptibility if any)
        try:
            df = pd.read_csv(summary_path)
        except Exception as e:
            print(f"[skip] could not read {summary_path}: {e}")
            continue

        sus_df = None
        if PLOT_SUSCEPTIBILITY and os.path.exists(sus_path):
            try:
                sus_df = pd.read_csv(sus_path)
            except Exception as e:
                print(f"[warn] could not read susceptibility {sus_path}: {e}")

        # Generate per-slice plots
        for sl in slices:
            for metric in METRICS:
                vmin_vmax = VMIN_VMAX.get(metric, (None, None))
                render_one_heatmap(
                    df=df,
                    outdir=plots_dir,
                    metric=metric,
                    xaxis=XAXIS,
                    yaxis=YAXIS,
                    agg=AGG,
                    slice_cfg=sl,
                    vmin_vmax=vmin_vmax,
                    argmax_line=ARGMAX_LINE,
                    save_pivot_csv=SAVE_PIVOT_CSV,
                    overwrite=OVERWRITE,
                )

            if sus_df is not None:
                render_one_sus_heatmap(
                    sus_df=sus_df,
                    outdir=plots_dir,
                    xaxis=XAXIS,
                    yaxis=YAXIS,
                    agg=AGG,
                    slice_cfg=sl,
                    overwrite=OVERWRITE,
                )

    print("[done] Phase plotting sweep complete.")


if __name__ == "__main__":
    main()
