#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze VEQ (wide format, 4 conditions) with a Bayesian Multifactor Repeated-Measures ANOVA
and (optionally) a frequentist RM-ANOVA for comparison.

- Input file (adjust if needed): VEQ_Total_Wide_Format.csv
- Expected columns: VP, VEQ_C1, VEQ_C2, VEQ_C3, VEQ_C4
- Mapping below encodes the 2×2 design for MovementType × Occlusion (edit if your coding differs).

This script mirrors the structure of your example run script and uses your
BayesianMultifactorANOVA class. Results and plots are saved like in the example.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# ---- robust import for your class (works whether it's in a package or next to this file) ----
try:
    from bayesian_test.BayesianMultifactorANOVA import BayesianMultifactorANOVA  # type: ignore
except Exception:
    # Allow import when BayesianMultifactorANOVA.py is in the same folder as this script
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from BayesianMultifactorANOVA import BayesianMultifactorANOVA  # type: ignore

# Optional plotting helper from your project (if present)
try:
    from utils.plotting import plot_densities  # type: ignore
    HAVE_PLOTTING = True
except Exception:
    HAVE_PLOTTING = False


def load_wide(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"VP", "VEQ_C1", "VEQ_C2", "VEQ_C3", "VEQ_C4"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df


def wide_to_long(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Reshape wide VEQ to long and add factor columns based on mapping."""
    cond_cols = list(mapping.keys())
    long = df.melt(id_vars=["VP"], value_vars=cond_cols, var_name="Condition", value_name="VEQ")
    long["MovementType"] = long["Condition"].map(lambda c: mapping[c]["MovementType"])
    long["Occlusion"] = long["Condition"].map(lambda c: mapping[c]["Occlusion"])
    # enforce categorical ordering as they appear in mapping
    mt_levels = [mapping[c]["MovementType"] for c in cond_cols]
    oc_levels = [mapping[c]["Occlusion"] for c in cond_cols]
    # preserve unique order
    mt_order = list(dict.fromkeys(mt_levels))
    oc_order = list(dict.fromkeys(oc_levels))
    long["MovementType"] = pd.Categorical(long["MovementType"], categories=mt_order, ordered=True)
    long["Occlusion"] = pd.Categorical(long["Occlusion"], categories=oc_order, ordered=True)
    return long


def build_model_inputs(long: pd.DataFrame):
    y = long["VEQ"].to_numpy(dtype=float)
    factors = {
        "Movement Type": long["MovementType"].astype(str).to_numpy(),
        "Occlusion": long["Occlusion"].astype(str).to_numpy(),
    }
    return y, factors


def run_bayesian(long: pd.DataFrame, args) -> dict:
    y, factors = build_model_inputs(long)

    # ROPEs: adjust to your domain. VEQ is typically ~1–7 Likert; a Δ of ±0.2–0.3 is often considered "negligible".
    rope = tuple(args.rope_main) if args.rope_main else (-0.2, 0.2)
    rope_cell = tuple(args.rope_cell) if args.rope_cell else (-0.2, 0.2)

    model = BayesianMultifactorANOVA(
        y=y,
        factors=factors,
        include_interactions=True,
        rope_main={"Movement Type": rope, "Occlusion": rope},
        rope_cell=rope_cell,
        seed=args.seed,
    )

    # Fit
    model.fit(chains=args.chains, iter_sampling=args.iter_sampling, iter_warmup=args.iter_warmup)

    # Analyse and save figures+json results
    results = model.analyse(
        posterior_predictive_check=True,
        plot=True,
        save=True,
        round_to=3,
        directory_path=args.out_dir,
        file_path="veq_multifactor_anova",
        file_name=args.out_name,
    )
    # Also dump as a standalone JSON in the timestamped folder
    out_dir = os.path.join(args.out_dir, "veq_multifactor_anova", model._execution_time)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.out_name}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {out_dir}")
    return results


def run_frequentist(long: pd.DataFrame):
    """Optional: frequentist RM-ANOVA with statsmodels, if available."""
    try:
        import statsmodels.api as sm  # noqa: F401
        from statsmodels.stats.anova import AnovaRM
    except Exception as e:
        print("Frequentist RM-ANOVA skipped (statsmodels not available):", e)
        return None

    aov = AnovaRM(data=long, depvar="VEQ", subject="VP",
                  within=["MovementType", "Occlusion"]).fit()
    print("\n=== Frequentist RM-ANOVA (statsmodels.AnovaRM) ===")
    print(aov)
    return aov


def main():
    parser = argparse.ArgumentParser(description="Bayesian Multifactor ANOVA on VEQ wide-format data")
    parser.add_argument("--csv", default="VEQ_Total_Wide_Format.csv", help="Path to VEQ wide CSV")
    parser.add_argument("--out-dir", default="results", help="Base directory for output")
    parser.add_argument("--out-name", default="veq_multifactor", help="Base name for saved results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--iter-sampling", type=int, default=3000)
    parser.add_argument("--iter-warmup", type=int, default=1000)
    parser.add_argument("--rope-main", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--rope-cell", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"))
    args = parser.parse_args()

    # 2×2 condition mapping — edit if your coding differs
    mapping = {
        "VEQ_C1": {"MovementType": "M1", "Occlusion": "NoOcc"},
        "VEQ_C2": {"MovementType": "M1", "Occlusion": "Occ"},
        "VEQ_C3": {"MovementType": "M2", "Occlusion": "NoOcc"},
        "VEQ_C4": {"MovementType": "M2", "Occlusion": "Occ"},
    }

    # Load data
    df = load_wide(args.csv)

    # Reshape & add factors
    long = wide_to_long(df, mapping)
    print("Long-format data preview:")
    print(long.head())

    # Optional density plots per cell if your plotting helper exists
    if HAVE_PLOTTING:
        # Create a 2x2 cell label in row-major order
        long["Cell"] = long["MovementType"].astype(str) + "-" + long["Occlusion"].astype(str)
        # Pivot to a matrix (rows = subjects, cols = cells) for plot_densities
        pivot = long.pivot(index="VP", columns="Cell", values="VEQ").sort_index(axis=1)
        try:
            plot_densities(pivot.to_numpy(), algorithm_labels=list(pivot.columns), show_plt=True, alpha=0.8)
        except Exception as e:
            print("Density plot skipped:", e)

    # Bayesian analysis
    results = run_bayesian(long, args)

    # Frequentist comparison (optional)
    run_frequentist(long)

    # Print a compact summary
    print("\n--- Compact summary ---")
    if 'eta_sq' in results:
        print("Eta^2 (posterior mean, 95% HDI):", results['eta_sq'])
    if 'sigma_effects' in results:
        print("Effect SDs:", json.dumps(results['sigma_effects'], indent=2))
    if 'pairwise' in results:
        print("\nPairwise ROPE probabilities per factor:")
        for entry in results['pairwise']:
            print(f"Factor {entry.get('name', entry['factor'])} (ROPE {entry['rope']}):")
            for p, l, i, r in zip(entry['pairs'], entry['left_prob'], entry['rope_prob'], entry['right_prob']):
                print(f"  {p}: P(<)={l}, P(in)={i}, P(>)={r}")


if __name__ == "__main__":
    main()
