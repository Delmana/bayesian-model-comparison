"""
Example: Run a Bayesian Independent t-Test (BEST) on your CSV.

Accepted input formats:
  1) Long:  columns = group,value            (e.g., Group1, Group2 in "group")
  2) Wide:  exactly two numeric columns      (e.g., Gruppe1,Gruppe2) -> auto-convert to long

Usage:
  python examples/simple_examples/bayesian_independent_tTest.py \
    --csv examples/simple_examples/data/t_test_example.csv \
    --group1 Control --group2 Meditation \
    --rope -0.2 0.2 --iter 4000 --warmup 1000 --chains 4 \
    --ppc --plot

Results are saved under: examples/results/bayesian_independent_t_test/<timestamp>/
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime

from utils.helper import load_results
from utils.plotting import plot_densities
from bayesian_test.BayesianIndependentTTest import BayesianIndependentTTest


def _wide_to_long(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """
    Convert a wide table with two columns (one per group) into a long table with columns (group, value).
    - Keeps only non-null rows.
    - Converts decimal commas to dots when needed.
    - Forces numeric conversion and drops non-convertible rows.
    """
    long = df.melt(value_vars=[col1, col2], var_name="group", value_name="value").dropna()
    # Convert decimal comma to dot if values are strings (e.g., "5,12" -> "5.12")
    if long["value"].dtype == object:
        long["value"] = long["value"].str.replace(",", ".", regex=False)
    # Ensure numeric dtype; coerce errors to NaN, then drop them
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    return long[["group", "value"]]


def run_best_from_csv(
    csv_path: str,
    group1: str,
    group2: str,
    rope: tuple[float, float] | None,
    iter_sampling: int,
    iter_warmup: int,
    chains: int,
    ppc: bool,
    plot: bool,
    seed: int,
) -> str:
    """
    End-to-end routine:
      - Read CSV (long or wide).
      - Validate/prepare long format (group,value).
      - Extract y1/y2 arrays for the two groups.
      - Fit the BEST model.
      - Analyse and save figures/results under a timestamped folder.
      - Print a compact summary and return the output directory path.
    """
    # Robust CSV read: detects separators (comma/semicolon/tab) automatically
    df = pd.read_csv(csv_path, sep=None, engine="python")

    # Ensure long format: either (group,value) already, or convert from exactly two wide columns
    if {"group", "value"}.issubset(df.columns):
        long_df = df[["group", "value"]].copy()
    elif len(df.columns) == 2:
        # Wide format detected. Convert two columns (e.g., Gruppe1, Gruppe2) into long format.
        c1, c2 = list(df.columns)
        print(f"[INFO] Detected wide format: {c1!r}, {c2!r} -> converting to (group,value).")
        long_df = _wide_to_long(df, c1, c2)
        # If user-specified group names don't match the two wide column headers, fail fast.
        if group1 not in [c1, c2] or group2 not in [c1, c2]:
            raise ValueError(
                f"Groups must match the wide CSV column names. Found: {c1!r}, {c2!r}."
            )
    else:
        # Neither long nor exactly two wide columns -> unsupported layout
        raise ValueError("CSV must either contain (group,value) or exactly two columns (wide).")

    # Sanity check: group names must be present
    if group1 not in long_df["group"].unique() or group2 not in long_df["group"].unique():
        raise ValueError(
            f"Groups '{group1}'/'{group2}' not found. Available: {sorted(long_df['group'].unique())}"
        )

    # Extract numeric arrays for both groups
    y1 = long_df.loc[long_df["group"] == group1, "value"].to_numpy(dtype=float)
    y2 = long_df.loc[long_df["group"] == group2, "value"].to_numpy(dtype=float)

    # Quick descriptive stats printed to console
    print(f"[INFO] y1={group1}: n={len(y1)}, mean={np.mean(y1):.3f}, sd={np.std(y1, ddof=1):.3f}")
    print(f"[INFO] y2={group2}: n={len(y2)}, mean={np.mean(y2):.3f}, sd={np.std(y2, ddof=1):.3f}")

    # Optional: visualize raw distributions of both groups before modeling
    try:
        data_for_plot = np.stack((y1, y2), axis=1)
        plot_densities(
            data_for_plot,
            algorithm_labels=[group1, group2],
            show_plt=plot,  # opens a window only if --plot was set
            alpha=1.0,
        )
    except Exception as e:
        # Keep the pipeline robust; plotting is nice to have, not critical
        print(f"[WARN] Could not plot densities: {e}")

    # --- Model ---
    # Initialize BEST with the two samples and optional ROPE; set a seed for reproducibility
    bit = BayesianIndependentTTest(y1=y1, y2=y2, rope=rope, seed=seed)

    # Run MCMC sampling with the provided settings (iter/warmup/chains)
    bit.fit(
        iter_sampling=iter_sampling,
        iter_warmup=iter_warmup,
        chains=chains
    )

    # Prepare a timestamped output directory to store plots and serialized results
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    out_dir = os.path.join("examples", "results", "bayesian_independent_t_test", timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # Analyse posterior, create figures, and save to disk
    # file_name will prefix saved files; plot toggles interactive windows; save=True writes files
    file_name = os.path.splitext(os.path.basename(csv_path))[0]
    plot_params = dict(
        plt_imp_param=True,     # include the main parameters in the posterior plots
        hpdi_prob=0.95,         # HPDI coverage
        plot_type="kde",        # density plots (alternative: 'hist')
        plt_rope=True,          # draw the ROPE boundaries
        plt_rope_text=True,     # annotate ROPE bounds
        plt_within_rope=True,   # show share of posterior within ROPE
        plt_mean=True,          # mark posterior mean
        plt_mean_text=True,     # annotate mean
        plt_hpdi=True,          # draw HPDI bar
        plt_hpdi_text=True,     # annotate HPDI endpoints
        plt_samples=True,       # show individual posterior sample points
        plt_title=True,         # add plot titles
        alpha=0.6,              # transparency for layers
        font_size=10,           # base font size
    )

    results = bit.analyse(
        posterior_predictive_check=ppc,  # also run PPC and save PPC plots
        file_name=file_name,
        plot=plot,                       # show plots in a window if requested
        save=True,                       # always save outputs to disk
        round_to=3,                      # round reported numbers
        directory_path=out_dir,          # target folder for results
        **plot_params,
    )

    # Helper to print a compact textual summary for key quantities
    def pr(block: str, label: str):
        b = results.get(block, {})
        probs = (b.get("posterior_probabilities") or {})
        add = (b.get("additional") or {})
        mean = add.get("posterior_mean")
        sd = add.get("posterior_sd")
        lp, rp = probs.get("left_prob"), probs.get("right_prob")
        ropep = probs.get("rope_prob")
        print(f"\n[{label}] mean={mean:.3f} sd={sd:.3f}")
        if ropep is None:
            # No ROPE: report P(<0) and P(>0)
            if lp is not None and rp is not None:
                print(f"  P(<0)={lp:.3f} | P(>0)={rp:.3f}")
        else:
            # With ROPE: report left / within ROPE / right probabilities
            print(f"  P(left)={lp:.3f} | P(ROPE)={ropep:.3f} | P(right)={rp:.3f}")

    # Print concise key results to console
    print("\n====================  SUMMARY  ====================")
    pr("difference_mean", f"Δμ = {group2} − {group1}")       # mean difference μ2 − μ1
    pr("effect_size", "δ (standardized effect size)")        # Cohen-like delta from posteriors
    pr("difference_sigma", f"Δσ = σ({group2}) − σ({group1})")# difference in group std devs
    print("===================================================")

    print(f"\n[INFO] Saved results to: {out_dir}")
    return out_dir


def main():
    """
    CLI entry point:
      - Parse command-line arguments.
      - Set random seed.
      - Run BEST on the provided CSV.
      - Optionally load saved results and print posterior probabilities.
    """
    parser = argparse.ArgumentParser(description="Run BEST on your CSV (long or wide).")
    parser.add_argument("--csv", required=True, help="Path to CSV (long: group,value | wide: 2 columns).")
    parser.add_argument("--group1", required=True, help="Group name for y1 (exactly as in CSV).")
    parser.add_argument("--group2", required=True, help="Group name for y2 (exactly as in CSV).")
    parser.add_argument("--rope", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None,
                        help="ROPE interval, e.g., --rope -0.2 0.2")
    parser.add_argument("--iter", type=int, default=4000, help="iter_sampling per chain") #how many posterior samples are drawn per chain.
    parser.add_argument("--warmup", type=int, default=1000, help="iter_warmup per chain") #how many are discarded at the beginning to stabilize.
    parser.add_argument("--chains", type=int, default=4, help="number of chains")#how many independent chains run in parallel (for convergence control and more samples).Chain is a single “track” (or simulation) of the Markov Chain Monte Carlo (MCMC) method.
    parser.add_argument("--ppc", action="store_true", help="Run posterior predictive checks")
    parser.add_argument("--plot", action="store_true", help="Show interactive plots (files are saved regardless)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Ensure deterministic NumPy behavior for reproducibility of synthetic steps (if any)
    np.random.seed(args.seed)

    # Execute the end-to-end pipeline and get the output directory
    save_dir = run_best_from_csv(
        csv_path=args.csv,
        group1=args.group1,
        group2=args.group2,
        rope=tuple(args.rope) if args.rope else None,
        iter_sampling=args.iter,
        iter_warmup=args.warmup,
        chains=args.chains,
        ppc=args.ppc,
        plot=args.plot,
        seed=args.seed,
    )

    # Optional: Load the saved results back (if compatible with your helper) and print posterior probabilities
    try:
        file_stem = os.path.splitext(os.path.basename(args.csv))[0]
        res = load_results(file_path=save_dir, file_name=file_stem)
        for key in ['difference_mean', 'difference_sigma', 'effect_size', 'mu1', 'mu2', 'sigma1', 'sigma2', 'nu']:
            print(f"{key}: {res[key]['posterior_probabilities']}")
    except Exception as e:
        # Not fatal; just informs that auto-loading/printing did not succeed
        print(f"[WARN] Could not reload saved results: {e}")


if __name__ == "__main__":
    # Standard Python entry guard: run main() only if the script is executed directly
    main()
