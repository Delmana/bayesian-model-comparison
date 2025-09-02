
"""
Bayesian Multifactor ANOVA (up to 3 factors, robust Student-t likelihood)
------------------------------------------------------------------------
Implements a Bayesian multifactor ANOVA following the approach in
Kruschke (2015), Chapter 20: main effects with sum-to-zero constraints
and optional interactions, robust Student-t likelihood, partial pooling via
hierarchical normal priors on factor effects and interactions.

This module mirrors the style of `BayesianOneWayANOVA.py` in your project:
  - Uses `AbstractBayesian` to compile/run a Stan model
  - Adds posterior predictive checks to `InferenceData`
  - Provides pairwise ROPE probabilities for factor-level differences
  - Computes cell and marginal means in generated quantities

Requirements (same as your one-way implementation):
  numpy, arviz, matplotlib, cmdstanpy via your AbstractBayesian.

Usage (example, two factors A,B with interactions):
------------------------------------------------------------------
    import numpy as np
    from BayesianMultifactorANOVA import BayesianMultifactorANOVA

    # y: shape (N,)
    # factors: dict[str, array_like] with equal length N
    y = np.array([...], float)
    factors = {
        "A": np.array([...]),
        "B": np.array([...]),
    }

    mf = BayesianMultifactorANOVA(
        y=y,
        factors=factors,
        include_interactions=True,   # AB (and AC/BC/ABC if 3 factors)
        rope_main={
            "A": (-1.0, 1.0),       # ROPE for pairwise differences of A's marginal means
            "B": (-1.0, 1.0),
        },
        rope_cell=None,              # e.g. (-1, 1) for cell-mean diffs
        seed=42,
    )
    mf.fit(chains=4, draws=1000, tune=1000)
    results = mf.analyse(posterior_predictive_check=True, plot=True, save=False)

Notes
-----
- Supports up to 3 factors. With 1 factor it reduces to a one-way setup.
- Factor effects are sum-to-zero constrained via mean-centering of the raw
  free parameters (Kruschke style coding).
- Interactions are centered to have zero marginal sums (e.g., for AB,
  row- & column-centering with compensation by grand mean).
- Generated quantities include:
    * y_rep
    * cell means (mu_cell) for all observed levels
    * marginal means for each factor (mu_A, mu_B, mu_C as applicable)
    * pairwise differences of marginal means per factor (diff_A, diff_B, diff_C)
    * total eta^2 (variance explained by the linear predictor)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import arviz as az
import matplotlib.pyplot as plt

from bayesian_test.AbstractBayesian import AbstractBayesian
from utils.plotting import plot_posterior_predictive_check, plot_posterior_pdf
from bayesian_test.utils import (
    print_result,
    posterior_predictive_check_metrics,
    calculate_statistics,
)


# ============ helpers ============

@dataclass
class _FactorMap:
    name: str
    index: np.ndarray          # 1..J per observation (int)
    labels: List[str]
    J: int


def _map_factor(name: str, x: np.ndarray) -> _FactorMap:
    g = np.asarray(x).reshape(-1)
    uniq, inv = np.unique(g, return_inverse=True)
    J = uniq.shape[0]
    idx = inv + 1  # 1..J
    labels = [str(u) for u in uniq.tolist()]
    return _FactorMap(name=name, index=idx.astype(int), labels=labels, J=int(J))


def _pairwise_names(labels: List[str]) -> List[Tuple[int, int, str]]:
    out = []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            out.append((i, j, f"{labels[i]} - {labels[j]}"))
    return out


# ============ model ============

class BayesianMultifactorANOVA(AbstractBayesian):
    def __init__(
        self,
        y: np.ndarray,
        factors: Dict[str, np.ndarray],
        include_interactions: bool = True,
        rope_main: Optional[Dict[str, Tuple[float, float]]] = None,
        rope_cell: Optional[Tuple[float, float]] = None,
        seed: int = 42,
    ) -> None:
        # Limit: up to 3 factors for now (A,B,C)
        if not (1 <= len(factors) <= 3):
            raise ValueError("This implementation supports 1 to 3 factors. "
                             "Provide 1, 2, or 3 factor arrays in the 'factors' dict.")

        super().__init__(stan_file="bayesian_multifactor_anova.stan", rope=None, seed=seed)

        y = np.asarray(y).reshape(-1).astype(float)
        if np.any(np.isnan(y)):
            raise ValueError("y contains NaN values")
        self.y = y
        self.N = y.shape[0]

        # Map & validate factors
        self.factor_names: List[str] = list(factors.keys())
        if len(set(self.factor_names)) != len(self.factor_names):
            raise ValueError("Duplicate factor names found.")

        mapped: List[_FactorMap] = []
        for nm in self.factor_names:
            fm = _map_factor(nm, np.asarray(factors[nm]).reshape(-1))
            if fm.index.shape[0] != self.N:
                raise ValueError(f"Factor '{nm}' length does not match y.")
            mapped.append(fm)

        self.factors_mapped = mapped
        self.k = len(mapped)

        self.include_interactions = bool(include_interactions)
        self.rope_main = rope_main or {}
        self.rope_cell = rope_cell

        # Assemble Stan data
        self._stan_data = self._transform_data()

    # ---------- data to Stan ----------
    def _transform_data(self) -> dict:
        y_mean = float(np.mean(self.y))
        y_sd = float(np.std(self.y, ddof=1) if self.N > 1 else max(1.0, np.std(self.y)))

        data = dict(
            N=self.N,
            y=self.y,
            y_mean=y_mean,
            y_sd=y_sd,
            K=int(self.k),               # number of factors (1..3)
            include_interactions=int(self.include_interactions),
        )

        # Fill per-factor slots (A,B,C)
        # If < 3 factors, set J=0 and provide dummy index vector of size N with 1s
        for i in range(3):
            keyJ = f"J_{i+1}"
            keyIdx = f"idx_{i+1}"
            if i < self.k:
                fm = self.factors_mapped[i]
                data[keyJ] = int(fm.J)
                data[keyIdx] = fm.index.astype(int)
            else:
                data[keyJ] = 0
                data[keyIdx] = np.ones(self.N, dtype=int)

        return data

    # ---------- PPC ----------
    def _posterior_predictive_check(
        self,
        directory_path: str,
        file_path: str,
        file_name: str = "posterior_predictive_check",
        font_size: int = 12,
        save: bool = True,
    ) -> None:
        print("INFO: Running posterior predictive check...")
        y_rep = self._fit.stan_variable("y_rep")  # (cd, N)
        n_cd, n_samples = y_rep.shape
        assert n_samples == self.N

        metrics = [posterior_predictive_check_metrics(self.y, y_rep[i], ranks=False) for i in range(n_cd)]
        means, std_devs = calculate_statistics(metrics)
        print("\nPosterior Predictive Check Metrics")
        print(f"Means: {means}\nStdDevs: {std_devs}\n")

        # reshape to (chains, draws, N)
        n_draws = int(n_cd / self.chains)
        y_rep = y_rep.reshape((self.chains, n_draws, self.N))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="the default dims 'chain' and 'draw' will be added automatically")
            if not hasattr(self.inf_data, "observed_data"):
                self.inf_data.add_groups(observed_data=dict(y=self.y))
            if not hasattr(self.inf_data, "posterior_predictive"):
                self.inf_data.add_groups(posterior_predictive=dict(y_rep=y_rep))
            else:
                self.inf_data.posterior_predictive = self.inf_data.posterior_predictive.assign(
                    y_rep=(("chain", "draw", "obs_dim"), y_rep)
                )

        plot_posterior_predictive_check(
            inf_data=self.inf_data,
            variables=["y"],
            n_draws=n_draws,
            show_plt=not save,
            font_size=font_size,
            seed=self.seed,
        )
        if save:
            self.save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name)
        else:
            plt.tight_layout()

    # ---------- analysis ----------
    def analyse(
        self,
        posterior_predictive_check: bool = True,
        plot: bool = True,
        save: bool = True,
        round_to: int = 4,
        directory_path: str = "results",
        file_path: str = "bayesian_multifactor_anova",
        file_name: Optional[str] = None,
        plt_imp_param: bool = False,
        **kwargs,
    ) -> dict:
        if posterior_predictive_check:
            file_name_ppc = f"{self._execution_time}" if file_name is None else file_name
            self._posterior_predictive_check(
                directory_path=directory_path,
                file_path=file_path,
                file_name=f"{file_name_ppc}_ppc",
                font_size=10,
                save=save,
            )

        summary = self._simple_analysis(round_to=round_to)
        results: Dict[str, Any] = dict(summary=summary.to_dict())

        # Extract generated quantities for pairwise ROPE per factor
        # We compute pairwise diffs of marginal means in Stan: diff_A/B/C
        # Arrays are stored only for factors that exist.
        # diff_* shapes:
        #  - For factor with J levels, Stan stores a packed upper-triangular vector of size J*(J-1)/2
        #  - We'll reconstruct per factor from shapes stored in 'meta' variables: n_pairs_A etc.
        chain_draw = self._fit.stan_variable("eta_sq")  # to infer cd, but we'll also use actual diff vars
        n_cd = chain_draw.shape[0]

        # Helper to read packed differences & compute ROPE probabilities
        def rope_for_factor(prefix: str, labels: List[str], rope_rng: Tuple[float, float]):
            try:
                diffs = self._fit.stan_variable(f"diff_{prefix}")  # (cd, n_pairs)
            except Exception:
                return None
            lo, hi = rope_rng
            n_pairs = diffs.shape[1]
            idx_names = _pairwise_names(labels)
            assert n_pairs == len(idx_names)

            left, inside, right, pairs = [], [], [], []
            for p in range(n_pairs):
                d = diffs[:, p]
                lp = float(np.mean(d < lo))
                ip = float(np.mean((d >= lo) & (d <= hi)))
                rp = float(np.mean(d > hi))
                i, j, name = idx_names[p]
                pairs.append(name)
                left.append(round(lp, round_to))
                inside.append(round(ip, round_to))
                right.append(round(rp, round_to))
            return dict(
                factor=prefix,
                pairs=pairs,
                left_prob=left,
                rope_prob=inside,
                right_prob=right,
                rope=[float(lo), float(hi)],
            )

        # Factor labels for nice names
        factor_labels = [fm.labels for fm in self.factors_mapped]
        factor_keys = ["A", "B", "C"][:self.k]

        # Pairwise ROPE per factor (if provided)
        pairwise_all = []
        for k_idx, k_name in enumerate(factor_keys):
            nm = self.factor_names[k_idx]
            if nm in self.rope_main:
                pr = rope_for_factor(k_name, factor_labels[k_idx], self.rope_main[nm])
                if pr is not None:
                    pr["name"] = nm
                    pairwise_all.append(pr)
        if pairwise_all:
            results["pairwise"] = pairwise_all

        # Effect-size-ish metrics: posterior of sigma_a/sigma_b/... (hierarchical SDs)
        try:
            sigmas = {}
            for k_idx, k_name in enumerate(factor_keys):
                s = self._fit.stan_variable(f"sigma_{k_name.lower()}")
                sigmas[self.factor_names[k_idx]] = dict(
                    mean=float(np.mean(s)),
                    hdi95=list(az.hdi(s, hdi_prob=0.95).tolist())
                )
            results["sigma_effects"] = sigmas
        except Exception:
            pass

        # Total eta^2
        try:
            eta_sq_draws = self._fit.stan_variable("eta_sq")
            etamean = float(np.mean(eta_sq_draws))
            hdi_low, hdi_high = az.hdi(eta_sq_draws, hdi_prob=0.95).tolist()
            results["eta_sq"] = dict(
                mean=round(etamean, round_to),
                hdi95=[round(hdi_low, round_to), round(hdi_high, round_to)],
            )
        except Exception:
            pass

        # ----- plotting: show posterior of effect SDs and eta^2 -----
        if plot:
            panels = []
            # eta^2 if present
            try:
                eta_sq_draws = self._fit.stan_variable("eta_sq")
                panels.append((
                    r"Posterior Distribution of $\eta^2$",
                    eta_sq_draws,
                    None, None,
                    round(float(np.mean(eta_sq_draws)), round_to),
                    False, False, True
                ))
            except Exception:
                pass

            # hierarchical SDs of effects (sigma_a/b/c)
            for k_idx, k_name in enumerate(factor_keys):
                try:
                    s = self._fit.stan_variable(f"sigma_{k_name.lower()}")
                    panels.append((
                        rf"Posterior of $\sigma_{{{k_name}}}$ ({self.factor_names[k_idx]})",
                        s, None, None,
                        round(float(np.mean(s)), round_to),
                        False, False, (k_idx % 2 == 0)
                    ))
                except Exception:
                    pass

            # residual scale
            try:
                sigma_y = self._fit.stan_variable("sigma_y")
                panels.append((
                    r"Posterior of $\sigma_y$",
                    sigma_y, None, None,
                    round(float(np.mean(sigma_y)), round_to),
                    False, True, True
                ))
            except Exception:
                pass

            # nu
            try:
                nu = self._fit.stan_variable("nu")
                panels.append((
                    r"Posterior of $\nu$",
                    nu, None, None,
                    round(float(np.mean(nu)), round_to),
                    True, True, False
                ))
            except Exception:
                pass

            # Layout 2 cols
            ncols = 2
            n = len(panels)
            if n > 0:
                nrows = (n + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(6, 3 * nrows)))
                axes = np.atleast_2d(axes)
                for idx, (title, data, rope_rng, within_rope, mean_val, show_legend, xlab, ylab) in enumerate(panels):
                    r, c = divmod(idx, ncols)
                    ax = axes[r, c]
                    plot_posterior_pdf(
                        data=data,
                        rope=rope_rng,
                        within_rope=within_rope,
                        mean=mean_val,
                        round_to=round_to,
                        title=title,
                        ax=ax,
                        plt_legend=show_legend,
                        plt_x_label=xlab,
                        plt_y_label=ylab,
                        show_plt=False,
                    )
                # remove empty axes
                for idx in range(n, nrows * ncols):
                    r, c = divmod(idx, ncols)
                    fig.delaxes(axes[r, c])
                if save:
                    self.save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name)
                    plt.show()
                else:
                    plt.tight_layout()
                    plt.show()

        if save:
            self.save_results(results, directory_path=directory_path, file_path=file_path, file_name=file_name)

        print_result(results, round_to=round_to)
        return results
