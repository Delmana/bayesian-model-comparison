"""
Bayesian One-way ANOVA (hierarchical, robust Student-t likelihood)
------------------------------------------------------------------
Implements a Bayesian one-way ANOVA following the hierarchical
formulation in Kruschke (2015), Chapter 19, with a robust Student-t
likelihood and partial pooling of group means via a common
hyperprior. Uses CmdStan via cmdstanpy through AbstractBayesian.

Model (Stan):
    y_i ~ StudentT(nu, mu[group_i], sigma_y)
    mu_j ~ Normal(mu0, sigma_mu)
    mu0 ~ Normal(mean(y), 10 * sd(y))
    sigma_mu, sigma_y ~ Student-t(3, 0, sd(y))  # half-t via <lower=0>
    nu - 1 ~ Exponential(1/29)

Generated quantities include replicated data (y_rep), pairwise
differences of group means (diff_mu[K, K]), and eta^2
(variance explained by groups).
"""

from __future__ import annotations

import warnings
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

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

def _ensure_integer_groups(groups: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Map arbitrary labels to consecutive integers 1..K (Stan requirement)."""
    g = np.asarray(groups).reshape(-1)
    uniq, inv = np.unique(g, return_inverse=True)
    g_int = inv + 1  # 1..K
    labels = [str(u) for u in uniq.tolist()]
    return g_int.astype(int), labels

def _pairwise_indices(k: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


class BayesianOneWayANOVA(AbstractBayesian):
    def __init__(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        rope: Optional[tuple[float, float]] = None,
        seed: int = 42,
    ) -> None:
        super(BayesianOneWayANOVA, self).__init__(
            stan_file='bayesian_one-way_anova.stan', rope=rope, seed=seed
        )

        # Validate inputs
        y = np.asarray(y).reshape(-1)
        groups = np.asarray(groups).reshape(-1)
        assert y.ndim == 1, "y must be a 1D array"
        assert groups.ndim == 1, "groups must be a 1D array"
        assert y.shape[0] == groups.shape[0], "y and groups must have same length"
        assert not np.any(np.isnan(y)), "y contains NaN values"

        g_int, labels = _ensure_integer_groups(groups)
        self.y = y.astype(float)
        self.g = g_int
        self.labels = labels
        self.K = len(labels)
        self.N = self.y.shape[0]

        if self.K < 2:
            raise ValueError("At least two groups are required for one-way ANOVA.")

    # ---------- Stan data ----------
    def _transform_data(self) -> dict:
        return dict(
            N=self.N,
            K=self.K,
            g=self.g,
            y=self.y,
            y_mean=float(np.mean(self.y)),
            y_sd=float(np.std(self.y, ddof=1) if self.N > 1 else max(1.0, np.std(self.y))),
        )

    # ---------- PPC ----------
    def _posterior_predictive_check(
        self,
        directory_path: str,
        file_path: str,
        file_name: str = 'posterior_predictive_check',
        font_size: int = 12,
        save: bool = True,
    ) -> None:
        print(f'{datetime.now().time().strftime("%H:%M:%S")} - INFO: Running posterior predictive check.')
        # posterior predictive draws
        y_rep = self._fit.stan_variable('y_rep')  # shape (cd, N)
        n_cd, n_samples = y_rep.shape
        assert n_samples == self.N

        # metrics
        metrics = [posterior_predictive_check_metrics(self.y, y_rep[i], ranks=False) for i in range(n_cd)]
        means, std_devs = calculate_statistics(metrics)
        print('\nPosterior Predictive Check Metrics')
        print(f'y:\nMeans: {means}\nStdDevs: {std_devs}\n')

        # reshape to (chains, draws, N)
        n_draws = int(n_cd / self.chains)
        y_rep = y_rep.reshape((self.chains, n_draws, self.N))

        # add to InferenceData (only posterior_predictive & observed_data here)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="the default dims 'chain' and 'draw' will be added automatically")
            # add observed_data if not present
            if not hasattr(self.inf_data, "observed_data"):
                self.inf_data.add_groups(observed_data=dict(y=self.y))
            # add or update posterior_predictive
            if not hasattr(self.inf_data, "posterior_predictive"):
                self.inf_data.add_groups(posterior_predictive=dict(y_rep=y_rep))
            else:
                # assign with explicit dims
                self.inf_data.posterior_predictive = self.inf_data.posterior_predictive.assign(
                    y_rep=(("chain", "draw", "obs_dim"), y_rep)
                )

        # plot
        plot_posterior_predictive_check(
            inf_data=self.inf_data,
            variables=['y'],
            n_draws=n_draws,
            show_plt=not save,
            font_size=font_size,
            seed=self.seed,
        )
        if save:
            self.save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name)
        else:
            plt.tight_layout()

    # ---------- Posterior ----------
    @staticmethod
    def _plot_1d_posterior(
        samples: np.ndarray,
        title: str,
        rope: Optional[tuple[float, float]] = None,
        show_plt: bool = False,
    ) -> None:
        fig = plt.figure(figsize=(6.2, 4.0))
        ax = fig.gca()
        az.plot_posterior(samples, point_estimate="mean", hdi_prob=0.95, ax=ax)
        ax.set_title(title)
        if rope is not None:
            lo, hi = rope
            ax.axvline(lo, linestyle="--", alpha=0.6)
            ax.axvline(hi, linestyle="--", alpha=0.6)
            inside = np.mean((samples >= lo) & (samples <= hi)) * 100.0
            ax.text(0.99, 0.99,
                    f"ROPE [{lo:.2f}, {hi:.2f}] • {inside:.2f}% in ROPE",
                    ha="right", va="top", transform=ax.transAxes, fontsize=9)
        plt.tight_layout()
        if not show_plt:
            plt.close(fig)


    # ---------- Analysis ----------
    def analyse(
        self,
        posterior_predictive_check: bool = True,
        plot: bool = True,
        save: bool = True,
        round_to: int = 4,
        directory_path: str = 'results',
        file_path: str = 'bayesian_one_way_anova',
        file_name: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        ANOVA-Auswertung: Summary, η², optionale Posterior-Plots für μ[j] und Δμ[i,j] inkl. ROPE-Anteile.
        Rückgabe analog t-Test: dict mit 'summary', 'eta_sq' und ggf. 'pairwise'.
        """
        # --- PPC wie im t-Test ---
        if posterior_predictive_check:
            file_name_ppc = f'{self._execution_time}' if file_name is None else file_name
            self._posterior_predictive_check(
                directory_path=directory_path,
                file_path=file_path,
                file_name=f'{file_name_ppc}_ppc',
                font_size=10,
                save=save,
            )

        # --- ArviZ-Summary (wie im t-Test) ---
        summary = self._simple_analysis(round_to=round_to)

        # --- Posterior-Samples aus dem Fit ziehen ---
        eta_sq_draws = self._fit.stan_variable('eta_sq')      # (cd,)
        mu_draws     = self._fit.stan_variable('mu')          # (cd, K)
        diff_mu      = self._fit.stan_variable('diff_mu')     # (cd, K, K)

        # η² auch in InferenceData, damit utils.plotting funktioniert (wie beim t-Test)
        chains  = self.chains
        n_cd    = eta_sq_draws.shape[0]
        n_draws = int(n_cd / chains)
        eta_sq_posterior = eta_sq_draws.reshape((chains, n_draws))
    

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="the default dims 'chain' and 'draw' will be added automatically")
            self.inf_data.posterior = self.inf_data.posterior.assign(
                eta_sq=(("chain", "draw"), eta_sq_posterior)
            )

        # --- Ergebnisdict aufbauen (t-Test macht ähnliches) ---
        results = dict(summary=summary.to_dict())

        # η²-Kennzahlen (wie difference_* im t-Test)
        etamean = float(np.mean(eta_sq_draws))
        hdi_low, hdi_high = az.hdi(eta_sq_draws, hdi_prob=0.95).tolist()
        results['eta_sq'] = dict(
            mean=round(etamean, round_to),
            hdi95=[round(hdi_low, round_to), round(hdi_high, round_to)],
        )

        # --- Posterior-Plots für Gruppenmittel μ[j] (analog mu1/mu2 im t-Test) ---
        if plot:
            for j, lbl in enumerate(self.labels):
                samples = mu_draws[:, j]
                self._plot_1d_posterior(
                    samples=samples,
                    title=f"Posterior Distribution of μ[{lbl}]",
                    rope=None,
                    show_plt=not save,
                )
                if save:
                    self.save_plot(
                        directory_path=directory_path,
                        file_path=file_path,
                        file_name=(file_name or 'anova') + f"_mu_{lbl}",
                    )

        # --- ROPE & paarweise Unterschiede Δμ --- (entspricht difference_mean im t-Test, aber für alle Paare)
        if self.rope is not None and self.K >= 2:
            lo, hi = self.rope
            pair_names, left, inside, right = [], [], [], []
            for i in range(self.K):
                for j in range(i + 1, self.K):
                    name = f"{self.labels[i]} - {self.labels[j]}"
                    d = diff_mu[:, i, j]
                    lp = float(np.mean(d < lo))
                    ip = float(np.mean((d >= lo) & (d <= hi)))
                    rp = float(np.mean(d > hi))
                    pair_names.append(name)
                    left.append(round(lp, 4))
                    inside.append(round(ip, 4))
                    right.append(round(rp, 4))

                    if plot:
                        self._plot_1d_posterior(
                            samples=d,
                            title=f"Posterior Distribution of Δμ = μ[{self.labels[i]}] − μ[{self.labels[j]}]",
                            rope=self.rope,
                            show_plt=not save,
                        )
                        if save:
                            self.save_plot(
                                directory_path=directory_path,
                                file_path=file_path,
                                file_name=(file_name or 'anova') + f"_diff_{self.labels[i]}_{self.labels[j]}",
                            )

            results['pairwise'] = dict(
                pairs=pair_names,
                left_prob=left,
                rope_prob=inside,
                right_prob=right,
                rope=[float(lo), float(hi)],
            )

        # --- η² als Posterior-PDF mit deiner utils-Funktion (wie im t-Test) ---
        if plot:
            try:
                plot_posterior_pdf(
                    inf_data=self.inf_data,
                    variables=['eta_sq'],
                    rope=None,
                    show_plt=not save,
                    font_size=10,
                    seed=self.seed,
                )
                if save:
                    self.save_plot(
                        directory_path=directory_path,
                        file_path=file_path,
                        file_name=(file_name or 'eta_sq'),
                    )
            except Exception as e:
                warnings.warn(f"Plotting eta_sq failed: {e}")

        # --- Speichern der Ergebnisse wie im t-Test ---
        if save:
            self.save_results(results, directory_path=directory_path, file_path=file_path, file_name=file_name)

        # Ausgabe wie im t-Test
        print_result(results, round_to=round_to)
        return results
    
   