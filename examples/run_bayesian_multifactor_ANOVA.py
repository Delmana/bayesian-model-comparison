
import numpy as np
from utils.helper import load_results, save_results
from utils.plotting import plot_densities
from bayesian_test.BayesianMultifactorANOVA import BayesianMultifactorANOVA


def example_multifactor_anova(seed: int) -> dict:
    """
    Example of Bayesian Multifactor ANOVA (2 factors with interaction) on synthetic data.

    :param seed: Random seed for reproducibility.
    :return: Dict with results returned by model.analyse()
    """
    n_per_cell = 40  # observations per cell

    # --- Synthetic design: Factor A (2 levels), Factor B (3 levels) => 6 cells ---
    rng = np.random.default_rng(seed)
    levels_A = np.array(['A0', 'A1'])
    levels_B = np.array(['B0', 'B1', 'B2'])

    # Define cell means from main effects + mild AB interaction
    mu_A = {'A0': 0.0, 'A1': 0.6}
    mu_B = {'B0': -0.4, 'B1': 0.0, 'B2': 0.5}
    mu_AB = {
        ('A0','B0'): -0.1, ('A0','B1'): 0.0,  ('A0','B2'): 0.1,
        ('A1','B0'):  0.1, ('A1','B1'): 0.0,  ('A1','B2'): -0.1
    }
    sigma = 0.5  # residual sd

    Ys, As, Bs = [], [], []
    for a in levels_A:
        for b in levels_B:
            mu = mu_A[a] + mu_B[b] + mu_AB[(a, b)]
            y_cell = rng.normal(loc=mu, scale=sigma, size=n_per_cell)
            Ys.append(y_cell)
            As.append(np.repeat(a, n_per_cell))
            Bs.append(np.repeat(b, n_per_cell))

    # Flatten to vectors
    y = np.concatenate(Ys)
    A = np.concatenate(As)
    B = np.concatenate(Bs)

    # Plot densities per cell (6 columns), analogous zum One-Way-Dichteplot
    data = np.stack(Ys, axis=1)  # shape (n_per_cell, 6)
    labels = [f"{a}-{b}" for a in levels_A for b in levels_B]
    plot_densities(data, algorithm_labels=labels, show_plt=True, alpha=0.8)

    # Build factor dict expected by the model
    factors = {"A": A, "B": B}

    # Initialize Bayesian Multifactor ANOVA (with interactions)
    model = BayesianMultifactorANOVA(
        y=y,
        factors=factors,
        include_interactions=True,
        rope_main={"A": (-0.1, 0.1), "B": (-0.1, 0.1)},  # ROPE for marginal mean diffs
        rope_cell=None,
        seed=seed,
    )

    # Fit model (same parameter names as in the one-way example)
    model.fit(iter_sampling=5000, iter_warmup=1000, chains=4)

    # Analysis results (same structure & paths as in the one-way example)
    results = model.analyse(
        posterior_predictive_check=True,
        file_name='example_multifactor_anova',
        plot=True,
        save=True,
        round_to=3,
        directory_path='examples/results',
        file_path='bayesian_multifactor_anova', 
    )

    # Save the results in the same timestamp folder as the plots
    out_dir = f"examples/results/bayesian_multifactor_anova/{model._execution_time}"
    save_results(results, out_dir, "example_multifactor_anova")

    return {"results": results, "out_dir": out_dir}


def main():
    seed = 42
    np.random.seed(seed)

    # Run Multifactor ANOVA example (performs fit, analysis, save)
    run_out = example_multifactor_anova(seed)
    out_dir = run_out["out_dir"]

    # Load results from file (no hardcoding of timestamp)
    loaded = load_results(file_path=out_dir, file_name="example_multifactor_anova")

    # Print some key results
    print("Eta^2 (variance explained):", loaded.get('eta_sq'))

    # Pairwise ROPE summaries (falls vorhanden) – Liste mit Eintrag je Faktor
    if 'pairwise' in loaded:
        for entry in loaded['pairwise']:
            fname = entry.get('name', entry.get('factor', '?'))
            pairs = entry['pairs']
            lp = entry['left_prob']
            ip = entry['rope_prob']
            rp = entry['right_prob']
            print(f"\nPairwise ROPE results for factor {fname}:")
            for p, l, i, r in zip(pairs, lp, ip, rp):
                print(f"  {p}: P(<ROPE)={l}, P(ROPE)={i}, P(>ROPE)={r}")


if __name__ == '__main__':
    main()
