import numpy as np
from utils.helper import load_results, save_results
from utils.plotting import plot_densities
from bayesian_test.BayesianOneWayANOVA import BayesianOneWayANOVA


def example_anova(seed: int) -> dict:
    """
    Example of Bayesian One-way ANOVA with synthetic data.

    :param seed: Random seed for reproducibility.
    :return: Dict with results returned by model.analyse()
    """
    n_instances = 50  # Number of observations per group

    # Generate synthetic data for 3 groups
    group_A = np.random.normal(loc=0.0, scale=0.5, size=n_instances)
    group_B = np.random.normal(loc=0.5, scale=0.5, size=n_instances)
    group_C = np.random.normal(loc=1.0, scale=0.5, size=n_instances)

    # Combine into y and groups arrays
    y = np.concatenate([group_A, group_B, group_C])
    groups = np.array(['A'] * n_instances + ['B'] * n_instances + ['C'] * n_instances)

    # Plot densities per group
    data = np.stack((group_A, group_B, group_C), axis=1)
    plot_densities(data, algorithm_labels=['Group A', 'Group B', 'Group C'], show_plt=True, alpha=0.8)

    # Initialize Bayesian One-way ANOVA
    model = BayesianOneWayANOVA(y=y, groups=groups, rope=(-0.1, 0.1), seed=seed)

    # Fit model
    model.fit(iter_sampling=5000, iter_warmup=1000, chains=4)

    # Analyse results
    results = model.analyse(
        posterior_predictive_check=True,
        file_name='example_anova',
        plot=True,
        save=True,
        round_to=3,
        directory_path='examples/results',
        file_path='bayesian_one_way_anova',  
    )

    # Save the results in the same timestamp folder as the plots
    out_dir = f"examples/results/bayesian_one_way_anova/{model._execution_time}"
    save_results(results, out_dir, "example_anova")

    return {"results": results, "out_dir": out_dir}


def main():
    seed = 42
    np.random.seed(seed)

    # Run ANOVA example (performs fit, analysis, save)
    run_out = example_anova(seed)
    out_dir = run_out["out_dir"]

    # Load results from file (no more hardcoding of timestamp)
    loaded = load_results(file_path=out_dir, file_name="example_anova")

    # Print some key results
    print("Eta^2 (variance explained):", loaded['eta_sq'])
    if 'pairwise' in loaded:
        for pair, lp, ip, rp in zip(
            loaded['pairwise']['pairs'],
            loaded['pairwise']['left_prob'],
            loaded['pairwise']['rope_prob'],
            loaded['pairwise']['right_prob']
        ):
            print(f"Pair {pair}: P(<ROPE)={lp}, P(ROPE)={ip}, P(>ROPE)={rp}")


if __name__ == '__main__':
    main()
