import numpy as np
from utils.helper import load_results
from utils.plotting import plot_densities
from bayesian_test.BayesianIndependentTTest import BayesianIndependentTTest


def example_01(seed: int) -> None:
    """
    An example of Bayesian Independent T-Test with synthetic data.

    :param seed: Random seed for reproducibility.
    :return: None
    """
    n_instances = 100  # Number of instances to generate for each sample

    # Generate synthetic data for y1 and y2 from normal distributions
    y1 = np.random.normal(loc=0.1, scale=0.4, size=n_instances)
    y2 = np.random.normal(loc=0.3, scale=0.6, size=n_instances)

    # Plot the density distribution for y1 and y2
    data = np.stack((y1, y2), axis=1)
    plot_densities(data, algorithm_labels=['Alg1', 'Alg2'], show_plt=True, alpha=1.0)

    # Initialize the Bayesian Independent T-Test model
    model = BayesianIndependentTTest(y1=y1, y2=y2, rope=(-0.01, 0.01), seed=seed)

    # Fit the model with specified number of sampling iterations, warmup iterations and chains
    model.fit(iter_sampling=50000, iter_warmup=1000, chains=4)

    # Define plot parameters for posterior pdf plot
    plot_parameter = dict(
        plt_imp_param=True,
        # Whether only the mu difference, sigma difference, effect size, mu1 and mu2 should be plotted. Default is False
        hpdi_prob=0.99,  # Highest Posterior Density Interval (HPDI) probability. Default is 0.99.
        plot_type='kde',  # Type of plot ('kde' or 'hist'). Default is 'kde'.
        plt_rope=True,  # Whether to plot the Region of Practical Equivalence (ROPE). Default is True.
        plt_rope_text=True,  # Whether to display text for the ROPE. Default is True.
        plt_within_rope=True,  # Whether to display the percentage of samples within the ROPE. Default is True.
        plt_mean=True,  # Whether to plot the mean of the samples. Default is False.
        plt_mean_text=True,  # Whether to display text for the mean. Default is False.
        plt_hpdi=True,  # Whether to plot the HPDI. Default is True.
        plt_hpdi_text=True,  # Whether to display text for the HPDI. Default is True.
        plt_samples=True,  # Whether to plot the sample points. Default is True.
        plt_title=True,  # Whether to display the title. Default is True.
        alpha=0.5,  # Transparency level for plot elements. Default is 0.8.
        font_size=10,  # Font size for text annotations. Default is 12.
    )

    # Analyze the model and generate plots
    model.analyse(posterior_predictive_check=True, file_name='example01', plot=True, save=True, round_to=3,
                  directory_path='examples/results', **plot_parameter)


def main():
    seed = 42  # Set the random seed for reproducibility

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Run the example of a Bayesian Independent T-Test
    example_01(seed)

    # Load saved results from example01
    results = load_results(file_path='examples/results/bayesian_independent_t_test/250805_1824', file_name='example01')
    for key in ['difference_mean', 'difference_sigma', 'effect_size', 'mu1', 'mu2', 'sigma1', 'sigma2', 'nu']:
        print(f'{key}: {results[key]["posterior_probabilities"]}')


if __name__ == '__main__':
    main()  # Execute the main function
