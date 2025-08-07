import numpy as np
from utils.helper import load_results
from utils.plotting import plot_densities
from examples.data.DataLoader import DataLoader
from bayesian_test.BayesianCorrelatedTTest import BayesianCorrelatedTTest


def example_01(seed: int) -> None:
    """
    An example of Bayesian Correlated T-Test with synthetic data.

    :param seed: Random seed for reproducibility.
    :return: None
    """
    n_cv = 10  # Number of cross-validation folds
    n_runs = 10  # Number of runs
    n_instances = n_cv * n_runs  # Total number of samples
    rho = 1 / n_cv  # Correlation coefficient

    # Generate synthetic data for y1 and y2
    y1 = np.random.normal(loc=0.1, scale=0.4, size=n_instances)
    y2 = np.random.normal(loc=0.3, scale=0.6, size=n_instances)

    # Plot the density distribution for y1 and y2
    data = np.stack((y1, y2), axis=1)
    plot_densities(data, algorithm_labels=['Alg1', 'Alg2'], show_plt=True, alpha=1.0)

    # Initialize the Bayesian Correlated T-Test model
    model = BayesianCorrelatedTTest(y1=y1, y2=y2, rope=(-0.01, 0.01), rho=rho, seed=seed)

    # Fit the model with specified number of sampling iterations, warmup iterations and chains
    model.fit(iter_sampling=50000, iter_warmup=1000, chains=4)

    # Define plot parameters for posterior pdf plot
    plot_parameter = dict(
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
        font_size=12,  # Font size for text annotations. Default is 12.
    )

    # Analyze the model and generate plots
    model.analyse(posterior_predictive_check=True, file_name='example01', plot=True, save=True, round_to=3,
                  directory_path=f'examples/results', **plot_parameter)


def example_02(seed: int) -> None:
    """
    An example of Bayesian Correlated T-Test using real data.

    This is an adapted example from the scmamp repository:
    https://github.com/b0rxa/scmamp/blob/master/vignettes/Bayesian_analysis_of_the_differences.html#L194

    :param seed: Random seed for reproducibility.
    :return: None
    """
    # Load the dataset
    data_loader = DataLoader('data_kcv_example')

    # Extract samples for two algorithms from the dataset
    y1, y2 = data_loader.extract_samples_by_dataset(algorithms=['AlgA', 'AlgB'], dataset_id=5)

    # Plot the density distribution for y1 and y2
    data = np.stack((y1, y2), axis=1)
    plot_densities(data, algorithm_labels=['AlgA', 'AlgB'], show_plt=True, alpha=1.0)

    # Initialize the Bayesian Correlated T-Test model
    model = BayesianCorrelatedTTest(y1=y1, y2=y2, rope=(-0.01, 0.01), rho=0.1, seed=seed)

    # Fit the model with specified number of sampling iterations, warmup iterations and chains
    model.fit(iter_sampling=50000, iter_warmup=1000, chains=4)

    # Define plot parameters for posterior pdf plot
    plot_parameter = dict(
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
        font_size=12,  # Font size for text annotations. Default is 12.
    )

    # Analyze the model and generate plots
    model.analyse(posterior_predictive_check=True, file_name='example02', plot=True, save=True, round_to=3,
                  directory_path='examples/results', **plot_parameter)


def example_03(seed: int) -> None:
    """
    An example of Bayesian Correlated T-Test using real data.

    This is an adapted example from the scmamp repository:
    https://github.com/b0rxa/scmamp/blob/master/vignettes/Bayesian_analysis_of_the_differences.html#L194

    :param seed: Random seed for reproducibility.
    :return: None
    """
    # Load the dataset
    data_loader = DataLoader('data_kcv_example')

    # Extract samples for two algorithms from the dataset
    y1, y2 = data_loader.extract_samples_by_dataset(algorithms=['AlgC', 'AlgD'], dataset_id=5)

    # Plot the density distribution for y1 and y2
    data = np.stack((y1, y2), axis=1)
    plot_densities(data, algorithm_labels=['AlgC', 'AlgD'], show_plt=True, alpha=1.0)

    # Initialize the Bayesian Correlated T-Test model
    model = BayesianCorrelatedTTest(y1=y1, y2=y2, rope=(-0.01, 0.01), rho=0.1, seed=seed)

    # Fit the model with specified number of sampling iterations, warmup iterations and chains
    model.fit(iter_sampling=50000, iter_warmup=1000, chains=4)

    # Define plot parameters for posterior pdf plot
    plot_parameter = dict(
        hpdi_prob=0.99,  # Highest Posterior Density Interval (HPDI) probability. Default is 0.99.
        plot_type='hist',  # Type of plot ('kde' or 'hist'). Default is 'kde'.
        n_bins=50,  # Number of bins for histogram if plot_type is 'hist'. Default is 50.
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
        font_size=12,  # Font size for text annotations. Default is 12.
    )

    # Analyze the model and generate plots
    model.analyse(posterior_predictive_check=True, file_name='example03', plot=True, save=True, round_to=3,
                  directory_path=f'examples/results', **plot_parameter)


def main():
    seed = 42  # Set the random seed for reproducibility

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Run the example of a Bayesian Correlated T-Test
    example_01(seed)
    example_02(seed)
    example_03(seed)

    # Load saved results from example01
    results = load_results(file_path='examples/results/bayesian_correlated_t_test/250805_1723', file_name='example01')
    print(results['posterior_probabilities'])


if __name__ == '__main__':
    main()  # Execute the main function
