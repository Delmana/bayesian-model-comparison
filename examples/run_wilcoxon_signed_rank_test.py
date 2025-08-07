import numpy as np
from utils.helper import load_results
from examples.data.DataLoader import DataLoader
from bayesian_test.BayesianWilcoxonSignedRankTest import BayesianWilcoxonSignedRankTest


def example_01(seed: int) -> None:
    """
    Run an example Bayesian Wilcoxon Signed-Rank Test with synthetic data.

    :param seed: Random seed for reproducibility.
    :return: None
    """
    n_instances = 100  # Number of instances to generate for each sample
    # Generate synthetic data for y1 and y2 from beta distributions
    y1 = np.random.beta(a=2, b=6, size=n_instances)
    y2 = np.random.beta(a=1.5, b=5, size=n_instances)

    # Initialize the Bayesian Wilcoxon Signed-Rank Test model
    model = BayesianWilcoxonSignedRankTest(y1=y1, y2=y2, rope=(-0.01, 0.01), seed=seed)
    # Fit the model with specified number of samples
    model.fit(iter_sampling=50000, verbose=True)

    # Define plot parameters for the simplex plot
    plot_parameter = dict(
        algo_label1='y1',  # Label for the first algorithm. Default is 'Alg. A'.
        algo_label2='y2',  # Label for the second algorithm. Default is 'Alg. B'.
        plt_points=True,  # Whether to plot individual points. Default is True.
        plot_type='scatter',  # Type of plot ('scatter' or 'hexbin'). Default is 'scatter'.
        point_size=10,  # Size of points in the scatter plot. Default is 10.
        plt_density=True,  # Whether to plot density (ignored if plot_type is 'hexbin'). Default is True.
        posterior_label=True,  # Whether to label the posterior probabilities. Default is True.
        title='Simplex Plot of Posterior Probabilities',
        # Title of the plot. Default is 'Simplex Plot of Posterior Probabilities'.
        plt_title=True,  # Whether to display the title. Default is True.
        palette=('steelblue', 'cornflowerblue', 'royalblue'),
        # Tuple of colors for the plot. Default is ('steelblue', 'cornflowerblue', 'royalblue').
        font_size=12,  # Size of the front of the plot. Default is 12.
        alpha=0.8  # Transparency level for plot elements. Default is 0.8.
    )

    # Analyze the model and generate plots
    model.analyse(posterior_predictive_check=True, file_name='example01', plot=True, save=True, round_to=3,
                  directory_path='examples/results', **plot_parameter)


def example_02(seed: int) -> None:
    """
    Run an example Bayesian Wilcoxon Signed-Rank Test with synthetic data and without a defined ROPE.

    :param seed: Random seed for reproducibility.
    :return: None
    """
    n_instances = 100  # Number of instances to generate for each sample
    # Generate synthetic data for y1 and y2 from beta distributions
    y1 = np.random.beta(a=2, b=6, size=n_instances)
    y2 = np.random.beta(a=1.5, b=5, size=n_instances)

    # Initialize the Bayesian Wilcoxon Signed-Rank Test model
    model = BayesianWilcoxonSignedRankTest(y1=y1, y2=y2, rope=None, seed=seed)
    # Fit the model with specified number of samples
    model.fit(n_samples=50000, verbose=True)

    # Define plot parameters for the histogram plot
    plot_parameter = dict(
        algo_label1='y1',  # Label for the first algorithm. Default is 'Alg. A'.
        algo_label2='y2',  # Label for the second algorithm. Default is 'Alg. B'.
        n_bins=100,  # Number of bins for the histogram. Default is 50.
        posterior_label=True,  # Whether to label the posterior probabilities. Default is True.
        title='Histogram of Posterior Probabilities',
        # Title of the plot. Default is 'Histogram of Posterior Probabilities'.
        plt_title=True,  # Whether to display the title. Default is True.
        font_size=12,  # Size of the front of the plot. Default is 12.
        alpha=0.8  # Transparency level for plot elements. Default is 0.8.
    )

    # Analyze the model and generate plots
    model.analyse(posterior_predictive_check=True, file_name='example02', plot=True, save=True, round_to=3,
                  directory_path='examples/results', **plot_parameter)


def example_03(seed: int) -> None:
    """
    Run an example Bayesian Wilcoxon Signed-Rank Test using real data.

    This is an adapted example from the scmamp Repository:
    https://github.com/b0rxa/scmamp/blob/master/vignettes/Bayesian_analysis_of_the_differences.html#L234

    :param seed: Random seed for reproducibility.
    :return: None
    """
    # Load the dataset
    data_loader = DataLoader('data_kcv_example')
    # Extract samples for two algorithms from the dataset
    y1, y2 = data_loader.extract_summarized_samples_by_dataset(['AlgC', 'AlgD'], dataset_id=9)

    # Initialize the Bayesian Wilcoxon Signed-Rank Test model
    model = BayesianWilcoxonSignedRankTest(y1=y1, y2=y2, rope=(-0.01, 0.01), seed=seed)
    # Fit the model with specified number of samples
    model.fit(n_samples=50000, verbose=True)

    # Define plot parameters for the simplex plot
    plot_parameter = dict(
        algo_label1='AlgC',  # Label for the first algorithm. Default is 'Alg. A'.
        algo_label2='AlgD',  # Label for the second algorithm. Default is 'Alg. B'.
        plt_points=True,  # Whether to plot individual points. Default is True.
        plot_type='scatter',  # Type of plot ('scatter' or 'hexbin'). Default is 'scatter'.
        point_size=10,  # Size of points in the scatter plot. Default is 10.
        plt_density=True,  # Whether to plot density (ignored if plot_type is 'hexbin'). Default is True.
        posterior_label=True,  # Whether to label the posterior probabilities. Default is True.
        title='Simplex Plot of Posterior Probabilities',
        # Title of the plot. Default is 'Simplex Plot of Posterior Probabilities'.
        plt_title=True,  # Whether to display the title. Default is True.
        palette=('steelblue', 'cornflowerblue', 'royalblue'),
        # Tuple of colors for the plot. Default is ('steelblue', 'cornflowerblue', 'royalblue').
        font_size=12,  # Size of the front of the plot. Default is 12.
        alpha=0.8  # Transparency level for plot elements. Default is 0.8.
    )

    # Analyze the model and generate plots
    model.analyse(posterior_predictive_check=True, file_name='example03', plot=True, save=True, round_to=3,
                  directory_path='examples/results', **plot_parameter)


def example_04(seed: int) -> None:
    """
    Run an example Bayesian Wilcoxon Signed-Rank Test using real data.

    This is an adapted example from the scmamp Repository:
    https://github.com/b0rxa/scmamp/blob/master/vignettes/Bayesian_analysis_of_the_differences.html#L234

    :param seed: Random seed for reproducibility.
    :return: None
    """
    # Load the dataset
    data_loader = DataLoader('data_kcv_example')
    # Extract samples for two algorithms from the dataset
    y1, y2 = data_loader.extract_summarized_samples_by_dataset(['AlgC', 'AlgD'], dataset_id=8)

    # Initialize the Bayesian Wilcoxon Signed-Rank Test model
    model = BayesianWilcoxonSignedRankTest(y1=y1, y2=y2, rope=(-0.01, 0.01), seed=seed)
    # Fit the model with specified number of samples
    model.fit(n_samples=50000)

    # Define plot parameters for the simplex plot
    plot_parameter = dict(
        algo_label1='AlgC',  # Label for the first algorithm. Default is 'Alg. A'.
        algo_label2='AlgD',  # Label for the second algorithm. Default is 'Alg. B'.
        plt_points=True,  # Whether to plot individual points. Default is True.
        plot_type='scatter',  # Type of plot ('scatter' or 'hexbin'). Default is 'scatter'.
        point_size=10,  # Size of points in the scatter plot. Default is 10.
        plt_density=True,  # Whether to plot density (ignored if plot_type is 'hexbin'). Default is True.
        posterior_label=True,  # Whether to label the posterior probabilities. Default is True.
        title='Simplex Plot of Posterior Probabilities',
        # Title of the plot. Default is 'Simplex Plot of Posterior Probabilities'.
        plt_title=True,  # Whether to display the title. Default is True.
        palette=('steelblue', 'cornflowerblue', 'royalblue'),
        # Tuple of colors for the plot. Default is ('steelblue', 'cornflowerblue', 'royalblue').
        font_size=12,  # Size of the front of the plot. Default is 12.
        alpha=0.8  # Transparency level for plot elements. Default is 0.8.
    )

    # Analyze the model and generate plots
    model.analyse(posterior_predictive_check=True, file_name='example04', plot=True, save=True, round_to=3,
                  directory_path='examples/results', **plot_parameter)


def main():
    seed = 42  # Set the random seed for reproducibility

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Run the example Bayesian Wilcoxon Signed-Rank Tests
    example_01(seed)
    example_02(seed)
    example_03(seed)
    example_04(seed)

    # Load saved results from example01
    results = load_results(file_path='examples/results/bayesian_wilcoxon_signed_rank_test/250806_1444',
                           file_name='example01')
    print(results['posterior_probabilities'])


if __name__ == '__main__':
    main()  # Execute the main function
