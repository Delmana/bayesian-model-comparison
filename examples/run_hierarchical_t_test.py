import numpy as np
import pandas as pd
from utils.helper import load_results
from examples.data.DataLoader import DataLoader
from bayesian_test.BayesianHierarchicalTTest import BayesianHierarchicalTTest


def example_01(seed: int) -> None:
    """
    Example of a Bayesian Hierarchical T-Test with synthetic data.

    :param seed: Random seed for reproducibility.
    :return: None
    """
    n_cv = 10  # Number of cross-validation folds
    n_runs = 10  # Number of runs
    rho = 1 / n_cv  # Correlation coefficient for cross-validation
    shift = 0.05  # Shift to introduce a difference between algorithms

    # Generate synthetic data for two algorithms across multiple datasets
    data = {
        'alg0': {
            'dataset0': np.random.normal(loc=0.1, scale=0.4, size=n_cv * n_runs),
            'dataset1': np.random.normal(loc=0.2, scale=0.2, size=n_cv * n_runs),
            'dataset2': np.random.normal(loc=0.8, scale=0.4, size=n_cv * n_runs),
            'dataset3': np.random.normal(loc=0.24, scale=0.2, size=n_cv * n_runs),
            'dataset4': np.random.normal(loc=0.6, scale=0.3, size=n_cv * n_runs),
            'dataset5': np.random.normal(loc=0.2, scale=0.4, size=n_cv * n_runs),
            'dataset6': np.random.normal(loc=0.3, scale=0.1, size=n_cv * n_runs),
            'dataset7': np.random.normal(loc=0.4, scale=0.2, size=n_cv * n_runs),
            'dataset8': np.random.normal(loc=0.14, scale=0.2, size=n_cv * n_runs),
            'dataset9': np.random.normal(loc=0.19, scale=0.3, size=n_cv * n_runs),
        },
        'alg1': {
            'dataset0': np.random.normal(loc=shift + 0.15, scale=0.3, size=n_cv * n_runs),
            'dataset1': np.random.normal(loc=shift + 0.18, scale=0.22, size=n_cv * n_runs),
            'dataset2': np.random.normal(loc=shift + 0.7, scale=0.1, size=n_cv * n_runs),
            'dataset3': np.random.normal(loc=shift + 0.16, scale=0.11, size=n_cv * n_runs),
            'dataset4': np.random.normal(loc=shift + 0.6, scale=0.13, size=n_cv * n_runs),
            'dataset5': np.random.normal(loc=shift + 0.17, scale=0.3, size=n_cv * n_runs),
            'dataset6': np.random.normal(loc=shift + 0.28, scale=0.22, size=n_cv * n_runs),
            'dataset7': np.random.normal(loc=shift + 0.55, scale=0.3, size=n_cv * n_runs),
            'dataset8': np.random.normal(loc=shift + 0.12, scale=0.21, size=n_cv * n_runs),
            'dataset9': np.random.normal(loc=shift + 0.18, scale=0.73, size=n_cv * n_runs),
        }
    }

    # Convert the data to a pandas DataFrame and reshape it
    data = pd.DataFrame.from_dict(data).unstack().apply(pd.Series)
    y1 = data.loc['alg0'].to_numpy()  # Extract data for algorithm 0
    y2 = data.loc['alg1'].to_numpy()  # Extract data for algorithm 1
    # Initialize the Bayesian Hierarchical T-Test model
    model = BayesianHierarchicalTTest(y1=y1, y2=y2, rope=(-0.01, 0.01), rho=rho, seed=seed)

    sampling_parameters = dict(max_treedepth=15, adapt_delta=0.99)
    model.fit(iter_sampling=20000, iter_warmup=10000, chains=8, **sampling_parameters)

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
    Example of a Bayesian Hierarchical T-Test using real data.

    This is an adapted example from the scmamp Repository:
    https://github.com/b0rxa/scmamp/blob/master/vignettes/Bayesian_analysis_of_the_differences.html#L324

    :param seed: Random seed for reproducibility.
    :return: None
    """
    # Load the dataset
    data_loader = DataLoader('data_kcv_example')
    y1, y2 = data_loader.extract_all_samples_by_algorithm(['AlgC', 'AlgD'])

    # Initialize the Bayesian Hierarchical T-Test model
    model = BayesianHierarchicalTTest(y1=y1, y2=y2, rope=(-0.01, 0.01), rho=0.1, seed=seed)
    # Fit the model with specified number of sampling iterations, warmup iterations and chains
    sampling_parameters = dict(max_treedepth=15, adapt_delta=0.99)
    model.fit(iter_sampling=20000, iter_warmup=10000, chains=8, **sampling_parameters)

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
        alpha=0.5  # Transparency level for plot elements. Default is 0.8.
    )

    # Analyze the model and generate plots
    model.analyse(posterior_predictive_check=True, file_name='example02', plot=True, save=True, round_to=3,
                  directory_path='examples/results', **plot_parameter)


def main():
    seed = 42  # Set the random seed for reproducibility

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Run the example Bayesian Hierarchical T-Test
    example_01(seed)
    example_02(seed)

    # Load saved results from example01
    results = load_results(file_path='examples/results/bayesian_hierarchical_t_test/250805_2000', file_name='example01')
    print(results['posterior_probabilities'])


if __name__ == '__main__':
    main()  # Execute the main function
