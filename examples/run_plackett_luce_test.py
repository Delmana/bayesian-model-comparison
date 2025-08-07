import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any
from numpy import floating

from utils.helper import load_results
from examples.data.DataLoader import DataLoader
from bayesian_test.BayesianPlackettLuceTest import BayesianPlackettLuceTest


def _plot_post_analysis(samples: np.ndarray, x_label: str, n_bins: int = 20, alpha: float = 1.0,
                        round_to: int = 4) -> None:
    """
    Perform post-analysis on the given samples and plot a histogram.

    :param samples: An array of sample values to analyze.
    :param x_label: The label for the x-axis of the plot.
    :param n_bins: The number of bins for the histogram (default is 20).
    :param alpha: The transparency level for the histogram (default is 1.0).
    :param round_to: The number of decimal places to round the expected probability (default is 4).
    :return: None
    """
    expected_prob = round(np.mean(samples), round_to)
    sns.histplot(samples, bins=n_bins, alpha=alpha)
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.title(f'Expected probability is {expected_prob}')
    plt.tight_layout()
    plt.show()


def _estimate_probability_better(df: pd.DataFrame, col1: str, col2: str) -> floating[Any]:
    """
    Estimate the probability that values in col1 are greater than values in col2.

    :param df: A pandas DataFrame containing the data.
    :param col1: The name of the first column to compare.
    :param col2: The name of the second column to compare.
    :return: The estimated probability that values in col1 are greater than values in col2, rounded to 4 decimal places.
    """
    probability = np.mean(df[col1].values > df[col2].values)
    return np.round(probability, 4)


def example_01(data_loader: DataLoader, size: int, seed: int) -> None:
    """
    Run an example Bayesian Plackett-Luce Test using real data.

    This is an adapted example from the scmamp Repository:
    https://github.com/b0rxa/scmamp/blob/master/vignettes/Bayesian_analysis_of_algorithm_rankings.html#L219

    :param data_loader: An instance of DataLoader to load the data.
    :param size: The size of the sample to extract.
    :param seed: Random seed for reproducibility.
    :return: None
    """
    # Load the dataset
    sample, algorithm_labels = data_loader.extract_sample_by_size(size=size)

    # Initialize the Bayesian Plackett-Luce Test model
    model = BayesianPlackettLuceTest(x_matrix=sample, algorithm_labels=algorithm_labels, minimize=False, seed=seed)

    # Fit the model with specified number of sampling iterations, warmup iterations and chains
    model.fit(iter_sampling=10000, iter_warmup=2000, chains=4)

    # Define plot parameters for boxplot
    plot_parameter = dict(
        title='Boxplot of Posterior Weights',  # Title of the plot. Default is 'Boxplot of Posterior Weights'.
        x_label='Algorithm',  # Label of the x-Axis. Default is 'Algorithm'.
        y_label='Probability',  # Label of y-Axis. Default is 'Probability'.
        plt_title=True,  # Whether to display the title. Default is True.
        font_size=12,  # Size of the front of the plot. Default is 12.
        alpha=1  # Transparency level for plot elements. Default is 1.0.
    )

    # Analyze the model and generate plots
    model.analyse(posterior_predictive_check=True, file_name='example01', plot=True, save=True, round_to=3,
                  directory_path='examples/results', **plot_parameter)


def example_02(data_loader: DataLoader, size: int, seed: int) -> None:
    """
    Run a second example Bayesian Placket-Luce Test using a subset of real data.

    This is an adapted example from the scmamp Repository:
    https://github.com/b0rxa/scmamp/blob/master/vignettes/Bayesian_analysis_of_algorithm_rankings.html#L219

    :param data_loader: An instance of DataLoader to load the data.
    :param size: The size of the sample to extract.
    :param seed: Random seed for reproducibility.
    :return: None
    """
    # Load the dataset and extract a subset
    sample, algorithm_labels = data_loader.extract_sample_by_size(size=size)
    sub_sample = sample[:, [0, 6, 7]]
    algorithm_labels = ['FruitFly', 'FrogCOL', 'FrogMIS']

    # Initialize the Bayesian Plackett-Luce Test model
    model = BayesianPlackettLuceTest(x_matrix=sub_sample, algorithm_labels=algorithm_labels, minimize=False, seed=seed)

    # Fit the model with specified number of sampling iterations, warmup iterations and chains
    model.fit(iter_sampling=10000, iter_warmup=2000, chains=4)

    # Define plot parameters for boxplot
    plot_parameter = dict(
        title='Boxplot of Posterior Weights',  # Title of the plot. Default is 'Boxplot of Posterior Weights'.
        x_label='Algorithm',  # Label of the x-Axis. Default is 'Algorithm'.
        y_label='Probability',  # Label of y-Axis. Default is 'Probability'.
        plt_title=True,  # Whether to display the title. Default is True.
        font_size=12,  # Size of the front of the plot. Default is 12.
        alpha=1  # Transparency level for plot elements. Default is 1.0.
    )

    # Analyze the model and generate plots
    model.analyse(posterior_predictive_check=True, file_name='example02', plot=True, save=True, round_to=3,
                  directory_path='examples/results', **plot_parameter)


def example01_post_analysis(data_loader: DataLoader, size: int) -> None:
    """
    Perform post-analysis on the results of the first example.

    :param data_loader: An instance of DataLoader to load the data.
    :param size: The size of the sample to extract.
    :return: None
    """
    # Load the results from the pickle file
    results = load_results(file_path='examples/results/bayesian_plackett_luce_test/250806_1023', file_name='example01')

    # Extract posterior weights
    frogCOL_weights = results['posterior_weights']['FrogCOL'].values
    frogMIS_weights = results['posterior_weights']['FrogMIS'].values
    fruitFly_weights = results['posterior_weights']['FruitFly'].values

    # Perform post-analysis and plot histograms
    _plot_post_analysis(frogCOL_weights, x_label=f'Probability of FrogCOL being the best.')

    better_col_mis = frogCOL_weights / (frogCOL_weights + frogMIS_weights)
    better_col_fly = frogCOL_weights / (frogCOL_weights + fruitFly_weights)
    _plot_post_analysis(better_col_mis, x_label=f'Probability of FrogCOL better than FrogMIS')
    _plot_post_analysis(better_col_fly, x_label=f'Probability of FrogCOL better than FruitFly')

    # Estimate and print probabilities directly from the dataset
    df = data_loader.df
    df = df[df['Size'] == size]

    probability = _estimate_probability_better(df, 'FrogCOL', 'FruitFly')
    print(f'Directly estimated probability of FrogCOL being better than FruitFly: {probability}')

    probability = _estimate_probability_better(df, 'FrogMIS', 'Ikeda')
    print(f'Directly estimated probability of FrogMIS being better than Ikeda: {probability}')

    probability = _estimate_probability_better(df, 'FruitFly', 'Ikeda')
    print(f'Directly estimated probability of FruitFly being better than Ikeda: {probability}')


def example02_post_analysis(data_loader: DataLoader, size: int) -> None:
    """
    Perform post-analysis on the results of the second example.

    :param data_loader: An instance of DataLoader to load the data.
    :param size: The size of the sample to extract.
    :return: None
    """
    # Load the results from the pickle file
    sub_results = load_results(file_path='examples/results/bayesian_plackett_luce_test/250806_1038',
                               file_name='example02')

    # Extract posterior weights
    frogCOL_weights = sub_results['posterior_weights']['FrogCOL'].values
    frogMIS_weights = sub_results['posterior_weights']['FrogMIS'].values
    fruitFly_weights = sub_results['posterior_weights']['FruitFly'].values

    # Perform post-analysis and plot histograms
    better_col_mis = frogCOL_weights / (frogCOL_weights + frogMIS_weights)
    better_col_fly = frogCOL_weights / (frogCOL_weights + fruitFly_weights)
    _plot_post_analysis(better_col_mis, x_label=f'Probability of FrogCOL better than FrogMIS')
    _plot_post_analysis(better_col_fly, x_label=f'Probability of FrogCOL better than FruitFly')

    # Estimate and print probabilities directly from the dataset
    df = data_loader.df
    df = df[df['Size'] == size]

    probability = _estimate_probability_better(df, 'FrogCOL', 'FruitFly')
    print(f'Directly estimated probability of FrogCOL being better than FruitFly: {probability}')

    probability = _estimate_probability_better(df, 'FrogCOL', 'FrogMIS')
    print(f'Directly estimated probability of FrogCOL being better than FrogMIS: {probability}')

    probability = _estimate_probability_better(df, 'FrogMIS', 'FruitFly')
    print(f'Directly estimated probability of FrogMIS being better than FruitFly: {probability}')


def main():
    seed = 42  # Set the random seed for reproducibility

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Initialize DataLoader
    data_loader = DataLoader('data_blum_2015')
    size = 1000

    # Run the example Bayesian Plackett-Luce Test
    example_01(data_loader, size, seed)
    example_02(data_loader, size, seed)

    # Post analysis
    example01_post_analysis(data_loader, size)
    example02_post_analysis(data_loader, size)


if __name__ == '__main__':
    main()
