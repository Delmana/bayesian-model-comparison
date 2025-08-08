import os
from datetime import datetime
import numpy as np
from utils.helper import load_results
from bayesian_test.BayesianBetaBinomialTest import BayesianBetaBinomialTest


def example_01(seed: int) -> None:
    """
    An example of Bayesian non-negative Bimodal Test with synthetic data.

    Original code: https://github.com/dpaetzel/cmpbayes/blob/add-beta-binomial/scripts/examples.py#L138
    :param seed: Random seed for reproducibility.
    :return: None
    """

    n_success1 = 2
    n_success2 = 3
    n_runs1 = 20
    n_runs2 = 21

    model = BayesianBetaBinomialTest(n_success1=n_success1, n_success2=n_success2, n_runs1=n_runs1,
                                     n_runs2=n_runs2, rope=(-0.01, 0.01), seed=seed)
    model.fit()

    plot_parameter = dict(
        hpdi_prob=0.99,  # Highest Posterior Density Interval (HPDI) probability. Default is 0.99.
        plot_type='hist',  # Type of plot ('kde' or 'hist'). Default is 'kde'.
        n_bins=100,  # Number of bins for histogram if plot_type is 'hist'. Default is 50.
        plt_rope=True,  # Whether to plot the Region of Practical Equivalence (ROPE). Default is True.
        plt_rope_text=True,  # Whether to display text for the ROPE. Default is True.
        plt_within_rope=True,  # Whether to display the percentage of samples within the ROPE. Default is True.
        plt_mean=True,  # Whether to plot the mean of the samples. Default is False.
        plt_mean_text=True,  # Whether to display text for the mean. Default is False.
        plt_hpdi=True,  # Whether to plot the HPDI. Default is True.
        plt_hpdi_text=True,  # Whether to display text for the HPDI. Default is True.
        plt_samples=False,  # Whether to plot the sample points. Default is True.
        plt_title=True,  # Whether to display the title. Default is True.
        alpha=0.5,  # Transparency level for plot elements. Default is 0.8.
        font_size=12,  # Font size for text annotations. Default is 12.
    )

    # Analyze the model and generate plots
    model.analyse(posterior_predictive_check=True, file_name='example01', plot=True, save=True, round_to=3,
                  directory_path='examples/results', **plot_parameter)


def main():
    seed = 42  # Set the random seed for reproducibility

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Run the example of a Bayesian Non-Negative Bimodal
    example_01(seed)

    now = datetime.now()
    folder_name = now.strftime("%y%m%d_%H%M") 
    result_path = os.path.join('examples/results/bayesian_beta_binomial_test', folder_name)

    # Load saved results from example01
    results = load_results(file_path=result_path, file_name='example01')
  
    #results = load_results(file_path='examples/results/bayesian_beta_binomial_test/250805_1727', file_name='example01')
    print(results['posterior_probabilities'])


if __name__ == '__main__':
    main()  # Execute the main function
