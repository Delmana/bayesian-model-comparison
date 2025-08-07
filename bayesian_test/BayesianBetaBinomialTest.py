"""
von Pilchau, W. P., Pätzel, D., Stein, A., & Hähner, J. (2023, June).
Deep Q-Network Updates for the Full Action-Space Utilizing Synthetic Experiences.
In 2023 International Joint Conference on Neural Networks (IJCNN) (pp. 1-9). IEEE.
https://doi.org/10.1007/s10994-015-5486-z

Paper URL: https://ieeexplore.ieee.org/abstract/document/10191853
Code URL: https://github.com/dpaetzel/cmpbayes/blob/add-beta-binomial/src/cmpbayes/stan/nonnegative.stan
"""
import numpy as np
import arviz as az
import scipy.stats as st
import matplotlib.pyplot as plt
from typing import Optional, Any
from datetime import datetime

from numpy import floating

from bayesian_test.utils import print_result
from utils.plotting import plot_posterior_pdf
from bayesian_test.AbstractBayesian import AbstractBayesian


def calculate_probabilities(rope: tuple[float, float], posterior: np.ndarray) -> tuple[
    floating[Any], floating[Any] | None, floating[Any]]:
    """
    Calculate the probabilities of being less than, within, and greater than the ROPE (Region of Practical Equivalence).

    :param rope: Tuple representing the ROPE interval.
    :param posterior: Posterior samples distribution.
    :return: A tuple with the probabilities of being less than, within, and greater than the ROPE.
    """
    if rope is None or rope == (0, 0):
        left_prob = np.mean(posterior < 0)
        rope_prob = None
        right_prob = np.mean(posterior > 0)
    else:
        left_prob = np.mean(posterior < rope[0])
        rope_prob = np.mean((posterior >= rope[0]) & (posterior <= rope[1]))
        right_prob = np.mean(posterior > rope[1])
    return left_prob, rope_prob, right_prob


class BayesianBetaBinomialTest(AbstractBayesian):
    def __init__(self, n_success1: int, n_success2: int, n_runs1: int, n_runs2: int,
                 rope: Optional[tuple[float, float]], alpha1: float = 1.0, alpha2: float = 1.0, beta1: float = 1.0,
                 beta2: float = 1.0, seed: int = 42):
        """
        Initialize the Bayesian Beta Binomial Test.

        :param n_success1: Number of times that the first method was successful when run `n_runs1`
            times independently (must be ≥ 0).
        :param n_success2: Number of times that the second method was successful when run `n_runs2`
            times independently (must be ≥ 0).
        :param n_runs1: Number of times that the first method was run (must be > 0).
        :param n_runs2: Number of times that the second method was run (must be > 0).
        :param rope: Region of Practical Equivalence (ROPE) as a tuple (min, max).
            This defines the range of values within which the differences between the two methods are considered
            practically equivalent. If the difference falls within this range, the two methods are deemed to have
            no significant difference in practical terms.
        :param alpha1: Alpha parameters for the beta prior for the first method (must be > 0). Default is 1.0.
            Setting both `alpha` and `beta` to 1 corresponds to a uniform prior and is the default.
            Note that `alpha` and `beta` can be seen as the effective number of prior observations, i.e.
            setting `alpha1 = 10` and `beta1 = 1` means that our prior belief is that the first method is successful
            in 10 of 11 cases and unsuccessful in 1 of 11 cases.
        :param alpha2: Alpha parameter for the beta prior for the second method (must be > 0). Default is 1.0.
        :param beta1: Beta parameter for the beta prior for the first method (must be ≥ 0). Default is 1.0.
        :param beta2: Beta parameter for the beta prior for the second method (must be ≥ 0). Default is 1.0.
        :param seed: Random seed for reproducibility. Default is 42.
            This parameter ensures that the random processes in the model are consistent across runs, allowing for
            reproducible results.
        """
        super(BayesianBetaBinomialTest, self).__init__(stan_file='', rope=rope, seed=seed)

        # Ensure the correctness of model parameters
        assert not n_success1 < 0, 'The n_success1 value must be greater or equal than 0.'
        assert not n_success2 < 0, 'The n_success2 value must be greater or equal than 0.'
        assert not n_runs1 <= 0, 'The n_runs1 value must be greater than 0.'
        assert not n_runs2 <= 0, 'The n_runs2 value must be greater than 0.'
        assert not alpha1 <= 0, 'The alpha1 value must be greater than 0.'
        assert not alpha2 <= 0, 'The alpha2 value must be greater than 0.'
        assert not beta1 <= 0, 'The beta1 value must be greater than 0.'
        assert not beta2 <= 0, 'The beta2 value must be greater than 0.'

        self.n_success1 = n_success1
        self.n_success2 = n_success2
        self.n_runs1 = n_runs1
        self.n_runs2 = n_runs2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2

    def _transform_data(self) -> dict:
        pass

    def fit(self, iter_sampling: int = 50000, **kwargs):
        """
        Fit the Bayesian Beta Binomial Test model.

        :param iter_sampling: Number of samples to generate. Default is 50000
        :param kwargs: Additional arguments (not used).
        :return: None
        """
        # Set seed for reproducibility
        np.random.seed(seed=self.seed)

        posterior1 = st.beta(a=self.alpha1 + self.n_success1, b=self.beta1 + self.n_runs1 - self.n_success1)
        posterior2 = st.beta(a=self.alpha2 + self.n_success2, b=self.beta2 + self.n_runs2 - self.n_success2)

        self._posterior_model = (posterior1, posterior2)

        sample1 = posterior1.rvs(iter_sampling)
        sample2 = posterior2.rvs(iter_sampling)

        self._fit: np.ndarray = sample1 - sample2
        self.inf_data: az.InferenceData = az.convert_to_inference_data(
            dict(sample1=sample1, sample2=sample2, difference=self._fit))

    def _posterior_predictive_check(self, directory_path: str, file_path: str,
                                    file_name: str = 'posterior_predictive_check', font_size: int = 12,
                                    save: bool = True) -> None:
        """
        This function performs posterior predictive checks and generates plots comparing the observed data
        to the posterior predictive distributions.

        :param directory_path: Path to the parent directory where the files are stored.
        :param file_path: Path where the plot should be saved.
        :param file_name: Name of the file to save the plot. Default is 'posterior_predictive_check'.
        :param font_size: Font size for the plot text elements. Default is 12.
        :param save: Whether to save the plot to file. Default is True.
        :return: None
        """
        print(f'{datetime.now().time().strftime("%H:%M:%S")} - INFO: Running posterior predictive check.')

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        x = np.linspace(0, 1, 100)
        label1 = 'p(A successful)'
        label2 = 'p(B successful)'
        ax.plot(x, self._posterior_model[0].pdf(x), label=label1)
        ax.plot(x, self._posterior_model[1].pdf(x), label=label2)
        ax.set_title('Posterior Predictive Check', fontsize=font_size + 2)

        # Set x-axis and y-axis label
        ax.set_xlabel('Success Probability', fontsize=font_size)
        ax.set_ylabel('Density', fontsize=font_size)

        ax.legend(fontsize=font_size)
        if save:
            self.save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name)
        else:
            plt.tight_layout()
        plt.show()

    def analyse(self, posterior_predictive_check: bool = True, plot: bool = True, save: bool = True, round_to: int = 4,
                directory_path: str = 'results', file_path: str = 'bayesian_beta_binomial_test',
                file_name: Optional[str] = None, **kwargs) -> dict:
        """
        Analyse the results using the Bayesian Beta Binomial Test model.

        :param posterior_predictive_check: Whether to do a posterior predictive check. Default is True.
        :param plot: Whether to generate and display plots. Default is True.
        :param save: Whether to save the results and plots to files. Default is True.
        :param round_to: Number of decimal places to round the results to. Default is 4.
        :param directory_path: Path to the parent directory where the files are stored. Default is 'results'.
        :param file_path: Directory path to save the files. Default is 'bayesian_beta_binomial_test'.
        :param file_name: Name of the file to save the results and plots. If None, a default name based on the current timestamp will be used.
        :param kwargs: Additional keyword arguments for customized analysis and plotting.
        :return: A dictionary containing the analysis results, including posterior probabilities and additional details.
        """
        # Perform PPC check
        if posterior_predictive_check:
            file_name_ppc = f'{self._execution_time}' if file_name is None else file_name
            self._posterior_predictive_check(directory_path, file_path, file_name=f'{file_name_ppc}_ppc',
                                             font_size=12, save=save)

        # Perform a simple analysis and print the summary
        # self._simple_analysis()
        posterior = self.inf_data.posterior

        samples = posterior.difference.to_numpy()[0]
        mean = np.mean(samples)
        sd = np.std(samples)

        left_prob, rope_prob, right_prob = calculate_probabilities(self.rope, samples)
        wr = rope_prob * 100 if rope_prob else None

        # Prepare the results
        posterior_probs = dict(
            left_prob=left_prob,  # Probability that the effect is less than the lower bound of the ROPE
            rope_prob=rope_prob,  # Probability that the effect is within the ROPE
            right_prob=right_prob,  # Probability that the effect is greater than the upper bound of the ROPE
        )
        additional = dict(
            samples=samples,  # Posterior samples of the parameter mu
            posterior_mean=mean,  # Mean of the posterior distribution
            posterior_sd=sd,  # Standard deviation of the posterior distribution
            within_rope=wr,  # Proportion of samples within the ROPE
        )
        parameters = dict(
            rope=self.rope,  # Region of Practical Equivalence (ROPE)
            n_success1=self.n_success1,  # Number of times that the first method was successful
            n_success2=self.n_success2,  # Number of times that the second method was successful
            n_runs1=self.n_runs1,  # Number of times that the first method was run
            n_runs2=self.n_runs2,  # Number of times that the second method was run
            alpha1=self.alpha1,  # Alpha parameter for the beta prior for the first method
            alpha2=self.alpha2,  # Alpha parameter for the beta prior for the second method
            beta1=self.beta1,  # Beta parameter for the beta prior for the first method
            beta2=self.beta2,  # Beta parameter for the beta prior for the second method
            iter_sampling=self.iter_sampling,  # Number of draws from the posterior for each chain
            seed=self.seed  # Random seed for reproducibility
        )

        results = dict(
            method='Bayesian Beta Binomial Test',  # Method used for the analysis
            inference_data=self.inf_data,  # arviz InferenceData: Container for inference data storage using xarray.
            parameters=parameters,  # Parameters used in the analysis
            posterior_probabilities=posterior_probs,  # Posterior probabilities
            additional=additional  # Additional details from the analysis
        )

        # Save the results if requested
        if save:
            self.save_results(results, directory_path=directory_path, file_path=file_path, file_name=file_name)

        # Print the rounded results
        rounded_results = print_result(results, round_to=round_to)
        print(f'\nPosterior Probabilities: {rounded_results["posterior_probabilities"]}')

        if self.rope:
            print(f'\nROPE: {self.rope}\nWithin ROPE: {rounded_results["additional"]["within_rope"]}%')

        # Plot the posterior distribution if requested
        if plot:
            plot_posterior_pdf(data=samples, rope=self.rope, within_rope=wr, mean=mean, round_to=round_to,
                               title=r'Posterior Distribution of  $\Delta \mu = \mu_1 - \mu_2$', show_plt=False,
                               **kwargs)
            # Save the plot if requested
            if save:
                self.save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name)
                plt.show()
            else:
                plt.tight_layout()
                plt.show()
        return results
