"""
Introduced by David Pätzel <david.paetzel@posteo.de>

Code URL: https://github.com/dpaetzel/cmpbayes/blob/add-nonnegative-unimodal/src/cmpbayes/__init__.py#L319
"""
import warnings
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Any
from datetime import datetime

from numpy import floating

from bayesian_test.AbstractBayesian import AbstractBayesian
from utils.plotting import plot_posterior_predictive_check, plot_posterior_pdf
from bayesian_test.utils import print_result, posterior_predictive_check_metrics, calculate_statistics


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


class BayesianNonNegativeBimodalTest(AbstractBayesian):
    def __init__(self, y1: np.ndarray, y2: np.ndarray, rope: Optional[tuple[float, float]], var_lower: Optional[float],
                 var_upper: Optional[float], seed: int = 42):
        """
        Initialize the Bayesian non-negative Binomial Test.

        :param y1: 1D array (num_instances,) of first datapoints.
            This array represents the independently generated performance statistics of the first method being compared.
            Each element corresponds to a single instance. The data should be bimodal distributed and non-negative.
        :param y2: 1D array (num_instances,) of second datapoints.
            This array represents the independently generated performance statistics of the second method being compared.
            Each element corresponds to a single instance. The data should be bimodal distributed and non-negative.
        :param rope: Region of Practical Equivalence (ROPE) as a tuple (min, max).
            This defines the range of values within which the differences between the two methods are considered
            practically equivalent. If the difference falls within this range, the two methods are deemed to have
            no significant difference in practical terms.
        :param var_lower: Lower border for the variances of the submodels (must be 0 < var_lower < var_upper).
            If not defined, a default value of 0.001 is used.
            It is assumed that the variances of the submodels lies within [var_lower * Var(y), var_upper * Var(y)] for
            y from {y1, y2}.
        :param var_upper: Upper border for the variances of the submodels (must be var_lower < var_upper).
        If not defined, a default value of 1.0 is used.
        :param seed: Random seed for reproducibility. Default is 42.
            This parameter ensures that the random processes in the model are consistent across runs, allowing for
            reproducible results.
        """
        super(BayesianNonNegativeBimodalTest, self).__init__(stan_file='bayesian_non-negative_bimodal_test.stan',
                                                             rope=rope, seed=seed)

        assert not np.any(
            np.isnan(y1)), 'The dataset y1 contains NaN values. Please remove or handle these NaNs before proceeding.'
        assert not np.any(
            np.isnan(y2)), 'The dataset y2 contains NaN values. Please remove or handle these NaNs before proceeding.'
        assert not np.any(
            y1 < 0), ('The dataset y1 contains negative values. The Non-Negative Test is designed for positive inputs '
                      'only. Please verify and correct your input data.')
        assert not np.any(
            y2 < 0), ('The dataset y2 contains negative values. The Non-Negative Test is designed for positive inputs '
                      'only. Please verify and correct your input data.')
        if var_lower:
            assert not var_lower <= 0, 'The var_lower value must be greater than 0.'
            assert not var_lower >= var_upper, 'The var_lower value must be less than the var_upper value.'
        if var_upper:
            assert not var_upper <= var_lower, 'The var_upper value must be greater than var_lower.'

        self.y1 = y1
        self.y2 = y2
        self.var_lower = var_lower
        self.var_upper = var_upper

    def _transform_data(self) -> dict:
        self.var_lower = self.var_lower if self.var_lower is not None else 0.001
        self.var_upper = self.var_upper if self.var_upper is not None else 1.0

        n_runs1, = self.y1.shape
        n_runs2, = self.y2.shape

        return dict(
            n_runs1=n_runs1,
            n_runs2=n_runs2,
            y1=self.y1,
            y2=self.y2,
            var_lower=self.var_lower,
            var_upper=self.var_upper,
        )

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
        y1_rep = self._fit.stan_variable('y1_rep')
        y2_rep = self._fit.stan_variable('y2_rep')
        n_cd, n_samples = y1_rep.shape

        metrics_y1 = [posterior_predictive_check_metrics(self.y1, y1_rep[i], ranks=False) for i in range(n_cd)]
        metrics_y2 = [posterior_predictive_check_metrics(self.y2, y2_rep[i], ranks=False) for i in range(n_cd)]

        means_y1, std_devs_y1 = calculate_statistics(metrics_y1)
        means_y2, std_devs_y2 = calculate_statistics(metrics_y2)
        print('\nPosterior Predictive Check Metrics')
        print(f'y1:\nMeans: {means_y1}\nStdDevs: {std_devs_y1}\n'
              f'y2:\nMeans: {means_y2}\nStdDevs: {std_devs_y2}\n')

        n_draws = int(n_cd / self.chains)
        y1_rep = y1_rep.reshape((self.chains, n_draws, n_samples))
        y2_rep = y2_rep.reshape((self.chains, n_draws, n_samples))

        self.inf_data.add_groups(posterior_predictive=dict(y1_rep=y1_rep, y2_rep=y2_rep))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="the default dims 'chain' and 'draw' will be added automatically")
            self.inf_data.add_groups(observed_data=dict(y1=self.y1, y2=self.y2))

        variables = ['y1', 'y2']
        plot_posterior_predictive_check(inf_data=self.inf_data, variables=variables, n_draws=n_draws, show_plt=not save,
                                        font_size=font_size, seed=self.seed)

        if save:
            self.save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name)
        else:
            plt.tight_layout()
        # Plot Posterior densities in the style of John K. Kruschke’s book.
        az.plot_posterior(self.inf_data)
        plt.show()

    def analyse(self, posterior_predictive_check: bool = True, plot: bool = True, save: bool = True, round_to: int = 4,
                directory_path: str = 'results', file_path: str = 'bayesian_non_negative_bimodal_test',
                file_name: Optional[str] = None, **kwargs) -> dict:
        """
        Analyse the results using the Bayesian non-negative bimodal test.

        :param posterior_predictive_check: Whether to do a posterior predictive check. Default is True.
        :param plot: Whether to generate and display plots. Default is True.
        :param save: Whether to save the results and plots to files. Default is True.
        :param round_to: Number of decimal places to round the results to. Default is 4.
        :param directory_path: Path to the parent directory where the files are stored. Default is 'results'.
        :param file_path: Directory path to save the files. Default is 'bayesian_non_negative_bimodal_test'.
        :param file_name: Name of the file to save the results and plots. If None, a default name based on the current timestamp will be used.
        :param kwargs: Additional keyword arguments for customized analysis and plotting.
        :return: A dictionary containing the analysis results, including posterior probabilities and additional details.
        """
        # Perform PPC check
        if posterior_predictive_check:
            file_name_ppc = f'{self._execution_time}' if file_name is None else file_name
            self._posterior_predictive_check(directory_path, file_path, file_name=f'{file_name_ppc}_ppc', font_size=10,
                                             save=save)

        self._simple_analysis()
        posterior = self.inf_data.posterior

        # Compute means over these dims.
        dim = ['chain', 'draw']
        # Calculate the posterior mean differences
        samples = (posterior.alpha1 / posterior.beta1 - posterior.alpha2 / posterior.beta2).stack(
            samples=dim).values.flatten()

        mean = np.mean(samples)
        sd = np.std(samples)

        # Calculate probabilities for the posterior distribution
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
            var_lower=self.var_lower,  # Lower bound of the variance.
            var_upper=self.var_upper,  # Upper bound of the variance.
            iter_sampling=self.iter_sampling,  # Number of draws from the posterior for each chain
            iter_warmup=self.iter_warmup,  # Number of warm-up (burn-in) samples
            chains=self.chains,  # Number of chains in the MCMC sampling
            sampling_parameters=self.sampling_parameters,  # Additional parameters for the MCMC sampling
            seed=self.seed  # Random seed for reproducibility
        )
        results = dict(
            method='Bayesian Non-Negative Bimodal Test',  # Method used for the analysis
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
