"""
von Pilchau, W. P., Pätzel, D., Stein, A., & Hähner, J. (2023, June).
Deep Q-Network Updates for the Full Action-Space Utilizing Synthetic Experiences.
In 2023 International Joint Conference on Neural Networks (IJCNN) (pp. 1-9). IEEE.
https://doi.org/10.1007/s10994-015-5486-z

Paper URL: https://ieeexplore.ieee.org/abstract/document/10191853
Code URL: https://github.com/dpaetzel/cmpbayes/blob/add-beta-binomial/src/cmpbayes/stan/nonnegative.stan
"""
import warnings
import cmdstanpy
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from datetime import datetime
from bayesian_test.AbstractBayesian import AbstractBayesian
from utils.plotting import plot_posterior_predictive_check, plot_posterior_pdf
from bayesian_test.utils import print_result, posterior_predictive_check_metrics, calculate_statistics


def calculate_probabilities(rope: tuple[float, float], posterior: np.ndarray) -> np.ndarray:
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


class BayesianNonNegativeUnimodalTest(AbstractBayesian):
    def __init__(self, y1: np.ndarray, y2: np.ndarray, rope: Optional[tuple[float, float]], var_lower: Optional[float],
                 var_upper: Optional[float], mean_upper: Optional[float], n_censored1: int = 0, n_censored2: int = 0,
                 censoring_point: Optional[float] = 3000.0, seed: int = 42):
        """
        Initialize the Bayesian non-negative Unimodal Test.

        :param y1: 1D array (num_instances,) of first datapoints.
            This array represents the independently generated performance statistics of the first method being compared.
            Each element corresponds to a single instance. The data should be unimodally distributed and non-negative.
        :param y2: 1D array (num_instances,) of second datapoints.
            This array represents the independently generated performance statistics of the second method being compared.
            Each element corresponds to a single instance. The data should be unimodally distributed and non-negative.
        :param rope: Region of Practical Equivalence (ROPE) as a tuple (min, max).
            This defines the range of values within which the differences between the two methods are considered
            practically equivalent. If the difference falls within this range, the two methods are deemed to have
            no significant difference in practical terms.
        :param var_lower: Lower bound of the variance.
            This parameter sets the lower bound for the hyperprior on the variances. It assumes that the variances of
            the submodels lie within the range [var_lower * Var(y), var_upper * Var(y)] for y from {y1, y2}. If `None`,
            the default non-committal  values of `0.001` are used, as suggested by Kruschke (2013).
        :param var_upper: Upper bound of the variance.
            This parameter sets the upper bound for the hyperprior on the variances, defining the range within which the
            variances  of the submodels are assumed to lie.
        :param mean_upper: Upper bound of the mean.
            This parameter defines the upper bound for the hyperprior on the means. It is assumed that, with a
            probability of 90%,  the true data means lie within the range [min(y)/100, min(y)/100 + mean_upper].
            If `None`, the default value of `2 * max(y)` is used.
        :param n_censored1: Number of censored data points in y1 (must be ≥ 0). Default is 0.
            This parameter specifies the number of censored data points in the first dataset (y1). Censoring occurs when
            data points exceed a certain value and are truncated at that point.
        :param n_censored2: Number of censored data points in y2 (must be ≥ 0). Default is 0.
            This parameter specifies the number of censored data points in the second dataset (y2). Censoring occurs
            when data points exceed a certain value and are truncated at that point.
        :param censoring_point: Censoring point, the value above which data has been censored. Default is 3000.
            This parameter defines the censoring point, which is the value above which data points have been censored.
            It is only relevant if at least one of `n_censored1` or `n_censored2` is greater than 0.
        :param seed: Random seed for reproducibility. Default is 42.
            This parameter ensures that the random processes in the model are consistent across runs, allowing for
            reproducible results.
        """
        super(BayesianNonNegativeUnimodalTest, self).__init__(stan_file='bayesian_non-negative_unimodal_test.stan',
                                                              rope=rope, seed=seed)

        # Ensure there are no NaN values in the datasets
        assert not np.any(np.isnan(y1)), ('The dataset y1 contains NaN values. '
                                          'Please remove or handle these NaNs before proceeding.')
        assert not np.any(np.isnan(y2)), ('The dataset y2 contains NaN values. '
                                          'Please remove or handle these NaNs before proceeding.')

        # Ensure there are no negative values in the datasets
        assert not np.any(y1 < 0), ('The dataset y1 contains negative values. '
                                    'The Non-Negative Test is designed for positive inputs only. '
                                    'Please verify and correct your input data.')
        assert not np.any(y2 < 0), ('The dataset y2 contains negative values. '
                                    'The Non-Negative Test is designed for positive inputs only. '
                                    'Please verify and correct your input data.')

        # Ensure the correctness of model parameters
        if var_lower:
            assert not var_lower <= 0, 'The var_lower value must be greater than 0.'
        if var_upper:
            assert not var_upper <= 0, 'The var_upper value must be greater than 0.'
        if not n_censored1:
            assert not n_censored1 < 0, 'The n_censored1 value must be equal or greater than 0.'
        if not n_censored2:
            assert not n_censored2 < 0, 'The n_censored2 value must be equal or greater than 0.'
        if censoring_point:
            assert not censoring_point <= 0, 'The censoring point must be greater than 0.'
        self.y1 = y1
        self.y2 = y2
        self.var_lower = var_lower
        self.var_upper = var_upper
        self.mean_upper = mean_upper
        self.n_censored1 = n_censored1
        self.n_censored2 = n_censored2
        self.censoring_point = censoring_point

    def _transform_data(self) -> dict:
        """
        Transform the data for the Stan model.

        :return: Dictionary containing the transformed data.
        """
        # Ensure var_lower is set, defaulting to 0.001 if None
        self.var_lower = self.var_lower if self.var_lower is not None else 0.001

        # Ensure var_upper is set, defaulting to 1000.0 if None
        self.var_upper = self.var_upper if self.var_upper is not None else 1000.0

        # Ensure mean_upper is set, defaulting to 2.0 * max(y2) if None
        self.mean_upper = self.mean_upper if self.mean_upper is not None else 2.0 * self.y2.max()

        # Get the number of runs (instances) in y1 and y2
        n_runs1, = self.y1.shape
        n_runs2, = self.y2.shape

        # Return a dictionary containing all the transformed data needed for the Stan model
        return dict(
            n_runs1=n_runs1,  # Number of runs in the first data set
            n_runs2=n_runs2,  # Number of runs in the second data set
            y1=self.y1,  # Data points for the first method
            y2=self.y2,  # Data points for the second method
            var_lower=self.var_lower,  # Lower bound of the variance
            var_upper=self.var_upper,  # Upper bound of the variance
            mean_upper=self.mean_upper,  # Upper bound of the mean
            n_censored1=self.n_censored1,  # Number of censored data points in y1
            n_censored2=self.n_censored2,  # Number of censored data points in y2
            censoring_point=self.censoring_point,  # Censoring point, the value above which data has been censored.
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
        :param font_size: Font size for the plot text elements. Default is 12
        :param save: Whether to save the plot to file. Default is True.
        :return: None
        """
        print(f'{datetime.now().time().strftime("%H:%M:%S")} - INFO: Running posterior predictive check.')
        # Retrieve posterior predictive samples
        y1_rep = self._fit.stan_variable('y1_rep')
        y2_rep = self._fit.stan_variable('y2_rep')
        n_cd, n_samples = y1_rep.shape

        # Calculate and print PPC Metrics
        metrics_y1 = [posterior_predictive_check_metrics(self.y1, y1_rep[i], ranks=False) for i in range(n_cd)]
        metrics_y2 = [posterior_predictive_check_metrics(self.y2, y2_rep[i], ranks=False) for i in range(n_cd)]

        means_y1, std_devs_y1 = calculate_statistics(metrics_y1)
        means_y2, std_devs_y2 = calculate_statistics(metrics_y2)
        print('\nPosterior Predictive Check Metrics')
        print(f'y1:\nMeans: {means_y1}\nStdDevs: {std_devs_y1}\n'
              f'y2:\nMeans: {means_y2}\nStdDevs: {std_devs_y2}\n')

        # Reshape _rep to have dimensions (chain, draw, n_samples)
        n_draws = int(n_cd / self.chains)
        y1_rep = y1_rep.reshape((self.chains, n_draws, n_samples))
        y2_rep = y2_rep.reshape((self.chains, n_draws, n_samples))

        # Adding posterior predictive samples to InferenceData
        self.inf_data.add_groups(posterior_predictive=dict(y1_rep=y1_rep, y2_rep=y2_rep))

        # Suppress warning of incorrect dimensions for observed data
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="the default dims 'chain' and 'draw' will be added automatically")
            self.inf_data.add_groups(observed_data=dict(y1=self.y1, y2=self.y2))

        # Generate the PPC plot
        variables = ['y1', 'y2']
        plot_posterior_predictive_check(inf_data=self.inf_data, variables=variables, n_draws=n_draws, show_plt=not save,
                                        font_size=font_size, seed=self.seed)

        # Save the plot if requested
        if save:
            self.save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name)
        else:
            plt.tight_layout()

        # Plot Posterior densities in the style of John K. Kruschke’s book.
        az.plot_posterior(self.inf_data)
        plt.show()

    def analyse(self, posterior_predictive_check: bool = True, plot: bool = True, save: bool = True, round_to: int = 4,
                directory_path: str = 'results', file_path: str = 'bayesian_non_negative_unimodal_test',
                file_name: Optional[str] = None, **kwargs) -> dict:
        """
        Analyse the results using the Bayesian correlated t-test.

        :param posterior_predictive_check: Whether to do a posterior predictive check. Default is True.
        :param plot: Whether to generate and display posterior distribution plot. Default is True.
        :param save: Whether to save the results and plots to files. Default is True.
        :param round_to: Number of decimal places to round the results to. Default is 4.
        :param directory_path: Path to the parent directory where the files are stored. Default is 'results'.
        :param file_path: Directory path to save the files. Default is 'bayesian_non_negative_unimodal_test'.
        :param file_name: Name of the file to save the results and plots. If None, a default name based on the current timestamp will be used.
        :param kwargs: Additional keyword arguments for customized analysis and plotting.
        :return: A dictionary containing the analysis results, including posterior probabilities and additional details.
        """
        # Perform PPC check
        if posterior_predictive_check:
            file_name_ppc = f'{self._execution_time}' if file_name is None else file_name
            self._posterior_predictive_check(directory_path, file_path, file_name=f'{file_name_ppc}_ppc', font_size=10,
                                             save=save)
        # Perform a simple analysis and print the summary
        summary = self._simple_analysis()

        mean = summary.loc['difference_mean', 'mean']
        sd = summary.loc['difference_mean', 'sd']

        # Extract samples from the posterior distribution
        samples = self._fit.stan_variable('difference_mean')

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
            mean_upper=self.mean_upper,  # Upper bound of the mean.
            n_censored1=self.n_censored1,  # Number of censored data points for y1
            n_censored2=self.n_censored2,  # Number of censored data points for y2
            censoring_point=self.censoring_point,  # Censoring point, the value above which data has been censored.
            iter_sampling=self.iter_sampling,  # Number of draws from the posterior for each chain
            iter_warmup=self.iter_warmup,  # Number of warm-up (burn-in) samples
            chains=self.chains,  # Number of chains in the MCMC sampling
            sampling_parameters=self.sampling_parameters,  # Additional parameters for the MCMC sampling
            seed=self.seed  # Random seed for reproducibility
        )
        results = dict(
            method='Bayesian Non-Negative Unimodal Test',  # Method used for the analysis
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
