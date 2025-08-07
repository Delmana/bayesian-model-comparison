"""
Corani, G., Benavoli, A. A Bayesian approach for comparing cross-validated algorithms on multiple data sets.
Mach Learn 100, 285–304 (2015). https://doi.org/10.1007/s10994-015-5486-z

Paper URL: https://link.springer.com/article/10.1007/s10994-015-5486-z
"""
import warnings
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional
from datetime import datetime
from bayesian_test.AbstractBayesian import AbstractBayesian
from utils.plotting import plot_posterior_pdf, plot_posterior_predictive_check
from bayesian_test.utils import print_result, posterior_predictive_check_metrics, calculate_statistics


def _ppos(x: float, mean: float, sd: float, df: float) -> float:
    """
    Calculate the cumulative distribution function (CDF) of the Student's t-distribution up to a given value x.

    :param x: The value up to which to evaluate the CDF.
    :param mean: The mean of the distribution.
    :param sd: The standard deviation of the distribution.
    :param df: The degrees of freedom of the Student's t-distribution.
    :return: The cumulative distribution function value up to x.
    """
    return stats.t.cdf(x, df, loc=mean, scale=sd)


def calculate_probabilities(rope: tuple[float, float], mean: float, sd: float, nu: float) -> tuple[
    float, float | None, float]:
    """
    Calculate the probabilities of being less than, within, and greater than the ROPE (Region of Practical Equivalence).

    :param rope: Tuple representing the ROPE interval.
    :param mean: The mean of the distribution.
    :param sd: The standard deviation of the distribution.
    :param nu: The degrees of freedom of the Student's t-distribution.
    :return: A tuple with the probabilities of being less than, within, and greater than the ROPE.
    """
    if rope is None or rope == (0, 0):
        if sd == 0:
            right_prob = (mean > 0) + 0.5 * (mean == 0)  # Special case when standard deviation is zero
        else:
            right_prob = 1 - _ppos(0, mean, sd, nu)
        left_prob = 1 - right_prob
        rope_prob = None
    else:
        if sd == 0:
            left_prob = float(mean < rope[0])
            right_prob = float(mean > rope[1])
        else:
            left_prob = _ppos(rope[0], mean, sd, nu)
            right_prob = 1 - _ppos(rope[1], mean, sd, nu)
        rope_prob = 1 - left_prob - right_prob
    return left_prob, rope_prob, right_prob


class BayesianCorrelatedTTest(AbstractBayesian):
    def __init__(self, y1: np.ndarray, y2: np.ndarray, rope: Optional[tuple[float, float]], rho: float, seed: int = 42):
        """
        Initialize the Bayesian Correlated t-Test class.

         :param y1: 1D array (num_instances,) of first data points.
            This array represents the dependently generated performance statistics of the first method being compared.
            Each element corresponds to a single instance. The data should be normally distributed.
        :param y2: 1D array (num_instances,) of second data points.
            This array represents the dependently generated performance statistics of the second method being compared.
            Each element corresponds to a single instance. The data should be normally distributed.
        :param rope: Region of Practical Equivalence (ROPE) as a tuple (min, max).
            This defines the range of values within which the differences between the two methods are considered
            practically equivalent. If the difference falls within this range, the two methods are deemed to have
            no significant difference in practical terms.
        :param rho: Correlation coefficient (must be 0 < rho < 1).
            This parameter represents the correlation between the two sets of data points, influenced by the
            cross-validation procedure. It is used to estimate the correlation introduced by cross-validation, based on
            the fraction of the data used as the test set (i.e., `n_test / (n_test + n_train)`), as described by
            Nadeau and Bengio (2003).
        :param seed: Random seed for reproducibility. Default is 42.
            This parameter ensures that the random processes in the model are consistent across runs, allowing for
            reproducible results.
        """
        super(BayesianCorrelatedTTest, self).__init__(stan_file='bayesian_correlated_t-test.stan', rope=rope, seed=seed)

        n_runs1 = y1.shape[0]
        n_runs2 = y2.shape[0]

        # Ensure the number of runs in both datasets are equal
        assert n_runs1 == n_runs2, (f'The correlated T-Test is a paired test. The number of runs in the first dataset '
                                    f'({n_runs1}) must match the number of runs in the second dataset ({n_runs2}).')
        # Ensure there are no NaN values in the datasets
        assert not np.any(np.isnan(y1)), ('The dataset y1 contains NaN values. '
                                          'Please remove or handle these NaNs before proceeding.')
        assert not np.any(np.isnan(y2)), ('The dataset y2 contains NaN values. '
                                          'Please remove or handle these NaNs before proceeding.')
        # Ensure the correlation coefficient is positive
        assert 0 <= rho <= 1, 'The correlation coefficient must be between 0 and 1.'

        self.y1 = y1
        self.y2 = y2
        self.rho = rho
        self.n_runs = n_runs1
        self.nu = self.n_runs - 1

    def _transform_data(self) -> dict:
        """
        Transform the data for the Stan model.

        :return: Dictionary containing the transformed data.
        """
        return dict(n=self.n_runs, y1=self.y1, y2=self.y2, rho=self.rho)

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
        # Calculate the difference between y1 and y2
        x = self.y1 - self.y2

        # Retrieve posterior predictive samples
        x_rep = self._fit.stan_variable('x_rep')
        y1_rep = self._fit.stan_variable('y1_rep')
        y2_rep = self._fit.stan_variable('y2_rep')
        n_cd, n_samples = x_rep.shape

        # Calculate and print PPC Metrics
        metrics_x = [posterior_predictive_check_metrics(x, x_rep[i], ranks=False) for i in range(n_cd)]
        metrics_y1 = [posterior_predictive_check_metrics(self.y1, y1_rep[i], ranks=False) for i in range(n_cd)]
        metrics_y2 = [posterior_predictive_check_metrics(self.y2, y2_rep[i], ranks=False) for i in range(n_cd)]

        means_x, std_devs_x = calculate_statistics(metrics_x)
        means_y1, std_devs_y1 = calculate_statistics(metrics_y1)
        means_y2, std_devs_y2 = calculate_statistics(metrics_y2)
        print('\nPosterior Predictive Check Metrics')
        print(f'x:\nMeans: {means_x}\nStdDevs: {std_devs_x}\n'
              f'y1:\nMeans: {means_y1}\nStdDevs: {std_devs_y1}\n'
              f'y2:\nMeans: {means_y2}\nStdDevs: {std_devs_y2}\n')

        # Reshape _rep to have dimensions (chain, draw, n_samples)
        n_draws = int(n_cd / self.chains)
        x_rep = x_rep.reshape((self.chains, n_draws, n_samples))
        y1_rep = y1_rep.reshape((self.chains, n_draws, n_samples))
        y2_rep = y2_rep.reshape((self.chains, n_draws, n_samples))

        # Adding posterior predictive samples to InferenceData
        self.inf_data.add_groups(posterior_predictive=dict(x_rep=x_rep, y1_rep=y1_rep, y2_rep=y2_rep))

        # Suppress warning of incorrect dimensions for observed data
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="the default dims 'chain' and 'draw' will be added automatically")
            self.inf_data.add_groups(observed_data=dict(x=x, y1=self.y1, y2=self.y2))

        # Generate the PPC plot
        variables = ['x', 'y1', 'y2']
        plot_posterior_predictive_check(inf_data=self.inf_data, variables=variables, n_draws=n_draws,
                                        show_plt=not save, font_size=font_size, seed=self.seed)

        # Save the plot if requested
        if save:
            self.save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name)
        else:
            plt.tight_layout()

        # Plot Posterior densities in the style of John K. Kruschke’s book.
        az.plot_posterior(self.inf_data)
        plt.show()

    def analyse(self, posterior_predictive_check: bool = True, plot: bool = True, save: bool = True, round_to: int = 4,
                directory_path: str = 'results', file_path: str = 'bayesian_correlated_t_test',
                file_name: Optional[str] = None, **kwargs) -> dict:
        """
        Analyse the results using the Bayesian correlated t-test.

        :param posterior_predictive_check: Whether to do a posterior predictive check. Default is True.
        :param plot: Whether to generate and display posterior distribution plot. Default is True.
        :param save: Whether to save the results and plots to files. Default is True.
        :param round_to: Number of decimal places to round the results to. Default is 4.
        :param directory_path: Path to the parent directory where the files are stored. Default is 'results'.
        :param file_path: Directory path to save the files. Default is 'bayesian_correlated_t_test'.
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

        # Extract the mean and standard deviation of the posterior distribution of mu
        mean = summary.loc['mu', 'mean']
        sd = summary.loc['mu', 'sd']

        # Calculate probabilities for the posterior distribution
        left_prob, rope_prob, right_prob = calculate_probabilities(self.rope, mean, sd, self.nu)

        # Extract samples from the posterior distribution
        samples = self._fit.stan_variable('mu')
        wr = rope_prob * 100 if rope_prob else None

        # Prepare the results
        posterior_probs = dict(
            left_prob=left_prob,  # Probability that the effect is less than the lower bound of the ROPE
            rope_prob=rope_prob,  # Probability that the effect is within the ROPE
            right_prob=right_prob,  # Probability that the effect is greater than the upper bound of the ROPE
        )
        additional = dict(
            samples=samples,  # Posterior samples of the parameter mu
            posterior_df=self.nu,  # Degrees of freedom of the posterior distribution
            posterior_mean=mean,  # Mean of the posterior distribution
            posterior_sd=sd,  # Standard deviation of the posterior distribution
            within_rope=wr,  # Proportion of samples within the ROPE
        )
        parameters = dict(
            rope=self.rope,  # Region of Practical Equivalence (ROPE)
            rho=self.rho,  # Correlation coefficient
            iter_sampling=self.iter_sampling,  # Number of draws from the posterior for each chain
            iter_warmup=self.iter_warmup,  # Number of warm-up (burn-in) samples
            chains=self.chains,  # Number of chains in the MCMC sampling
            sampling_parameters=self.sampling_parameters,  # Additional parameters for the MCMC sampling
            seed=self.seed  # Random seed for reproducibility
        )
        results = dict(
            method='Bayesian correlated t-test',  # Method used for the analysis
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
                               title=r'Posterior Distribution of  $\mu$', show_plt=False, **kwargs)
            # Save the plot if requested
            if save:
                self.save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name)
                plt.show()
            else:
                plt.tight_layout()
                plt.show()
        return results
