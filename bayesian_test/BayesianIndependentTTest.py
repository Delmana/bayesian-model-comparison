"""
Kruschke, J. K. (2013). Bayesian estimation supersedes the t test.
Journal of Experimental Psychology: General, 142(2), 573–603. https://doi.org/10.1037/a0029146

Paper URL: https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf
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


class BayesianIndependentTTest(AbstractBayesian):
    def __init__(self, y1: np.ndarray, y2: np.ndarray, rope: Optional[tuple[float, float]], seed: int = 42):
        """
        Initialize the Bayesian Independent t-Test class.

        :param y1: A 1D array (num_instances,) of first datapoints.
            This array represents the independently generated performance statistics of the first method being compared.
            Each element corresponds to a single instance. The data should be normally distributed.
        :param y2: A 1D array (num_instances,) of second datapoints.
            This array represents the independently generated performance statistics of the second method being compared.
            Each element corresponds to a single instance. The data should be normally distributed.
        :param rope: Region of Practical Equivalence (ROPE) as a tuple (min, max).
            This defines the range of values within which the differences between the two methods are considered
            practically equivalent. If the difference falls within this range, the two methods are deemed to have
            no significant difference in practical terms.
        :param seed: Random seed for reproducibility. Default is 42.
            This parameter ensures that the random processes in the model are consistent across runs, allowing for
            reproducible results.
        """
        super(BayesianIndependentTTest, self).__init__(stan_file='bayesian_independent_t-test.stan', rope=rope,
                                                       seed=seed)

        # Ensure there are no NaN values in the datasets
        assert not np.any(np.isnan(y1)), ('The dataset y1 contains NaN values. '
                                          'Please remove or handle these NaNs before proceeding.')
        assert not np.any(np.isnan(y2)), ('The dataset y2 contains NaN values. '
                                          'Please remove or handle these NaNs before proceeding.')

        # Ensure the number of runs in both datasets are equal
        n_runs1 = y1.shape[0]
        n_runs2 = y2.shape[0]
        assert n_runs1 == n_runs2, (f'The Independent t-Test is a paired test. The number of runs in the first '
                                    f'dataset ({n_runs1}) must match the number of runs in the second dataset '
                                    f'({n_runs2}). Please verify and correct your input data.')

        self.y1 = y1
        self.y2 = y2

    def _transform_data(self) -> dict:
        """
        Transform the data for the Stan model.

        :return: Dictionary containing the transformed data.
        """
        n_runs1 = self.y1.shape[0]
        n_runs2 = self.y2.shape[0]

        return dict(n_runs1=n_runs1,
                    n_runs2=n_runs2,
                    y1=self.y1,
                    y2=self.y2)

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
                directory_path: str = 'results', file_path: str = 'bayesian_independent_t_test',
                file_name: Optional[str] = None, **kwargs) -> dict:
        """
        Analyse the results using the Bayesian independent t-test.

        :param posterior_predictive_check: Whether to do a posterior predictive check. Default is True.
        :param plot: Whether to generate and display plots. Default is True.
        :param save: Whether to save the results and plots to files. Default is True.
        :param round_to: Number of decimal places to round the results to. Default is 4.
        :param directory_path: Path to the parent directory where the files are stored. Default is 'results'.
        :param file_path: Directory path to save the files. Default is 'bayesian_independent_t_test'.
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

        # Extract and analyze the difference_mean parameter
        diff_mean_mean = summary.loc['difference_mean', 'mean']
        diff_mean_sd = summary.loc['difference_mean', 'sd']
        diff_mean_df = len(summary.loc['difference_mean']) - 1
        diff_mean_left_prob, diff_mean_rope_prob, diff_mean_right_prob = calculate_probabilities(self.rope,
                                                                                                 diff_mean_mean,
                                                                                                 diff_mean_sd,
                                                                                                 diff_mean_df)
        diff_mean_posterior_probs = dict(
            left_prob=diff_mean_left_prob,  # Probability that the effect is less than the lower bound of the ROPE
            rope_prob=diff_mean_rope_prob,  # Probability that the effect is within the ROPE
            right_prob=diff_mean_right_prob,  # Probability that the effect is greater than the upper bound of the ROPE
        )
        diff_mean_samples = self._fit.stan_variable('difference_mean')
        diff_mean_wr = diff_mean_rope_prob * 100 if diff_mean_rope_prob else None
        diff_mean_additional = dict(
            samples=diff_mean_samples,  # Posterior samples of the difference_mean parameter
            posterior_df=diff_mean_df,  # Degrees of freedom of the posterior distribution
            posterior_mean=diff_mean_mean,  # Mean of the posterior distribution
            posterior_sd=diff_mean_sd,  # Standard deviation of the posterior distribution
            within_rope=diff_mean_wr,  # Proportion of samples within the ROPE
        )

        # Extract and analyze the difference_sigma parameter
        diff_sigma_mean = summary.loc['difference_sigma', 'mean']
        diff_sigma_sd = summary.loc['difference_sigma', 'sd']
        diff_sigma_df = len(summary.loc['difference_sigma']) - 1
        diff_sigma_left_prob, diff_sigma_rope_prob, diff_sigma_right_prob = calculate_probabilities(self.rope,
                                                                                                    diff_sigma_mean,
                                                                                                    diff_sigma_sd,
                                                                                                    diff_sigma_df)
        diff_sigma_posterior_probs = dict(
            left_prob=diff_sigma_left_prob,  # Probability that the effect is less than the lower bound of the ROPE
            rope_prob=diff_sigma_rope_prob,  # Probability that the effect is within the ROPE
            right_prob=diff_sigma_right_prob,  # Probability that the effect is greater than the upper bound of the ROPE
        )
        diff_sigma_samples = self._fit.stan_variable('difference_sigma')
        diff_sigma_wr = diff_sigma_rope_prob * 100 if diff_sigma_rope_prob else None
        diff_sigma_additional = dict(
            samples=diff_sigma_samples,  # Posterior samples of the difference_sigma parameter
            posterior_df=diff_sigma_df,  # Degrees of freedom of the posterior distribution
            posterior_mean=diff_sigma_mean,  # Mean of the posterior distribution
            posterior_sd=diff_sigma_sd,  # Standard deviation of the posterior distribution
            within_rope=diff_sigma_wr,  # Proportion of samples within the ROPE
        )

        # Extract and analyze the effect_size parameter
        effect_size_mean = summary.loc['effect_size', 'mean']
        effect_size_sd = summary.loc['effect_size', 'sd']
        effect_size_df = len(summary.loc['effect_size']) - 1
        effect_size_left_prob, effect_size_rope_prob, effect_size_right_prob = calculate_probabilities(self.rope,
                                                                                                       effect_size_mean,
                                                                                                       effect_size_sd,
                                                                                                       effect_size_df)
        effect_size_posterior_probs = dict(
            left_prob=effect_size_left_prob,  # Probability that the effect is less than the lower bound of the ROPE
            rope_prob=effect_size_rope_prob,  # Probability that the effect is within the ROPE
            right_prob=effect_size_right_prob,
            # Probability that the effect is greater than the upper bound of the ROPE
        )
        effect_size_samples = self._fit.stan_variable('effect_size')
        effect_size_wr = effect_size_rope_prob * 100 if effect_size_rope_prob else None
        effect_size_additional = dict(
            samples=effect_size_samples,  # Posterior samples of the effect_size parameter
            posterior_df=effect_size_df,  # Degrees of freedom of the posterior distribution
            posterior_mean=effect_size_mean,  # Mean of the posterior distribution
            posterior_sd=effect_size_sd,  # Standard deviation of the posterior distribution
            within_rope=effect_size_wr,  # Proportion of samples within the ROPE
        )

        # Extract and analyze the mu1 parameter
        mu1_mean = summary.loc['mu1', 'mean']
        mu1_sd = summary.loc['mu1', 'sd']
        mu1_df = len(summary.loc['mu1']) - 1
        mu1_left_prob, mu1_rope_prob, mu1_right_prob = calculate_probabilities(self.rope, mu1_mean, mu1_sd, mu1_df)
        mu1_posterior_probs = dict(
            left_prob=mu1_left_prob,  # Probability that mu1 is less than the lower bound of the ROPE
            rope_prob=mu1_rope_prob,  # Probability that mu1 is within the ROPE
            right_prob=mu1_right_prob,  # Probability that mu1 is greater than the upper bound of the ROPE
        )
        mu1_samples = self._fit.stan_variable('mu1')
        mu1_wr = mu1_rope_prob * 100 if mu1_rope_prob else None
        mu1_additional = dict(
            samples=mu1_samples,  # Posterior samples of the mu1 parameter
            posterior_df=mu1_df,  # Degrees of freedom of the posterior distribution
            posterior_mean=mu1_mean,  # Mean of the posterior distribution
            posterior_sd=mu1_sd,  # Standard deviation of the posterior distribution
            within_rope=mu1_wr,  # Proportion of samples within the ROPE
        )

        # Extract and analyze the mu2 parameter
        mu2_mean = summary.loc['mu2', 'mean']
        mu2_sd = summary.loc['mu2', 'sd']
        mu2_df = len(summary.loc['mu2']) - 1
        mu2_left_prob, mu2_rope_prob, mu2_right_prob = calculate_probabilities(self.rope, mu2_mean, mu2_sd, mu2_df)
        mu2_posterior_probs = dict(
            left_prob=mu2_left_prob,  # Probability that mu2 is less than the lower bound of the ROPE
            rope_prob=mu2_rope_prob,  # Probability that mu2 is within the ROPE
            right_prob=mu2_right_prob,  # Probability that mu2 is greater than the upper bound of the ROPE
        )
        mu2_samples = self._fit.stan_variable('mu2')
        mu2_wr = mu2_rope_prob * 100 if mu2_rope_prob else None
        mu2_additional = dict(
            samples=mu2_samples,  # Posterior samples of the mu2 parameter
            posterior_df=mu2_df,  # Degrees of freedom of the posterior distribution
            posterior_mean=mu2_mean,  # Mean of the posterior distribution
            posterior_sd=mu2_sd,  # Standard deviation of the posterior distribution
            within_rope=mu2_wr,  # Proportion of samples within the ROPE
        )

        # Extract and analyze the sigma1 parameter
        sigma1_mean = summary.loc['sigma1', 'mean']
        sigma1_sd = summary.loc['sigma1', 'sd']
        sigma1_df = len(summary.loc['sigma1']) - 1
        sigma1_left_prob, sigma1_rope_prob, sigma1_right_prob = calculate_probabilities(self.rope, sigma1_mean,
                                                                                        sigma1_sd, sigma1_df)
        sigma1_posterior_probs = dict(
            left_prob=sigma1_left_prob,  # Probability that sigma1 is less than the lower bound of the ROPE
            rope_prob=sigma1_rope_prob,  # Probability that sigma1 is within the ROPE
            right_prob=sigma1_right_prob,  # Probability that sigma1 is greater than the upper bound of the ROPE
        )
        sigma1_samples = self._fit.stan_variable('sigma1')
        sigma1_wr = sigma1_rope_prob * 100 if sigma1_rope_prob else None
        sigma1_additional = dict(
            samples=sigma1_samples,  # Posterior samples of the sigma1 parameter
            posterior_df=sigma1_df,  # Degrees of freedom of the posterior distribution
            posterior_mean=sigma1_mean,  # Mean of the posterior distribution
            posterior_sd=sigma1_sd,  # Standard deviation of the posterior distribution
            within_rope=sigma1_wr,  # Proportion of samples within the ROPE
        )

        # Extract and analyze the sigma2 parameter
        sigma2_mean = summary.loc['sigma2', 'mean']
        sigma2_sd = summary.loc['sigma2', 'sd']
        sigma2_df = len(summary.loc['sigma2']) - 1
        sigma2_left_prob, sigma2_rope_prob, sigma2_right_prob = calculate_probabilities(self.rope, sigma2_mean,
                                                                                        sigma2_sd, sigma2_df)
        sigma2_posterior_probs = dict(
            left_prob=sigma2_left_prob,  # Probability that sigma2 is less than the lower bound of the ROPE
            rope_prob=sigma2_rope_prob,  # Probability that sigma2 is within the ROPE
            right_prob=sigma2_right_prob,  # Probability that sigma2 is greater than the upper bound of the ROPE
        )
        sigma2_samples = self._fit.stan_variable('sigma2')
        sigma2_wr = sigma2_rope_prob * 100 if sigma2_rope_prob else None
        sigma2_additional = dict(
            samples=sigma2_samples,  # Posterior samples of the sigma2 parameter
            posterior_df=sigma2_df,  # Degrees of freedom of the posterior distribution
            posterior_mean=sigma2_mean,  # Mean of the posterior distribution
            posterior_sd=sigma2_sd,  # Standard deviation of the posterior distribution
            within_rope=sigma2_wr,  # Proportion of samples within the ROPE
        )

        # Extract and analyze the nu parameter
        nu_mean = summary.loc['nu', 'mean']
        nu_sd = summary.loc['nu', 'sd']
        nu_df = len(summary.loc['nu']) - 1
        nu_left_prob, nu_rope_prob, nu_right_prob = calculate_probabilities(self.rope, nu_mean, nu_sd, nu_df)
        nu_posterior_probs = dict(
            left_prob=nu_left_prob,  # Probability that nu is less than the lower bound of the ROPE
            rope_prob=nu_rope_prob,  # Probability that nu is within the ROPE
            right_prob=nu_right_prob,  # Probability that nu is greater than the upper bound of the ROPE
        )
        nu_samples = self._fit.stan_variable('nu')
        nu_wr = nu_rope_prob * 100 if nu_rope_prob else None
        nu_additional = dict(
            samples=nu_samples,  # Posterior samples of the nu parameter
            posterior_df=nu_df,  # Degrees of freedom of the posterior distribution
            posterior_mean=nu_mean,  # Mean of the posterior distribution
            posterior_sd=nu_sd,  # Standard deviation of the posterior distribution
            within_rope=nu_wr,  # Proportion of samples within the ROPE
        )

        # Compile all parameters into a dictionary
        parameters = dict(
            rope=self.rope,  # Region of Practical Equivalence (ROPE)
            iter_sampling=self.iter_sampling,  # Number of draws from the posterior for each chain
            iter_warmup=self.iter_warmup,  # Number of warm-up (burn-in) samples
            chains=self.chains,  # Number of chains in the MCMC sampling
            sampling_parameters=self.sampling_parameters,  # Additional parameters for the MCMC sampling
            seed=self.seed  # Random seed for reproducibility
        )

        # Create result dictionary
        results = dict(
            method='Bayesian independent t-test',  # Method used for the analysis
            inference_data=self.inf_data,  # arviz InferenceData: Container for inference data storage using xarray.
            parameters=parameters,  # Parameters used in the analysis
            difference_mean=dict(posterior_probabilities=diff_mean_posterior_probs, additional=diff_mean_additional),
            # Posterior probabilities and additional details for the difference_mean parameter
            difference_sigma=dict(posterior_probabilities=diff_sigma_posterior_probs, additional=diff_sigma_additional),
            # Posterior probabilities and additional details for the difference_sigma parameter
            effect_size=dict(posterior_probabilities=effect_size_posterior_probs, additional=effect_size_additional),
            # Posterior probabilities and additional details for the effect_size parameter
            mu1=dict(posterior_probabilities=mu1_posterior_probs, additional=mu1_additional),
            # Posterior probabilities and additional details for the mu1 parameter
            mu2=dict(posterior_probabilities=mu2_posterior_probs, additional=mu2_additional),
            # Posterior probabilities and additional details for the mu2 parameter
            sigma1=dict(posterior_probabilities=sigma1_posterior_probs, additional=sigma1_additional),
            # Posterior probabilities and additional details for the sigma1 parameter
            sigma2=dict(posterior_probabilities=sigma2_posterior_probs, additional=sigma2_additional),
            # Posterior probabilities and additional details for the sigma2 parameter
            nu=dict(posterior_probabilities=nu_posterior_probs, additional=nu_additional)
            # Posterior probabilities and additional details for the nu parameter
        )

        # Save results if requested
        if save:
            self.save_results(results, directory_path=directory_path, file_path=file_path, file_name=file_name)

        # Print the rounded results
        print_result(results, round_to=round_to)

        # Initialize plt_imp_param to False
        plt_imp_param = False
        # Check if 'plt_imp_param' key is present in kwargs
        if 'plt_imp_param' in kwargs:
            plt_imp_param = kwargs['plt_imp_param']
            del kwargs['plt_imp_param']

        # Plot the posterior distributions if requested
        if plot:
            if plt_imp_param:
                fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            else:
                fig, axes = plt.subplots(4, 2, figsize=(12, 12))

            plot_posterior_pdf(data=diff_mean_samples, rope=self.rope, within_rope=diff_mean_wr, mean=diff_mean_mean,
                               round_to=round_to, title=r'Posterior Distribution of $\Delta\mu = \mu_1 - \mu_2$',
                               ax=axes[0, 0], plt_legend=False, plt_x_label=False, plt_y_label=True,
                               show_plt=False, **kwargs)
            plot_posterior_pdf(data=diff_sigma_samples, rope=self.rope, within_rope=diff_sigma_wr, mean=diff_sigma_mean,
                               round_to=round_to,
                               title=r'Posterior Distribution of $\Delta\sigma = \sigma_1 - \sigma_2$',
                               ax=axes[0, 1], plt_legend=False, plt_x_label=False, plt_y_label=False,
                               show_plt=False, **kwargs)
            plot_posterior_pdf(data=effect_size_samples, rope=self.rope, within_rope=effect_size_wr,
                               mean=effect_size_mean, round_to=round_to,
                               title=r'Posterior Distribution of $\delta = (\mu_1 - \mu_2) / \sqrt{\sigma_1^2 + '
                                     r'\sigma_2^2}$', ax=axes[1, 0], plt_legend=False, plt_x_label=False,
                               plt_y_label=True, show_plt=False, **kwargs)
            plot_posterior_pdf(data=mu1_samples, rope=self.rope, within_rope=mu1_wr, mean=mu1_mean, round_to=round_to,
                               title=r'Posterior Distribution of $\mu_1$', ax=axes[1, 1], plt_legend=False,
                               plt_x_label=plt_imp_param, plt_y_label=False, show_plt=False, **kwargs)
            plot_posterior_pdf(data=mu2_samples, rope=self.rope, within_rope=mu2_wr, mean=mu2_mean, round_to=round_to,
                               title=r'Posterior Distribution of $\mu_2$', ax=axes[2, 0], plt_legend=False,
                               plt_x_label=plt_imp_param, plt_y_label=True, show_plt=False, **kwargs)

            if not plt_imp_param:
                plot_posterior_pdf(data=sigma1_samples, rope=self.rope, within_rope=sigma1_wr, mean=sigma1_mean,
                                   round_to=round_to, title=r'Posterior Distribution of $\sigma_1$', ax=axes[2, 1],
                                   plt_legend=False, plt_x_label=False, plt_y_label=False, show_plt=False, **kwargs)
                plot_posterior_pdf(data=sigma2_samples, rope=self.rope, within_rope=sigma2_wr, mean=sigma2_mean,
                                   round_to=round_to, title=r'Posterior Distribution of $\sigma_2$', ax=axes[3, 0],
                                   plt_legend=False, plt_x_label=True, plt_y_label=True, show_plt=False, **kwargs)
                plot_posterior_pdf(data=nu_samples, rope=self.rope, within_rope=nu_wr, mean=nu_mean, round_to=round_to,
                                   title=r'Posterior Distribution of $\nu$', ax=axes[3, 1], plt_legend=True,
                                   plt_x_label=True, plt_y_label=False, show_plt=False, **kwargs)
            else:
                # Remove the last unused plot
                fig.delaxes(axes[2, 1])
                # Plot the legend
                font_size = 12
                if 'font_size' in kwargs:
                    font_size = kwargs['font_size']
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=font_size)

            # Save the plot if requested
            if save:
                self.save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name)
                plt.show()
            else:
                plt.tight_layout()
                plt.show()
        return results
