"""
Corani, G., Benavoli, A., Demšar, J. et al. Statistical comparison of classifiers through Bayesian hierarchical modelling.
Mach Learn 106, 1817–1837 (2017). https://doi.org/10.1007/s10994-017-5641-9

Paper URL: https://link.springer.com/article/10.1007/s10994-017-5641-9
Code URL: https://github.com/BayesianTestsML/tutorial/blob/master/hierarchical/stan/hierarchical-t-test.stan
"""
import warnings
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional
from datetime import datetime
from bayesian_test.AbstractBayesian import AbstractBayesian
from utils.plotting import plot_simplex, plot_posterior_predictive_check
from bayesian_test.utils import print_result, posterior_predictive_check_metrics, calculate_statistics


class BayesianHierarchicalTTest(AbstractBayesian):
    def __init__(self, y1: np.ndarray, y2: np.ndarray, rope: tuple[float, float], rho: float, alpha_lower: float = 0.5,
                 alpha_upper: float = 5.0, beta_lower: float = 0.05, beta_upper: float = 0.15,
                 sigma_upper: float = 1000.0, d0_lower: Optional[float] = None, d0_upper: Optional[float] = None,
                 seed: int = 42):
        """
        Initialize the Bayesian Hierarchical t-Test class.

        :param y1: A 2D array (num_instances, num_datasets) of the first dataset points.
            This array represents the dependently generated performance statistics of the first method being compared.
            Each row corresponds to an instance, and each column represents different datasets. The data is
            expected to be normally distributed.
        :param y2: A 2D array (num_instances, num_datasets) of the second dataset points.
            This array represents the dependently generated performance statistics of the second method being compared.
            Each row corresponds to an instance, and each column represents different datasets.
            The data should be normally distributed.
        :param rope: Region of Practical Equivalence (ROPE) as a tuple (min, max).
            This defines the range of values within which the differences between the two methods are considered
            practically equivalent. If the difference falls within this range, the two methods are deemed to have
            no significant difference in practical terms.
        :param rho: Correlation coefficient (must be 0 < rho < 1).
            This parameter represents the correlation between the two sets of data points, influenced by the
            cross-validation procedure. It is used to estimate the correlation introduced by cross-validation, based on
            the fraction of the data used as the test set (i.e., `n_test / (n_test + n_train)`), as described by
            Nadeau and Bengio (2003).
        :param alpha_lower: Lower bound for the alpha prior (must be > 0). Default is 0.5.
            This parameter defines the minimum value for the prior distribution of the alpha parameter, which influences
            the location parameter of the hierarchical t-distribution model. It affects the shrinkage level in the
            hierarchical model, determining how much group means are pulled towards the overall mean.
        :param alpha_upper: Upper bound for the alpha prior (must be > 0). Default is 5.0.
            This parameter sets the maximum value for the prior distribution of the alpha parameter. Together with
            alpha_lower, it defines the range within which the alpha parameter can vary, impacting the model's
            flexibility to adapt to the data.
        :param beta_lower: Lower bound for the beta prior (must be > 0). Default is 0.05.
            This parameter sets the minimum value for the prior distribution of the beta parameter, which affects the
            scale (or spread) of the data in the hierarchical model. It controls the degree of variability allowed
            within each group in the hierarchical structure.
        :param beta_upper: Upper bound for the beta prior (must be > 0). Default is 0.15.
            This parameter sets the maximum value for the prior distribution of the beta parameter. It works in
            conjunction with beta_lower to determine the variability of the scale parameter, affecting the model's
            flexibility to account for data variability.
        :param sigma_upper: Upper bound for the sigma prior (must be > 0). Default is 1000.
            This parameter sets the maximum value for the prior distribution of the sigma parameter, representing the
            overall standard deviation in the hierarchical model. It influences the model's ability to account for noise
             and outliers in the data.
        :param d0_lower: Lower bound for the delta0 prior (must be > 0).
            This optional parameter sets the minimum value for the prior distribution of the delta0 parameter, which
            relates to the mean difference between the two datasets. If not specified, the model uses default values
            derived from the data.
        :param d0_upper: Upper bound for the delta0 prior (must be > 0).
            This optional parameter sets the maximum value for the prior distribution of the delta0 parameter.
            It, together with d0_lower, defines the range within which the mean difference between the datasets can
            vary. If not specified, default values are used.
        :param seed: Random seed for reproducibility. Default is 42.
            This parameter ensures that the random processes in the model are consistent across runs, allowing for
            reproducible results.
        """
        super(BayesianHierarchicalTTest, self).__init__('bayesian_hierarchical-t-test.stan', rope, seed=seed)
        n_samples1, n_dataset1 = y1.shape
        n_samples2, n_dataset2 = y2.shape

        # Ensure the number of datasets and samples match
        assert n_dataset1 == n_dataset2, (f'The hierarchical T-Test is a paired test. The number of datasets '
                                          f'({n_dataset1}) must match the number of the second dataset ({n_dataset2}).')
        assert n_samples1 == n_samples2, (
            f'The hierarchical T-Test is a paired test. The number of samples in the first dataset ({n_samples1}) '
            f'must match the number of samples in the second dataset ({n_samples2}).')

        # Ensure there are no NaN values in the datasets
        assert not np.any(np.isnan(y1)), ('The dataset y1 contains NaN values. '
                                          'Please remove or handle these NaNs before proceeding.')
        assert not np.any(np.isnan(y2)), ('The dataset y2 contains NaN values. '
                                          'Please remove or handle these NaNs before proceeding.')
        assert rope is not None and not rope == (0, 0), ('The Region of Practical Equivalence (ROPE) must be defined. '
                                                         'Please specify the ROPE to proceed with the analysis.')

        # Ensure the correlation coefficient is positive
        assert 0 <= rho <= 1, 'The correlation coefficient must be between 0 and 1.'

        # Ensure the correctness of model parameters
        assert not alpha_lower <= 0, 'Lower bound for alpha prior must be greater than zero.'
        assert not alpha_upper <= 0, 'Upper bound for alpha prior must be greater than zero.'
        assert not beta_lower <= 0, 'Lower bound for beta prior must be greater than zero.'
        assert not beta_upper <= 0, 'Upper bound for beta prior must be greater than zero'
        assert not sigma_upper <= 0, 'Upper bound for sigma prior must be greater than zero.'
        if d0_lower:
            assert not d0_lower >= 0, 'Lower bound for delta0 prior must be greater than zero.'
        if d0_upper:
            assert not d0_upper >= 0, 'Upper bound for delta0 prior must be greater than zero.'

        self.y1 = np.reshape(y1, (n_samples1, n_dataset1))
        self.y2 = np.reshape(y2, (n_samples2, n_dataset2))
        self.rho = rho
        self.alpha_lower = alpha_lower
        self.alpha_upper = alpha_upper
        self.beta_lower = beta_lower
        self.beta_upper = beta_upper
        self.sigma_upper = sigma_upper
        self.d0_lower = d0_lower
        self.d0_upper = d0_upper

        self._scale_factor = 0
        self.x_observed = None

    def _transform_data(self) -> dict:
        """
        Transform the data for the Stan model.

        :return: Dictionary containing the transformed data.
        """
        x = self.y1 - self.y2

        # Ensure homogenous scale across all datasets
        self._scale_factor = np.mean(np.std(x, axis=1))
        self.rope /= self._scale_factor
        x /= self._scale_factor

        n_datasets, n_samples = x.shape

        # Avoid numerical problems with zero variance
        for sample in x:
            if np.var(sample) == 0:
                sample[:n_samples // 2] = np.random.uniform(self.rope[0], self.rope[0], n_samples // 2)
                sample[n_samples // 2:] = -sample[:n_samples // 2]

        std_within = np.mean(np.std(x, axis=1))
        std_among = np.std(np.mean(x, axis=1)) if n_datasets > 1 else std_within

        # Set lower and upper bounds for delta0
        if self.d0_lower is None:
            self.d0_lower = -np.max(np.abs(x))
        else:
            self.d0_lower /= self._scale_factor

        if self.d0_upper is None:
            self.d0_upper = np.max(np.abs(x))
        else:
            self.d0_upper /= self._scale_factor

        self.x_observed = x
        return dict(
            n_samples=n_samples,
            q=n_datasets,
            x=x,
            delta_low=self.d0_lower,
            delta_high=self.d0_upper,
            std_low=0,
            std_high=std_within * self.sigma_upper,
            std0_low=0,
            std0_high=std_among * self.sigma_upper,
            rho=self.rho,
            alpha_lower=self.alpha_lower,
            alpha_upper=self.alpha_upper,
            beta_lower=self.beta_lower,
            beta_upper=self.beta_upper
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
        # Calculate the difference between y1 and y2
        x = self.x_observed  # n_dataset, n_samples

        # Retrieve posterior predictive samples
        x_rep = self._fit.stan_variable('x_rep')  # n_cd, n_dataset, n_samples
        n_cd, n_dataset, n_instances = x_rep.shape

        n_draws = int(n_cd / self.chains)
        x_rep_dict = dict()
        observed_x_dict = dict()

        print('\nPosterior Predictive Check Metrics')
        # Create InferenceData object for each algorithm
        for d in range(n_dataset):
            observed_x = x[d]  # (n_instances)
            x_rep_data = x_rep[:, d, :]  # shape (n_cd, n_instances)

            # Calculate and print PPC Metrics
            metrics = [posterior_predictive_check_metrics(observed_x, x_rep_data[i], ranks=False) for i in range(n_cd)]
            means, std_devs = calculate_statistics(metrics)
            print(f'Dataset {d}\nMeans: {means}\nStdDevs: {std_devs}\n')

            x_rep_data = x_rep_data.reshape((self.chains, n_draws, n_instances))  # shape (chains, n_draws, n_instances)

            x_rep_dict[f'x_rep_{d}'] = x_rep_data
            observed_x_dict[f'x_{d}'] = observed_x

        # Add groups to InferenceData
        self.inf_data.add_groups(posterior_predictive=x_rep_dict)

        # Suppress warning of incorrect dimensions for observed data
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="the default dims 'chain' and 'draw' will be added automatically")
            self.inf_data.add_groups(observed_data=observed_x_dict)

        # Generate the PPC plot
        variables = [f'Task {d}' for d in range(n_dataset)]
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
                directory_path: str = 'results', file_path: str = 'bayesian_hierarchical_t_test',
                file_name: Optional[str] = None, **kwargs) -> dict:
        """
        Analyse the results using the Bayesian hierarchical t-test.

        :param posterior_predictive_check: Whether to do a posterior predictive check. Default is True.
        :param plot: Whether to generate and display plots. Default is True.
        :param save: Whether to save the results and plots to files. Default is True.
        :param round_to: Number of decimal places to round the results to. Default is 4.
        :param directory_path: Path to the parent directory where the files are stored. Default is 'results'.
        :param file_path: Directory path to save the files. Default is 'bayesian_hierarchical_t_test'.
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
        self._simple_analysis()

        inf_data_posterior = self.inf_data.posterior
        # Remove irrelevant variables
        for key in ['x', 'diag_quad', 'one_over_sigma2', 'nu_minus_one', 'log_lik']:
            if key in inf_data_posterior.data_vars:
                inf_data_posterior = inf_data_posterior.drop_vars(key)

        # Calculate probabilities for each dataset
        delta = inf_data_posterior['delta'].values

        # Collect probabilities for each dataset
        draws, chains, datasets = delta.shape
        probs_per_dataset_list = []
        for i in range(datasets):
            j = delta[:, :, i].flatten()  # Flatten across draws and chains
            probs = {
                'mean delta': np.mean(j) * self._scale_factor,
                'left': np.mean(j < self.rope[0]),
                'rope': np.mean((j >= self.rope[0]) & (j <= self.rope[1])),
                'right': np.mean(j > self.rope[1])
            }
            probs_per_dataset_list.append(probs)
        probs_per_dataset = pd.DataFrame(probs_per_dataset_list)

        # Analyze the posterior distribution of delta parameter
        delta0 = inf_data_posterior['delta0'].values.flatten()
        std0 = inf_data_posterior['std0'].values.flatten()
        nu = inf_data_posterior['nu'].values.flatten()

        cum_left = stats.t.cdf(self.rope[0], df=nu, loc=delta0, scale=std0)
        cum_right = 1 - stats.t.cdf(self.rope[1], df=nu, loc=delta0, scale=std0)
        cum_rope = 1 - cum_left - cum_right

        posterior_distribution = pd.DataFrame({
            'left': cum_left,  # Cumulative probability that the effect is less than the lower bound of the ROPE.
            'rope': cum_rope,  # Cumulative probability that the effect is within the ROPE.
            'right': cum_right  # Cumulative probability that the effect is greater than the upper bound of the ROPE.
        })

        left_wins = (cum_left > cum_right) & (cum_left > cum_rope)
        right_wins = (cum_right > cum_left) & (cum_right > cum_rope)
        rope_wins = ~(left_wins | right_wins)

        prob_left_win = np.mean(left_wins)
        prob_right_win = np.mean(right_wins)
        prob_rope_win = np.mean(rope_wins)

        prob_positive = np.mean(delta0 > 0)
        prob_negative = 1 - prob_positive

        global_sign = pd.Series([prob_negative, prob_positive], index=['negative', 'positive'])

        # left_prob: Probability that the effect is less than the lower bound of the ROPE.
        # rop_prob: Probability that the effect is within the ROPE.
        # right_prob: Probability that the effect is greater than the upper bound of the ROPE.
        global_wins = pd.Series([prob_left_win, prob_rope_win, prob_right_win],
                                index=['left_prob', 'rope_prob', 'right_prob'])

        parameters = dict(
            rho=self.rho,  # Correlation coefficient.
            std_upper=self.sigma_upper,  # Upper bound for sigma prior.
            d0_lower=self.d0_lower,  # Lower bound for delta0 prior.
            d0_upper=self.d0_upper,  # Upper bound for delta0 prior.
            alpha_lower=self.alpha_lower,  # Lower bound for alpha prior.
            alpha_upper=self.alpha_upper,  # Upper bound for alpha prior.
            beta_lower=self.beta_lower,  # Lower bound for beta prior.
            beta_upper=self.beta_upper,  # Upper bound for beta prior.
            rope=self.rope,  # Region of Practical Equivalence (ROPE
            iter_sampling=self.iter_sampling,  # Number of draws from the posterior for each chain
            iter_warmup=self.iter_warmup,  # Number of warm-up (burn-in) samples
            chains=self.chains,  # Number of chains in the MCMC sampling
            sampling_parameters=self.sampling_parameters,  # Additional parameters for the MCMC sampling
            seed=self.seed  # Random seed for reproducibility.
        )

        additional = dict(
            probs_per_dataset=probs_per_dataset,  # DataFrame containing the probabilities for each dataset.
            global_sign=global_sign,
            # Series indicating the global probability of the effect being negative or positive.
            inf_data_posterior=inf_data_posterior  # Inference data posterior.
        )
        results = dict(
            method='Bayesian hierarchical correlated t-test',  # Method used for the analysis
            inference_data=self.inf_data,  # arviz InferenceData: Container for inference data storage using xarray.
            parameters=parameters,  # Parameters used in the analysis.
            posterior_probabilities=global_wins,  # Dataframe containing posterior probabilities from the analysis.
            posterior=posterior_distribution,  # DataFrame containing the posterior distribution from the analysis
            additional=additional  # Additional details from the analysis.
        )

        # Save the results if requested
        if save:
            self.save_results(results, directory_path=directory_path, file_path=file_path, file_name=file_name)

        # Print the rounded results
        rounded_results = print_result(results, round_to=round_to)
        print(f'\nPosterior:\n{rounded_results["posterior"].mean().round(round_to)}')
        print(f'\nPosterior Probabilities:\n{rounded_results["posterior_probabilities"]}')
        print(f'\nProbabilities per Dataset:\n{rounded_results["additional"]["probs_per_dataset"]}')

        # Plot the posterior distribution if requested
        if plot:
            plot_simplex(posterior=posterior_distribution, posterior_probabilities=global_wins, round_to=round_to,
                         show_plt=False, **kwargs)
            # Save the plot if requested
            if save:
                self.save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name)
                plt.show()
            else:
                plt.tight_layout()
                plt.show()
        return results
