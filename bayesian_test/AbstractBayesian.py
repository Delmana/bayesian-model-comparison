import os
import pickle
import warnings
import cmdstanpy
import numpy as np
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from datetime import datetime
from abc import ABC, abstractmethod

# Define constants for folder paths
STAN_FILES_FOLDER = 'stan_files'

# Abstract base class for Bayesian analysis
class AbstractBayesian(ABC):
    def __init__(self, stan_file: str, rope: Optional[tuple[float, float]], seed: int = 42):
        """
        Initialize the AbstractBayesian class.

        :param stan_file: Path to the Stan file.
        :param rope: Optional; Region of Practical Equivalence (ROPE). Default is None.
        :param seed: Random seed for reproducibility. Default is 42.
        """
        np.random.seed(seed)

        self._stan_file = stan_file
        if rope is not None:
            if rope[1] < rope[0]:
                warnings.warn(
                    'The ROPE parameter should contain the ordered limits (min, max). '
                    'The provided values are not ordered. They will be swapped to proceed with the analysis.'
                )
                self.rope = rope[1], rope[0]
            else:
                self.rope = rope
        else:
            self.rope = rope
        self._posterior_model = None
        self._fit = None
        self._data = dict()
        self._execution_time = datetime.now()
        self.inf_data = az.InferenceData()

        self.iter_sampling = -1
        self.iter_warmup = -1
        self.chains = -1
        self.sampling_parameters = None
        self.seed = seed

    @abstractmethod
    def _transform_data(self) -> dict:
        """
        Transform the data for the Stan model. Must be implemented by subclasses.

        :return: A dictionary containing the transformed data ready for input to the Stan model.
        """
        pass

    @abstractmethod
    def _posterior_predictive_check(self, directory_path: str, file_path: str,
                                    file_name: str = 'posterior_predictive_check', font_size: int = 12,
                                    save: bool = True) -> None:
        """
        This function performs posterior predictive checks and generates plots comparing the observed data
        to the posterior predictive distributions.
        https://mc-stan.org/docs/2_24/stan-users-guide/simulating-from-the-posterior-predictive-distribution.html

        :param directory_path: Path to the parent directory where the files are stored.
        :param file_path: Path where the plot should be saved.
        :param file_name: Name of the file to save the plot. Default is 'posterior_predictive_check'.
        :param font_size: Font size for the plot text elements. Default is 12
        :param save: Whether to save the plot to file. Default is True.
        :return: None
        """
        pass

    @abstractmethod
    def analyse(self, posterior_predictive_check: bool = True, plot: bool = True, save: bool = True, round_to: int = 4,
                directory_path: str = 'results', file_path: str = 'bayesian_analysis', file_name: Optional[str] = None,
                **kwargs) -> dict:
        """
        Analyse the results. This method should be implemented in subclasses to perform specific analysis tasks.

        :param posterior_predictive_check: Whether to do a posterior predictive check. Default is True.
        :param plot: Whether to generate and display plots. Default is True.
        :param save: Whether to save the results and plots to files. Default is True.
        :param round_to: Number of decimal places to round the results to. Default is 4.
        :param directory_path: Path to the parent directory where the files are stored. Default is 'results'.
        :param file_path: Directory path to save the files. Default is 'bayesian_analysis'.
        :param file_name: Name of the file to save the results and plots. If None, a default name based on the current timestamp will be used.
        :param kwargs: Additional keyword arguments for customized analysis and plotting.
        :return: A dictionary containing the analysis results, including posterior probabilities and additional details.
        """
        pass

    def _prepare_file_path(self, directory_path: str, file_path: str, file_name: Optional[str]) -> tuple[str, str]:
        """
        Prepare the file path and name for saving results.

        :param directory_path: Path to the parent directory where the files are stored.
        :param file_path: Directory path to save the file.
        :param file_name: Name of the file. If None, a default name based on the current timestamp will be used.
        :return: Tuple containing the prepared file path and name.
        """
        file_path = f'{directory_path}/{file_path}/{self._execution_time.strftime("%y%m%d_%H%M")}'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = f'{self._execution_time.strftime("%y%m%d_%H%M")}' if file_name is None else file_name
        return file_path, file_name

    def save_results(self, results: dict, directory_path: str, file_path: str, file_name: Optional[str]) -> None:
        """
        Save the results to a pickle file.

        :param results: Dictionary containing the results.
        :param directory_path: Path to the parent directory where the files are stored.
        :param file_path: Directory path to save the file.
        :param file_name: Name of the file. If None, a default name based on the current timestamp will be used.
        :return: None
        """
        file_path, file_name = self._prepare_file_path(directory_path, file_path, file_name)
        with open(f'{file_path}/{file_name}.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'\nSaving results to {file_path}/{file_name}')

    def save_plot(self, directory_path: str, file_path: str, file_name: Optional[str]) -> None:
        """
        Save the plot to a PNG file.

        :param directory_path: Path to the parent directory where the files are stored.
        :param file_path: Directory path to save the file.
        :param file_name: Name of the file. If None, a default name based on the current timestamp will be used.
        :return: None
        """
        file_path, file_name = self._prepare_file_path(directory_path, file_path, file_name)
        plt.tight_layout()
        plt.savefig(f'{file_path}/{file_name}.png', bbox_inches='tight')
        print(f'\nSaving plot to {file_path}/{file_name}')

    def _simple_analysis(self, round_to: int = 4) -> pd.DataFrame:
        """
        Perform a simple analysis and print the summary.

        :param round_to: Number of decimal places to round to.
        :return: Summary of the analysis.
        """
        summary = az.summary(self.inf_data, round_to=round_to)
        print(summary)
        return summary

    def fit(self, iter_sampling: int = 50000, iter_warmup: int = 1000, chains: int = 4, **kwargs) -> None:
        """
        Fit the Stan model using MCMC sampling.

        :param iter_sampling: Number of draws from the posterior for each chain. Default is 50000
        :param iter_warmup: Number of warmup iterations for each chain. Default is 1000
        :param chains: Number of sampler chains, must be a positive integer. Default is 4
        :param kwargs: Additional keyword arguments for the sampling. See https://mc-stan.org/cmdstanpy/api.html#cmdstanmodel
        :return: None
        """
        assert iter_sampling > 0, 'Number of sampling iterations must be positive!'
        assert iter_warmup > 0, 'Number of warmup iterations must be positive!'
        assert chains > 0, 'Number of sampler chains must be positive!'

        self.iter_sampling = iter_sampling
        self.iter_warmup = iter_warmup
        self.chains = chains
        if kwargs:
            self.sampling_parameters = kwargs
        # Adjust data for the Stan model
        self._data = self._transform_data()

        # Load the Stan program code
        program_code = f'{STAN_FILES_FOLDER}/{self._stan_file}'

        # Build and compile the Stan model
        self._posterior_model: cmdstanpy.CmdStanModel = cmdstanpy.CmdStanModel(stan_file=program_code)

        # Fit the model using MCMC sampling
        self._fit: cmdstanpy.CmdStanMCMC = self._posterior_model.sample(data=self._data,
                                                                        iter_sampling=self.iter_sampling,
                                                                        iter_warmup=self.iter_warmup,
                                                                        chains=self.chains,
                                                                        seed=self.seed,
                                                                        show_console=False,
                                                                        **kwargs)

        # Convert the Stan fit to an Arviz InferenceData object
        self.inf_data: az.InferenceData = az.from_cmdstanpy(posterior=self._fit)

        # Run convergence checks
        print(f'{datetime.now().time().strftime("%H:%M:%S")} - '
              f'INFO: Running convergence diagnose on the inference data.')
        print(self._fit.diagnose())
