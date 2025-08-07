import numpy as np
import pandas as pd
from numpy import ndarray, dtype


class DataLoader:
    def __init__(self, dataset: str):
        """
        Initialize the DataLoader with the specified dataset.

        :param dataset: The name of the dataset to load. Must be either 'data_kcv_example' or 'data_blum_2015'.
        """
        assert dataset in ['data_blum_2015',
                           'data_gh_2008',
                           'data_kcv_example'], 'dataset must be "data_kcv_example", "data_gh_2008" or "data_blum_2015"'
        self.dataset = dataset
        self.df = self._load_dataset()

    def _load_dataset(self):
        """
        Load the dataset from a CSV file.

        :return: A pandas DataFrame containing the loaded dataset.
        """
        return pd.read_csv(f'examples/data/{self.dataset}.csv', sep=';')

    def get_samples(self) -> tuple[ndarray, pd.Index]:
        """
        Extract the data and column names from the DataFrame.

        :return: A tuple containing a numpy ndarray of the data values and a list of strings representing the column names.
        """
        return self.df.values, self.df.columns

    def extract_samples_by_dataset(self, algorithms: list[str], dataset_id: int) -> list[np.array]:
        """
        Extract samples for the specified algorithms from a specific dataset.

        :param algorithms: A list of algorithm names to extract samples for.
        :param dataset_id: The dataset identifier to filter samples by.
        :return: A list of numpy arrays, each containing the samples for a specified algorithm.
        """
        assert self.dataset == 'data_kcv_example', 'dataset must be "data_kcv_example"'
        filtered_df = self.df[self.df['DB'] == dataset_id]
        samples = [filtered_df[algorithm].to_numpy().astype(float) for algorithm in algorithms]
        return samples

    def extract_summarized_samples_by_dataset(self, algorithms: list[str], dataset_id: int) -> list[np.array]:
        """
        Extract summarized samples for the specified algorithms from a specific dataset by averaging over repetitions.

        :param algorithms: A list of algorithm names to extract summarized samples for.
        :param dataset_id: The dataset identifier to filter samples by.
        :return: A list of numpy arrays, each containing the summarized samples for a specified algorithm.
        """
        assert self.dataset == 'data_kcv_example', 'dataset must be "data_kcv_example"'
        summarized_data = self.df.groupby(['DB', 'Rep']).mean().reset_index()
        samples = [summarized_data[summarized_data['DB'] == dataset_id][algorithm].to_numpy() for algorithm in
                   algorithms]
        return samples

    def extract_all_samples_by_algorithm(self, algorithms: list[str]) -> list[np.array]:
        """
        Extract all samples for the specified algorithms across all datasets.

        :param algorithms: A list of algorithm names to extract samples for.
        :return: A list of numpy arrays, each containing the samples for a specified algorithm across all datasets.
        """
        assert self.dataset == 'data_kcv_example', 'dataset must be "data_kcv_example"'
        datasets = self.df['DB'].unique()
        samples = []
        for algorithm in algorithms:
            data = np.array([self.df[algorithm][self.df['DB'] == db].to_numpy().astype(float) for db in datasets])
            samples.append(data)
        return samples

    def extract_sample_by_size(self, size: int) -> tuple[ndarray[any, dtype[any]], pd.Index]:
        """
        Extract samples for all algorithms from the dataset filtered by the specified size.

        :param size: The size to filter the dataset by.
        :return: A tuple containing a numpy array of the samples and a list of algorithm labels.
        """
        assert self.dataset == 'data_blum_2015', 'dataset must be "data_blum_2015"'
        filtered_df = self.df[self.df['Size'] == size]
        filtered_df = filtered_df.drop(columns=['Size', 'Radius'])
        algorithm_labels = filtered_df.columns
        sample = filtered_df.to_numpy().astype(float)
        return sample, algorithm_labels
