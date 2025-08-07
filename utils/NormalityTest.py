import os
import datetime
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
from utils.plotting import plot_densities, qqplot_gaussian, plot_normal_distribution


class NormalityTest:
    def __init__(self, data: np.ndarray, algorithm_labels: list[str]):
        """
        Initialize the NormalityTest class.

        :param data: A 2D array with shape (num_instances, num_dataset) containing the dataset.
            Each row corresponds to an instance, and each column represents different datasets.
        :param algorithm_labels: Labels for the algorithms.
            This optional parameter allows to provide custom labels for the algorithms being compared.
            These labels will be used in the analysis the data. If `None`, the algorithms will be labeled numerically.
        """
        self.data = data
        self.algorithm_labels = algorithm_labels
        self._execution_time = datetime.datetime.now()

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

    def _save_plot(self, directory_path: str, file_path: str, file_name: Optional[str], name_suffix: Optional[str],
                   figure: Optional[plt.figure]) -> None:
        """
        Save the plot to a PNG file.

        :param directory_path: Path to the parent directory where the files are stored.
        :param file_path: Directory path to save the file.
        :param file_name: Name of the file. If None, a default name based on the current timestamp will be used.
        :param name_suffix: Suffix to append to the file name.
        :param figure: Figure to be saved. If None, the last created figure will be saved.
        :return: None
        """
        file_path, file_name = self._prepare_file_path(directory_path, file_path, file_name)
        if name_suffix:
            file_name += name_suffix
        plt.tight_layout()
        if figure:
            figure.savefig(f'{file_path}/{file_name}.png', bbox_inches='tight')
        else:
            plt.savefig(f'{file_path}/{file_name}.png', bbox_inches='tight')
        print(f'\nSaving plot to {file_path}/{file_name}')

    def analyse(self, save: bool = True, directory_path: str = 'results', file_path: str = 'normality_test',
                file_name: Optional[str] = None) -> None:
        """
        Analyse the dataset by generating density plots and QQ plots for each feature.

        :param save: Boolean indicating whether to save the plot to a file. Default is True.
        :param directory_path: Path to the parent directory where the files are stored. Default is 'results'.
        :param file_path: String specifying the directory path to save the file. Default is 'normality_test'.
        :param file_name: Optional string specifying the name of the file. If None, a default name will be used.
        :return: None
        """
        n_plots_y = (self.data.shape[1] // 2) + 1
        fig_1, axes_1 = plt.subplots(n_plots_y, 2, figsize=(12, 12))
        fig_2, axes_2 = plt.subplots(n_plots_y, 2, figsize=(12, 12))
        axes_1 = axes_1.flatten()  # Flatten the axes_1 array to simplify indexing
        axes_2 = axes_2.flatten()

        # Plot the densities in the top left corner (first subplot)
        plot_densities(self.data, algorithm_labels=self.algorithm_labels, show_plt=False, ax=axes_1[0])
        plot_densities(self.data, algorithm_labels=self.algorithm_labels, show_plt=False, ax=axes_2[0])

        # Plot QQ plots and normal distribution comparisons in the remaining subplots
        for i in range(self.data.shape[1]):
            x = self.data[:, i]
            label = self.algorithm_labels[i]
            axes_idx = i + 1
            plt_x_label = axes_idx >= len(axes_1) - 2 if self.data.shape[1] % 2 == 1 else axes_idx >= len(
                axes_1) - 3  # Display x-axis label for the bottom row
            plt_y_label = i == 0 or i % 2 == 1  # Display y-axis label for the first column
            plt_title = i < 2  # Display title for the first row
            qqplot_gaussian(data=x, label=label, show_plt=False, plt_x_label=plt_x_label, plt_y_label=plt_y_label,
                            plt_title=plt_title, ax=axes_1[axes_idx])
            plot_normal_distribution(data=x, label=label, show_plt=False, plt_x_label=plt_x_label,
                                     plt_y_label=plt_y_label, plt_title=plt_title, ax=axes_2[axes_idx])

        # Hide any unused subplots
        for j in range(self.data.shape[1] + 1, len(axes_1)):
            fig_1.delaxes(axes_1[j])
            fig_2.delaxes(axes_2[j])

        # Adjust spacing between subplots
        fig_1.tight_layout(pad=1.0)
        fig_2.tight_layout(pad=1.0)

        if save:
            self._save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name,
                            name_suffix='_density.png', figure=fig_1)
            self._save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name,
                            name_suffix='_qqplot.png', figure=fig_2)
        else:
            plt.tight_layout()
        plt.show()

        plt.close(fig_1)
        plt.close(fig_2)
