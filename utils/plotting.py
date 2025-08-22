import datetime
import os

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional
from matplotlib.lines import Line2D

# Set the style for seaborn plots
sns.set(style='whitegrid', color_codes=True)


def plot_posterior_pdf(data: np.ndarray, rope: Optional[tuple[float, float]], within_rope: Optional[float],
                       mean: Optional[float], hpdi_prob: float = 0.99, plot_type: str = 'kde', n_bins: int = 50,
                       plt_rope: bool = True, plt_rope_text: bool = True, plt_within_rope: bool = True,
                       plt_mean: bool = False, plt_mean_text: bool = False, plt_hpdi: bool = True,
                       plt_hpdi_text: bool = True, plt_samples: bool = True, title: str = 'Posterior PDF Plot',
                       plt_title: bool = True, plt_legend: bool = True, plt_x_label: bool = True,
                       plt_y_label: bool = True, show_plt: bool = False, round_to: int = 4, alpha: float = 0.8,
                       font_size: float = 12, ax: plt.Axes = None) -> None:
    """
    Plot the posterior Probability Density Function (PDF) of the samples.

    :param data: 1D array (num_instances,) of posterior samples.
    :param rope: Optional; Tuple indicating the Region of Practical Equivalence (ROPE). Default is None.
    :param within_rope: Optional; Percentage of samples within the ROPE. Default is None.
    :param mean: Optional; Mean of the samples. Default is None.
    :param hpdi_prob: Highest Posterior Density Interval (HPDI) probability. Default is 0.99.
    :param plot_type: Type of plot ('kde' or 'hist'). Default is 'kde'.
    :param n_bins: Number of bins for histogram if plot_type is 'hist'. Default is 50.
    :param plt_rope: Whether to plot the ROPE. Default is True.
    :param plt_rope_text: Whether to display text for the ROPE. Default is True.
    :param plt_within_rope: Whether to display the percentage of samples within the ROPE. Default is True.
    :param plt_mean: Whether to plot the mean of the samples. Default is False.
    :param plt_mean_text: Whether to display text for the mean. Default is False.
    :param plt_hpdi: Whether to plot the HPDI. Default is True.
    :param plt_hpdi_text: Whether to display text for the HPDI. Default is True.
    :param plt_samples: Whether to plot the sample points. Default is True.
    :param title: Title of the plot. Default is 'Posterior PDF Plot'.
    :param plt_title: Whether to display the title. Default is True.
    :param plt_legend: Whether to display the legend. Default is True.
    :param plt_x_label: Whether to display the x-axis label. Default is True.
    :param plt_y_label: Whether to display the y-axis label. Default is True.
    :param show_plt: Whether to display the plot. Default is False.    :param round_to: Number of decimal places to round text annotations. Default is 4.
    :param alpha: Transparency level for plot elements. Default is 0.8.
    :param font_size: Font size for text annotations. Default is 12.
    :param ax: Axis to plot on, if None a new axis will be created. Default is None.
    :return: None
    """
    assert plot_type in ['kde', 'hist'], 'point_type must be either "kde" or "hist"'

    if ax is None:
        rope_offset = 1.04
        mean_offset = 1.08
        title_offset = 1.12
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        rope_offset = 1.07
        mean_offset = 1.12
        title_offset = 1.14

    if plot_type == 'kde':
        sns.kdeplot(data, label='Posterior PDF', fill=True, ax=ax)
    else:
        sns.histplot(data, label='Posterior PDF', ax=ax, bins=n_bins, alpha=alpha)

    if plt_samples:
        ax.scatter(data, np.zeros_like(data), color='b', alpha=alpha, label='Samples')

    if plt_hpdi:
        hpdi_bounds = az.hdi(data, hpdi_prob)
        ax.axvline(x=hpdi_bounds[0], color='orange', linestyle='--', label=f'{int(hpdi_prob * 100)}% HPDI Lower')
        ax.axvline(x=hpdi_bounds[1], color='orange', linestyle='--', label=f'{int(hpdi_prob * 100)}% HPDI Upper')
        if plt_hpdi_text:
            ax.text(hpdi_bounds[0], ax.get_ylim()[1] * 1.06, f'{hpdi_bounds[0]:.{round_to}f}',
                    horizontalalignment='center', color='orange', fontsize=font_size - 2)
            ax.text(hpdi_bounds[1], ax.get_ylim()[1] * 1.06, f'{hpdi_bounds[1]:.{round_to}f}',
                    horizontalalignment='center', color='orange', fontsize=font_size - 2)

    if plt_rope and rope is not None:
        ax.axvline(x=rope[0], color='g', linestyle='--', label='ROPE Lower')
        ax.axvline(x=rope[1], color='g', linestyle='--', label='ROPE Upper')
        ax.fill_betweenx([0, ax.get_ylim()[1]], rope[0], rope[1], color='g', alpha=alpha - 0.3)

    if (plt_rope_text or plt_within_rope) and rope is not None:
        text = ''
        if plt_rope_text:
            text = f'({rope[0]}, {rope[1]})'
        if within_rope is not None and plt_within_rope:
            if text:
                text += f': {within_rope:.2f}% in ROPE'
            else:
                text = f'{within_rope:.2f}% in ROPE'
        ax.text(rope[0], ax.get_ylim()[1] * rope_offset, text, horizontalalignment='center', color='g',
                fontsize=font_size - 2)

    if plt_mean and mean is not None:
        ax.axvline(x=mean, color='r', linestyle='--', label='Mean')
        if plt_mean_text:
            ax.text(mean, ax.get_ylim()[1] * mean_offset, f'{mean:.{round_to}f}',
                    horizontalalignment='center', color='r', fontsize=font_size - 2)

    if plt_x_label:
        ax.set_xlabel('Sample', fontsize=font_size)
    else:
        ax.set(xlabel='')
    if plt_y_label:
        ax.set_ylabel('Density', fontsize=font_size)
    else:
        ax.set(ylabel='')

    if plt_title:
        ax.set_title(title, fontsize=font_size + 2, y=title_offset)
    if plt_legend:
        ax.legend(fontsize=font_size)

    if show_plt:
        plt.tight_layout()
        plt.show()


def _bary2cart(simplex_coords: np.ndarray, bary_coords: np.ndarray) -> np.ndarray:
    """
    Convert from barycentric coordinates to Cartesian coordinates.

    :param simplex_coords: Coordinates of the simplex vertices.
    :param bary_coords: Barycentric coordinates to convert.
    :return: A numpy array containing the cartesian coordinates.
    """
    return np.dot(bary_coords, simplex_coords)


def plot_simplex(posterior: pd.DataFrame, posterior_probabilities: dict, algo_label1: str = 'Alg. A',
                 algo_label2: str = 'Alg. B', plt_points: bool = True, plot_type: str = 'scatter',
                 point_size: int = 10, plt_density: bool = True, posterior_label: bool = True,
                 title: str = 'Simplex Plot of Posterior Probabilities', plt_title: bool = True, show_plt: bool = False,
                 round_to: int = 4, palette: tuple = ('steelblue', 'cornflowerblue', 'royalblue'), font_size: int = 12,
                 alpha: float = 0.8) -> None:
    """
    Plot the posterior probabilities in a simplex plot.

    :param posterior: DataFrame containing posterior probabilities for 'left', 'rope', and 'right'.
    :param posterior_probabilities: Dictionary with probabilities for 'left', 'rope', and 'right'.
    :param algo_label1: Label for the first algorithm. Default is 'Alg. A'.
    :param algo_label2: Label for the second algorithm. Default is 'Alg. B'.
    :param plt_points: Whether to plot individual points. Default is True.
    :param plot_type: Type of plot ('scatter' or 'hexbin'). Default is 'scatter'.
    :param point_size: Size of points in the scatter plot. Default is 10.
    :param plt_density: Whether to plot density (ignored if plot_type is 'hexbin'). Default is True.
    :param posterior_label: Whether to label the posterior probabilities. Default is True.
    :param title: Title of the plot. Default is 'Simplex Plot of Posterior Probabilities'.
    :param plt_title: Whether to display the title. Default is True.
    :param show_plt: Whether to display the plot. Default is False.    :param round_to: Number of decimal places to round text annotations. Default is 4.
    :param palette: Tuple of colors for the plot. Default is ('steelblue', 'cornflowerblue', 'royalblue').
    :param font_size: Size of the front of the plot. Default is 12.
    :param alpha: Transparency level for plot elements. Default is 0.8.
    :return: None
    """
    assert plot_type in ['scatter', 'hexbin'], 'point_type must be either "scatter" or "hexbin"'
    plt_density = False if plot_type == 'hexbin' else plt_density

    post_sample = posterior[['left', 'rope', 'right']]

    # Normalize the post_sample to ensure each row sums to 1
    post_sample = post_sample.div(post_sample.sum(axis=1), axis=0)
    # Filter out rows where any coordinate is negative or greater than 1
    post_sample = post_sample[(post_sample >= 0).all(axis=1) & (post_sample <= 1).all(axis=1)]

    # Get the winner for the color of the points
    aux = post_sample.idxmax(axis=1)
    colors = aux.map({'left': palette[2], 'rope': palette[1], 'right': palette[0]})

    # Coordinates of the edges of the Simplex
    simplex_coords = np.array([[2, 0], [1, 1], [0, 0]])
    # Convert from barycentric coords to cartesian and add the winner
    cart_coords = _bary2cart(simplex_coords, post_sample.to_numpy())
    points = pd.DataFrame(cart_coords, columns=['x', 'y'])
    points['color'] = colors

    p_left = posterior_probabilities['left_prob']
    p_rope = posterior_probabilities['rope_prob']
    p_right = posterior_probabilities['right_prob']

    # Create the plot, layer by layer
    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    # Optionally, add the points
    if plot_type == 'scatter':
        if plt_points:
            sns.scatterplot(x='x', y='y', hue='color', data=points, palette=palette, alpha=alpha, s=point_size, ax=ax,
                            legend=None)
        if plt_density:
            sns.kdeplot(x=points['x'], y=points['y'], fill=True, alpha=alpha, ax=ax, cmap='Blues_r')
    else:
        plt.hexbin(points['x'], points['y'], mincnt=1, cmap=plt.cm.Blues_r)

    # Add the triangle and annotations
    ax.add_line(Line2D([2, 1], [0, 1], color='dimgrey'))
    ax.add_line(Line2D([1, 0], [1, 0], color='dimgrey'))
    ax.add_line(Line2D([0, 2], [0, 0], color='dimgrey'))
    ax.add_line(Line2D([1, 0.5], [0.3333333, 0.5], color='dimgrey', linestyle='--'))
    ax.add_line(Line2D([1, 1.5], [0.3333333, 0.5], color='dimgrey', linestyle='--'))
    ax.add_line(Line2D([1, 1], [0.3333333, 0], color='dimgrey', linestyle='--'))

    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])

    if posterior_label:
        ax.text(0, -0.05, f'P({algo_label1} Win)= {p_right:.{round_to}f}', horizontalalignment='center',
                size=font_size)
        ax.text(2, -0.05, f'P({algo_label2} Win)= {p_left:.{round_to}f}', horizontalalignment='center', size=font_size)
        ax.text(1, 1.05, f'P(Rope Win)= {p_rope:.{round_to}f}', horizontalalignment='center', size=font_size)
    else:
        ax.text(0, -0.05, algo_label1, horizontalalignment='center', size=font_size)
        ax.text(2, -0.05, algo_label2, horizontalalignment='center', size=font_size)
        ax.text(1, 1.05, 'Rope', horizontalalignment='center', size=font_size)

    ax.axis('off')
    if plt_title:
        plt.title(title, fontsize=font_size + 2)
    if show_plt:
        plt.tight_layout()
        plt.show()


def plot_histogram(posterior: pd.DataFrame, algo_label1: str = 'Alg. A', algo_label2: str = 'Alg. B', n_bins: int = 50,
                   posterior_label: bool = True, title: str = 'Histogram of Posterior Probabilities',
                   plt_title: bool = True, show_plt: bool = False, round_to: int = 4, font_size: int = 12,
                   alpha: float = 0.8) -> None:
    """
    Plot a histogram of the posterior probabilities.

    :param posterior: DataFrame containing posterior probabilities.
    :param algo_label1: Label for the first algorithm. Default is 'Alg. A'.
    :param algo_label2: Label for the second algorithm. Default is 'Alg. B'.
    :param n_bins: Number of bins for the histogram. Default is 50.
    :param posterior_label: Whether to label the posterior probabilities. Default is True.
    :param title: Title of the plot. Default is 'Histogram of Posterior Probabilities'.
    :param plt_title: Whether to display the title. Default is True.
    :param show_plt: Whether to display the plot. Default is False.    :param round_to: Number of decimal places to round text annotations. Default is 4.
    :param font_size: Size of the front of the plot. Default is 12.
    :param alpha: Transparency level for plot elements. Default is 0.8.
    :return: None
    """
    points = posterior[['left', 'right']].to_numpy()[:, 0]

    p_left = (np.sum(points < 0.5) + 0.5 * np.sum(points == 0.5)) / len(points)
    p_right = 1 - p_left

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    sns.histplot(points, bins=n_bins, alpha=alpha)
    ax.set_xlim(0, 1)

    # Adjust y-limits to make space for the text
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max)
    if posterior_label:
        # Add text below the x-axis using transform=ax.transAxes for relative positioning
        ax.text(0, -0.1, f'P({algo_label1} Win)= {p_left:.{round_to}f}', horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes, size=font_size)
        ax.text(1, -0.1, f'P({algo_label2} Win)= {p_right:.{round_to}f}', horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes, size=font_size)

    ax.axvline(x=0.5, color='r', linestyle='--')

    if plt_title:
        plt.title(title, fontsize=font_size + 2)
    if show_plt:
        plt.tight_layout()
        plt.show()


def plot_boxplot(data: pd.DataFrame, title: str = 'Boxplot of Posterior Weights',
                 x_label: str = 'Algorithm', y_label: str = 'Probability', plt_title: bool = True,
                 show_plt: bool = False, font_size: int = 12, alpha: float = 1.0) -> None:
    """
    Plot a boxplot of the posterior weights.F

    :param data: DataFrame containing posterior weights, where each column represents a different algorithm and each row represents an instance.
    :param title: Title of the plot. Default is 'Boxplot of Posterior Weights'.
    :param x_label: Label of the x-Axis. Default is 'Algorithm'.
    :param y_label: Label of y-Axis. Default is 'Probability'.
    :param plt_title: Whether to display the title. Default is True.
    :param show_plt: Whether to display the plot. Default is False.    :param font_size: Size of the front of the plot. Default is 12.
    :param alpha: Transparency level for plot elements. Default is 1.0.
    :return: None
    """
    sns.boxplot(data, saturation=alpha).set(xlabel=x_label, ylabel=y_label)
    if plt_title:
        plt.title(title, fontsize=font_size + 2)
    if show_plt:
        plt.tight_layout()
        plt.show()


def plot_densities(data: np.ndarray, algorithm_labels: list[str], show_plt: bool = False, alpha: float = 0.8,
                   ax: Optional[plt.Axes] = None) -> None:
    """
    Plot the density (Kernel Density Estimation) for each column in the data array.

    :param data: 2D array (num_instances, num_algorithms) where each column represents data for a different algorithm.
    :param algorithm_labels: List of labels for each algorithm.
    :param show_plt: Whether to display the plot. Default is False.    :param alpha: Transparency level for the plot lines. Default is 0.8.
    :param ax: Axis to plot on, if None a new axis will be created. Default is None.
    :return: None
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for i in range(data.shape[1]):
        sns.kdeplot(data[:, i], label=algorithm_labels[i], alpha=alpha, ax=ax)
    ax.legend()
    ax.set_title('Kernel Density Estimation')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    plt.tight_layout()

    if show_plt:
        plt.show()


def plot_normal_distribution(data: np.ndarray, label: str, plt_title: bool = True, show_plt: bool = False,
                             plt_legend: bool = True, plt_x_label: bool = True, plt_y_label: bool = True, font_size=12,
                             alpha: float = 0.6, ax: plt.Axes = None) -> None:
    """
    Plot the kernel density estimation for each column in the data array and compare it to a normal distribution.

    :param data: 1D array (num_instances,) of data points to check.
    :param label: String label for the data points, used in the legend.
    :param plt_title: Whether to display the plot title. Default is True.
    :param show_plt: Whether to display the plot. Default is False.    :param plt_legend: Whether to display the legend. Default is True.
    :param plt_x_label: Whether to display the x-axis label. Default is True.
    :param plt_y_label: Whether to display the y-axis label. Default is True.
    :param font_size: Font size for the legend text. Default is 12.
    :param alpha: Transparency level for the plot points. Default is 0.6.
    :param ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created. Default is None.
    :return: None
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # KDE plot for the actual data
    sns.kdeplot(data, label=label, alpha=alpha, ax=ax, fill=True)

    # Fit a normal distribution to the data
    mean, std = stats.norm.fit(data)

    # Generate random samples from the fitted normal distribution
    rvs_norm = stats.norm.rvs(mean, std, size=10000)

    # KDE plot for the normal distribution
    sns.kdeplot(rvs_norm, label='Normal Distribution', alpha=1, color='r', ax=ax)

    if plt_x_label:
        ax.set_xlabel('Value')
    if plt_y_label:
        ax.set_ylabel('Density')
    if plt_legend:
        ax.legend(fontsize=font_size)

    if plt_title:
        ax.set_title('Kernel Density Estimation')

    if show_plt:
        plt.tight_layout()
        plt.show()


def qqplot_gaussian(data: np.ndarray, label: str, plt_title: bool = True, show_plt: bool = False,
                    plt_legend: bool = True, plt_x_label: bool = True, plt_y_label: bool = True, font_size=12,
                    alpha: float = 0.8, ax: plt.Axes = None) -> None:
    """
    Gaussian distribution quantile-quantile plot.

    This function creates a quantile-quantile plot to assess the goodness of fit of a Gaussian distribution to a given sample.

    :param data: 1D array (num_instances,) of data points to check.
    :param label: String label for the data points, used in the legend.
    :param plt_title: Whether to display the plot title. Default is True.
    :param show_plt: Whether to display the plot. Default is False.    :param plt_legend: Whether to display the legend. Default is True.
    :param plt_x_label: Whether to display the x-axis label. Default is True.
    :param plt_y_label: Whether to display the y-axis label. Default is True.
    :param font_size: Font size for the legend text. Default is 12.
    :param alpha: Transparency level for the plot points. Default is 0.8.
    :param ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created. Default is None.
    :return: None
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Generate Q-Q plot
    res = stats.probplot(data, plot=ax, dist='norm', fit=True)

    # Customize labels, legend, and title
    if not plt_x_label:
        ax.set_xlabel('')

    if not plt_y_label:
        ax.set_ylabel('')

    if plt_legend:
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [plt.Line2D([0], [0], marker='o', color='b', markerfacecolor='b', markersize=8,
                                                label=label)], fontsize=font_size)
    if plt_title:
        ax.set_title('Gaussian Q-Q Plot', fontsize=font_size)
    else:
        ax.set_title('')

    # Customize transparency
    for line in ax.get_lines():
        line.set_alpha(alpha)

    if show_plt:
        plt.tight_layout()
        plt.show()


def plot_posterior_predictive_check(inf_data: az.InferenceData, variables: list[str], n_draws: int,
                                    show_plt: bool = False, font_size: int = 12, seed: int = 42):
    """
    Generate posterior predictive checks for given variables using Arviz.

    :param inf_data: Inference data containing the posterior predictive samples and observed data.
    :param variables: List of variables for which to generate the plots.
    :param n_draws: Number of posterior predictive draws to use.
    :param show_plt: Whether to display the plot. Default is False.    :param font_size: Font size for plot text elements. Default is 12.
    :param seed: Random seed for reproducibility. Default is 42.
    :return: None
    """
    # Create a dictionary to pair observed data with posterior predictive data
    data_pairs = {key: value for key, value in zip(inf_data.observed_data.keys(), inf_data.posterior_predictive.keys())}

    # Limit the number of posterior predictive samples to a maximum of 100
    n_pp_samples = min(100, n_draws)

    # Create subplots: 3 rows for different kinds of plots and n_var columns for each variable
    n_var = len(variables)
    fig, axes = plt.subplots(3, n_var, figsize=(5 * n_var, 10), squeeze=False)

    # KDE plot for each variable
    ax = [axes[0, i] for i in range(n_var)]
    for i in range(n_var):
        axes[0, i].set_title(f'KDE for {variables[i]}', fontsize=font_size)
    az.plot_ppc(inf_data, data_pairs=data_pairs, kind='kde',
                num_pp_samples=n_pp_samples, random_seed=seed, ax=ax)

    # Scatter plot for each variable
    ax = [axes[1, i] for i in range(n_var)]
    for i in range(n_var):
        axes[1, i].set_title(f'Scatter for {variables[i]}', fontsize=font_size)
    az.plot_ppc(inf_data, data_pairs=data_pairs, kind='scatter',
                num_pp_samples=n_pp_samples, random_seed=seed, ax=ax)

    # Cumulative plot for each variable
    ax = [axes[2, i] for i in range(n_var)]
    for i in range(n_var):
        axes[2, i].set_title(f'Cumulative for {variables[i]}', fontsize=font_size)
    az.plot_ppc(inf_data, data_pairs=data_pairs, kind='cumulative',
                num_pp_samples=n_pp_samples, random_seed=seed, ax=ax)

    # Set font size for x-axis and y-axis labels
    for ax in axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=font_size)

    # Add legends manually if needed
    for ax in axes.flatten():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper left', fontsize=font_size)

    # Set the main title for the figure
    fig.suptitle('Posterior Predictive Check', fontsize=font_size + 2)

    # Adjust layout to make room for the main title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Display the plot if show_plt is True
    if show_plt:
        plt.show()


def plot_violin(data: pd.DataFrame, title: str, show_plot: bool = True, save_plot: bool = True,
                directory_path: str = 'results', file_path: str = 'violin_plots', file_name: Optional[str] = None,
                x_axis_label: Optional[str] = None, y_axis_label: Optional[str] = None, font_size: int = 12,
                x_ticks_rotation: int = 0):
    """
    Create and display a violin plot from a given DataFrame.
    Caution. A KDE plot assumes a normal distribution of the data.

    :param data: DataFrame where each column is a dataset to be plotted as a violin.
    :param title: Title of the plot.
    :param show_plot: Whether to display the plot. Default is True.
    :param save_plot: Whether to save the plot as a file. Default is True.
    :param directory_path: Path to the parent directory where the files are stored. Default is 'results'.
    :param file_path: Directory where the plot file should be saved. Default is 'violin_plots'.
    :param file_name: Name of the file to save the plot. If None, a timestamp-based name will be used. Default is None.
    :param x_axis_label: Label for the x-axis. If None, no label will be set. Default is None.
    :param y_axis_label: Label for the y-axis. If None, no label will be set. Default is None.
    :param font_size: Font size for the plot text elements. Default is 12.
    :param x_ticks_rotation: Rotation of the x-ticks. Default is 0.
    :return: None
    """
    # Create the violin plot using seaborn
    sns.violinplot(data=data)

    # Set x-axis label if provided
    if x_axis_label:
        plt.xlabel(x_axis_label, fontsize=font_size)

    # Set y-axis label if provided
    if y_axis_label:
        plt.ylabel(y_axis_label, fontsize=font_size)

    # Set the plot title
    plt.title(title, fontsize=font_size + 2)
    plt.tick_params(axis='x', rotation=x_ticks_rotation)
    # Adjust layout to prevent clipping of labels and title
    plt.tight_layout()

    # Save the plot
    if save_plot:
        # Generate a timestamp-based filename if no filename is provided
        file_name = datetime.datetime.now().strftime("%y%m%d_%H%M") if file_name is None else file_name

        # Ensure the directory exists; create it if it does not
        directory = f'{directory_path}/{file_path}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(f'{directory}/{file_name}.png')

    # Show the plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_ecdf(data: pd.DataFrame, title: str, n_bins: int = 50, plt_ccdf: bool = False,
              plt_cumulative_hist: bool = True, plt_theory: bool = False, show_plot: bool = True,
              save_plot: bool = True, directory_path: str = 'results', file_path: str = 'ecdf_plots',
              file_name: Optional[str] = None, font_size: int = 12, cmap: str = 'tab10'):
    """
    Generates empirical cumulative distribution function (ECDF) and complementary cumulative distribution function
    (CCDF) plots for each column in the provided DataFrame.

    :param data: DataFrame where each column is a dataset to be plotted as ECDF and CCDF.
    :param title: Title of the plot.
    :param n_bins: Number of bins in the ECDF plot.
    :param plt_ccdf: Whether to plot the complementary cumulative distribution function. Default is False.
    :param plt_cumulative_hist: Whether to plot the cumulative histogram. Default is True.
    :param plt_theory: Whether to plot the theoretical distribution. Should only be True if data is normally distributed. Default is False.
    :param show_plot: Whether to display the plot. Default is True.
    :param save_plot: Whether to save the plot as a PNG file. Default is True.
    :param directory_path: Path to the parent directory where the files are stored. Default is 'results'.
    :param file_path: Directory path where the plot will be saved. Default is 'ecdf_plots'.
    :param file_name: Name of the file to save the plot. If None, a timestamp-based filename is generated.
    :param font_size: Font size for the plot. Default is 12.
    :param cmap: Matplotlib color map. Default is 'tab10'.
    :return: None
    """
    # Create subplots: one if plt_ccdf is False, two otherwise
    if plt_ccdf:
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(18, 8), constrained_layout=True)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axs = [ax]

    # Create a color map with as many colors as there are columns in the data
    colors = plt.cm.get_cmap(cmap, len(data.columns))

    for i, column in enumerate(data.columns):
        color = colors(i)  # Get the color for the current column
        # # Plot ECDF
        axs[0].ecdf(data[column], label=f'CDF of {column}', color=color)
        if plt_ccdf:
            # Plot CCDF.
            axs[1].ecdf(data[column], complementary=True, label=f'CCDF of {column}', color=color)
        if plt_cumulative_hist:
            # Plot cumulative histogram
            _, bins, _ = axs[0].hist(data[column], n_bins, density=True, histtype='step',
                                     cumulative=True, label=f'Cumulative Histogram of {column}', color=color)
            if plt_ccdf:
                # Plot reversed cumulative histogram
                axs[1].hist(data[column], bins=bins, density=True, histtype='step', cumulative=-1,
                            label=f'Reversed Cumulative Histogram of {column}', color=color)

        if plt_theory:
            # Plot theoretical distribution
            mu = np.mean(data[column])
            sigma = np.std(data[column])
            x = np.linspace(data[column].min(), data[column].max())
            y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                 np.exp(-0.5 * (1 / sigma * (x - mu)) ** 2))
            y = y.cumsum()
            y /= y[-1]
            axs[0].plot(x, y, '--', linewidth=1.5, label=f'Theory of {column}', color=color)
            if plt_ccdf:
                axs[1].plot(x, 1 - y, '--', linewidth=1.5, label=f'Theory of {column}', color=color)

    # Set titles and labels
    axs[0].set_title('Cumulative Distribution Function (ECDF)', fontsize=font_size + 2)
    if plt_ccdf:
        axs[1].set_title('Complementary Cumulative Distribution Function (CCDF)', fontsize=font_size + 2)

    for ax in axs:
        ax.legend(fontsize=font_size)
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability of Occurrence')
        ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.

    fig.suptitle(title, fontsize=font_size + 4)  # Set the main title of the plot

    # Save the plot
    if save_plot:
        file_name = datetime.datetime.now().strftime("%y%m%d_%H%M") if file_name is None else file_name
        directory = f'{directory_path}/{file_path}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/{file_name}.png')

    # Show the plot
    if show_plot:
        plt.show()
    else:
        plt.close()
