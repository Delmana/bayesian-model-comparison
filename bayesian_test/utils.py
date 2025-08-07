from typing import Any

import numpy as np
from numpy import floating
from scipy import stats


def _round_values(data: any, round_to: int) -> any:
    """
    Recursively round values in a data structure to a specified number of decimal places.

    :param data: The data structure (dict, list, float, numpy array) to round.
    :param round_to: The number of decimal places to round to.
    :return: The rounded data structure.
    """
    if isinstance(data, dict):
        return {k: _round_values(v, round_to) for k, v in data.items()}
    elif isinstance(data, list):
        return [_round_values(v, round_to) for v in data]
    elif isinstance(data, float):
        return round(data, round_to)
    elif isinstance(data, np.ndarray):  # Check for numpy arrays
        return np.round(data, round_to)
    else:
        return data


def print_result(results: dict, round_to: int) -> dict:
    """
    Print the rounded results and return them.

    :param results: The results dictionary to print.
    :param round_to: The number of decimal places to round to.
    :return: The rounded results dictionary.
    """
    # Print the results
    rounded_results = _round_values(results, round_to)
    print(f'\nResults:\n{rounded_results}')
    return rounded_results


def _check_dimensions(observed: np.ndarray, simulated: np.ndarray) -> None:
    """
    Check if the observed and simulated data have the same dimensions.

    :param observed: numpy array of observed values.
    :param simulated: numpy array of simulated values.
    :raises ValueError: If the dimensions of the observed and simulated arrays do not match.
    """
    if observed.shape != simulated.shape:
        raise ValueError(f'Dimension mismatch: observed data has shape {observed.shape}, '
                         f'while simulated data has shape {simulated.shape}.')


def _spearman_correlation(observed: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    """
    Calculate the Spearman correlation coefficient between observed and simulated data.

    :param observed: numpy array of observed values.
    :param simulated: numpy array of simulated values.
    :return: Spearman correlation coefficient.
    """
    _check_dimensions(observed, simulated)
    return stats.spearmanr(observed, simulated).correlation


def _kendall_tau(observed: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    """
    Calculate the Kendall Tau correlation coefficient between observed and simulated data.

    :param observed: numpy array of observed values.
    :param simulated: numpy array of simulated values.
    :return: Kendall Tau correlation coefficient.
    """
    _check_dimensions(observed, simulated)
    return stats.kendalltau(observed, simulated).correlation


def _mean_squared_error(observed: np.ndarray, simulated: np.ndarray) -> floating[Any]:
    """
    Calculate the Mean Squared Error (MSE) between observed and simulated data.

    :param observed: numpy array of observed values.
    :param simulated: numpy array of simulated values.
    :return: Mean Squared Error.
    """
    _check_dimensions(observed, simulated)
    return np.mean((observed - simulated) ** 2)


def _root_mean_squared_error(observed: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    """
    Calculate the Root Mean Squared Error (RMSE) between observed and simulated data.

    :param observed: numpy array of observed values.
    :param simulated: numpy array of simulated values.
    :return: Root Mean Squared Error.
    """
    _check_dimensions(observed, simulated)
    return np.sqrt(_mean_squared_error(observed, simulated))


def _mean_absolute_error(observed: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    """
    Calculate the Mean Absolute Error (MAE) between observed and simulated data.

    :param observed: numpy array of observed values.
    :param simulated: numpy array of simulated values.
    :return: Mean Absolute Error.
    """
    _check_dimensions(observed, simulated)
    return np.mean(np.abs(observed - simulated))


def posterior_predictive_check_metrics(observed: np.ndarray, simulated: np.ndarray, ranks: bool = False) -> dict:
    """
    Calculate various metrics to compare observed and simulated data.

    :param observed: numpy array of observed values.
    :param simulated: numpy array of simulated values.
    :param ranks: If True, include Spearman and Kendall Tau correlations in the metrics. Default is False.
    :return: Dictionary of calculated metrics including MSE, RMSE, MAE, and optionally Spearman and Kendall Tau correlations.
    """
    _check_dimensions(observed, simulated)

    mse = _mean_squared_error(observed, simulated)
    rmse = _root_mean_squared_error(observed, simulated)
    mae = _mean_absolute_error(observed, simulated)

    metrics = {'Mean Squared Error': mse, 'Root Mean Squared Error': rmse, 'Mean Absolute Error': mae}

    if ranks:
        sp = _spearman_correlation(observed, simulated)
        kt = _kendall_tau(observed, simulated)
        metrics['Spearman Correlation'] = sp
        metrics['Kendall Tau'] = kt

    return metrics


def calculate_statistics(metrics: list[dict]) -> tuple[dict, dict]:
    """
    Calculate the mean and standard deviation for each metric across multiple sets of metrics.

    :param metrics: List of dictionaries containing metric values.
    :return: Tuple containing two dictionaries: means and standard deviations of the metrics.
    """
    keys = metrics[0].keys()
    means = {key: np.round(np.mean([m[key] for m in metrics]), 4) for key in keys}
    std_devs = {key: np.round(np.std([m[key] for m in metrics], ddof=1), 4) for key in keys}
    return means, std_devs
