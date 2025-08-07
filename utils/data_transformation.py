import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def _check_data(data: np.ndarray) -> np.ndarray | None:
    """
    Check the Input numpy array for NaN or Inf values.

    :param data: Input numpy array to be checked for NaN or Inf values.
    :return: The original numpy array if no NaN or Inf values are found. Otherwise, None.
    """
    # Check for NaN or Inf values in the data
    if np.isnan(data).any() or np.isinf(data).any():
        # Identify the indices and values that are NaN or Inf
        invalid_indices = np.where(np.isnan(data) | np.isinf(data))
        invalid_values = data[invalid_indices]

        # Raise a warning with detailed information about the invalid values
        warnings.warn(
            f'INVALID VALUE: The transformation resulted in NaN or Inf values. '
            f'Invalid indices: {invalid_indices}, Invalid values: {invalid_values}'
        )
        # Return None to indicate invalid data
        return None

    # Return the original data if no NaN or Inf values are found
    return data


def sqrt_transformation(data: np.ndarray, positively_skewed: bool = True) -> np.ndarray:
    """
    Apply square-root transformation for moderate skewness.
.
    :param data: Input numpy array.
    :param positively_skewed: Boolean indicating if the data is positively skewed. Default is True.
    :return: Transformed numpy array.
    """
    if positively_skewed:
        data_trans = np.sqrt(data)
    else:
        data_trans = np.sqrt(np.max(data + 1) - data)
    return _check_data(data_trans)


def log_transformation(data: np.ndarray, positively_skewed: bool = True) -> np.ndarray:
    """
    Apply logarithmic transformation for greater skewness.

    :param data: Input numpy array.
    :param positively_skewed: Boolean indicating if the data is positively skewed. Default is True.
    :return: Transformed numpy array.
    """
    if positively_skewed:
        data_trans = np.log(data)
    else:
        data_trans = np.log(np.max(data + 1) - data)
    return _check_data(data_trans)


def reciprocal_transformation(data: np.ndarray, positively_skewed: bool = True) -> np.ndarray:
    """
    Apply reciprocal (inverse) transformation for severe skewness.

    :param data: Input numpy array.
    :param positively_skewed: Boolean indicating if the data is positively skewed. Default is True.
    :return: Transformed numpy array.
    """
    if positively_skewed:
        data_trans = 1 / data
    else:
        data_trans = 1 / (np.max(data + 1) - data)
    return _check_data(data_trans)


def box_cox_transformation(data: np.ndarray, lower_bound: int = -5, upper_bound: int = 5) -> np.ndarray:
    """
    Apply Box-Cox transformation to normalize data.

    :param data: Input numpy array.
    :param lower_bound: Lower bound for the Box-Cox transformation plot. Default is -5.
    :param upper_bound: Upper bound for the Box-Cox transformation plot. Default is 5.
    :return: Transformed numpy array.
    """
    # Ensure all values are positive for Box-Cox transformation
    if np.min(data) <= 0:
        print('Transform data to be all positive')
        data = data - np.min(data) + 1e-6

    # Apply Box-Cox transformation
    box_cox_data, lambda_ = stats.boxcox(data)
    print(f'Best lambda parameter is {round(lambda_, 3)}')

    # Plot the Box-Cox transformation with the best lambda value
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.boxcox_normplot(data, lower_bound, upper_bound, plot=ax)
    ax.axvline(lambda_, color='r')
    plt.show()

    return box_cox_data
