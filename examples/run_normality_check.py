import numpy as np
import pandas as pd
from examples.data.DataLoader import DataLoader
from utils.NormalityTest import NormalityTest
from utils.data_transformation import box_cox_transformation
from utils.plotting import plot_violin, plot_ecdf


def example_01() -> None:
    """
    Example of a normality test applied to data that follows a normal distribution.

    :return: None
    """
    n_instances = 100  # Number of instances to generate for each sample

    # Define the parameters for the normal distributions
    locs = [0.0, 0.2, 0.4, 0.6, 0.8]
    scales = [0.4, 0.2, 0.5, 0.3, 0.1]

    # Generate the data from normal distributions
    data = np.stack([np.random.normal(loc=loc, scale=scale, size=n_instances) for loc, scale in zip(locs, scales)],
                    axis=1)

    # Generate the labels for the data
    labels = [f'normal{i}' for i in range(data.shape[1])]

    # Generate a violin plot
    data_df = pd.DataFrame(data, columns=labels)
    plot_violin(data=data_df, title='Normality Test: Example 01', directory_path='examples/results/normality_test',
                file_name='normality_test_example01')
    # Generate a empirical cumulative distribution plot
    plot_ecdf(data=data_df, title='Normality Test: Example 01', directory_path='examples/results/normality_test',
              file_name='normality_test_example01', plt_ccdf=True, plt_cumulative_hist=True, plt_theory=True,
              font_size=10)
    # Perform the normality test
    model = NormalityTest(data=data, algorithm_labels=labels)
    model.analyse(directory_path='examples/results', file_name='example01', save=True)


def example_02() -> None:
    """
    Example of a normality test applied to data that is not normally distributed.

    :return: None
    """
    n_instances = 100  # Number of instances to generate for each sample

    # Generate the data from non-normal distributions
    data0 = np.random.exponential(scale=1.0, size=n_instances)
    data1 = np.random.gamma(shape=2.0, scale=1.0, size=n_instances)
    data2 = np.random.weibull(a=1.5, size=n_instances)
    data3 = np.random.poisson(lam=1.0, size=n_instances)
    data4 = np.random.beta(a=1.0, b=4.0, size=n_instances)

    # Stack the data from different distributions
    data = np.stack((data0, data1, data2, data3, data4), axis=1)

    # Generate the labels for the data
    labels = ['exponential', 'gamma', 'weibull', 'poisson', 'beta']

    # Generate a empirical cumulative distribution plot
    data_df = pd.DataFrame(data, columns=labels)
    plot_ecdf(data=data_df, title='Normality Test: Example 02', directory_path='examples/results/normality_test',
              file_name='normality_test_example02', plt_ccdf=True, plt_cumulative_hist=True, plt_theory=False,
              font_size=10)
    # Perform the normality test
    model = NormalityTest(data=data, algorithm_labels=labels)
    model.analyse(directory_path='examples/results', file_name='example02', save=True)


def example_03() -> None:
    """
    Example of a normality test where the data is transformed to approximate a normal distribution.

    :return: None
    """
    n_instances = 100  # Number of instances to generate for each sample

    # Generate the data from non-normal distributions
    data0 = np.random.exponential(scale=1.0, size=n_instances)
    data1 = np.random.gamma(shape=2.0, scale=1.0, size=n_instances)
    data2 = np.random.weibull(a=1.5, size=n_instances)
    data3 = np.random.poisson(lam=1.0, size=n_instances)
    data4 = np.random.beta(a=1.0, b=4.0, size=n_instances)

    # Apply Box-Cox transformation to each data set
    data0 = box_cox_transformation(data0)
    data1 = box_cox_transformation(data1)
    data2 = box_cox_transformation(data2)
    data3 = box_cox_transformation(data3)
    data4 = box_cox_transformation(data4)

    # Stack the transformed data
    data = np.stack((data0, data1, data2, data3, data4), axis=1)

    # Generate the labels for the transformed data
    labels = ['box-cox exponential', 'box-cox gamma', 'box-cox weibull', 'box-cox poisson', 'box-cox beta']

    # Generate a violin plot
    data_df = pd.DataFrame(data, columns=labels)
    plot_violin(data=data_df, title='Normality Test: Example 03', directory_path='examples/results/normality_test',
                file_name='normality_test_example03')

    # Perform the normality test
    model = NormalityTest(data=data, algorithm_labels=labels)
    model.analyse(directory_path='examples/results', file_name='example03', save=True)


def example_04() -> None:
    """
    Example of a normality test using real data.

    This is an adapted example from the scmamp Repository:
    https://github.com/b0rxa/scmamp/blob/master/vignettes/Statistical_assessment_of_the_differences.html#L225C39-L226C1

    :return: None
    """
    # Load the dataset
    data_loader = DataLoader('data_gh_2008')

    # Get samples and labels from the data loader
    samples, labels = data_loader.get_samples()

    # Generate a violin plot
    data_df = pd.DataFrame(samples, columns=labels)
    plot_violin(data=data_df, title='Normality Test: Example 04', directory_path='examples/results/normality_test',
                file_name='normality_test_example04')

    # Perform the normality test
    model = NormalityTest(data=samples, algorithm_labels=labels)
    model.analyse(directory_path='examples/results', file_name='example04', save=True)


def main():
    seed = 42  # Set the random seed for reproducibility
    np.random.seed(seed)  # Set the random seed for NumPy

    # Run the examples
    example_01()
    example_02()
    example_03()
    example_04()


if __name__ == '__main__':
    main()  # Execute the main function
