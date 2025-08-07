import pickle


def load_results(file_path: str, file_name: str) -> dict:
    """
     Load the results from a pickle file.

    :param file_path: The directory path where the pickle file is stored.
    :param file_name: The name of the pickle file to load.
    :return: A dictionary containing the loaded results.
    """
    path = f'{file_path}/{file_name}.pickle'
    with open(path, 'rb') as file:
        results = pickle.load(file)
    return results
