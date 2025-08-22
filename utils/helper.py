import pickle
import os


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

def save_results(results: dict, file_path: str, file_name: str) -> None:
    """
    Save results dictionary as pickle file.

    :param results: Dictionary with results
    :param file_path: Directory where file should be stored
    :param file_name: Filename (without extension)
    """
    os.makedirs(file_path, exist_ok=True)
    path = os.path.join(file_path, f"{file_name}.pickle")
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {path}")
