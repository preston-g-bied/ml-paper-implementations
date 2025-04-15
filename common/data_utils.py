# common/data_utils.py

"""
Common data utilities for paper implementations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional, Union

def load_dataset(dataset_name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a common dataset used across multiple paper implementations.
    
    Args:
        dataset_name: Name of the dataset to load
        **kwargs: Additional arguments specific to the dataset
        
    Returns:
        Tuple of (features, labels)
    """
    if dataset_name.lower() == "iris":
        from sklearn.datasets import load_iris
        data = load_iris()
        return data.data, data.target
    
    elif dataset_name.lower() == "mnist":
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        return X.astype(np.float32), y.astype(np.int64)
    
    elif dataset_name.lower() == "cifar10":
        # implement CIFAR-10 loading
        raise NotImplementedError("CIFAR-10 loading not implemented yet")
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
def train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split as sk_split
    return sk_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(
        X: np.ndarray,
        y: np.ndarray,
        normalize: bool = True,
        shuffle: bool = True,
        random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess data for training.
    
    Args:
        X: Features
        y: Labels
        normalize: Whether to normalize features
        shuffle: Whether to shuffle data
        random_state: Random seed for reproducibility
        
    Returns:
        Processed X, y
    """
    # shuffle data if requested
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

    # normalize features if requested
    if normalize:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    return X, y

def create_batches(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create mini-batches from data.
    
    Args:
        X: Features
        y: Labels
        batch_size: Size of each batch
        
    Returns:
        List of (X_batch, y_batch) tuples
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batches.append((X[batch_indices], y[batch_indices]))

    return batches

def save_results(
        results: Dict[str, Any],
        paper_dir: str,
        filename: str = "results.npz"
) -> None:
    """
    Save experiment results.
    
    Args:
        results: Dictionary of results to save
        paper_dir: Directory of the paper implementation
        filename: Name of the results file
    """
    results_dir = os.path.join(paper_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, filename)
    np.savez(results_path, **results)
    print(f"Results saved to {results_path}")