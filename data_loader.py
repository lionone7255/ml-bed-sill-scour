"""
Data Loading and Preprocessing Module
"""

import numpy as np
from sklearn.model_selection import train_test_split


class DataHandler:
    """
    Handles dataset loading and preprocessing operations

    Attributes:
        test_size (float): Proportion of dataset for testing
        random_state (int): Random seed for reproducibility
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def stratified_split(self, features: np.ndarray, labels: np.ndarray):
        """
        Perform stratified train-test split based on label distribution

        Args:
            features (np.ndarray): Input features matrix
            labels (np.ndarray): Target values

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            features,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels
        )

    def load_demo_data(self, n_samples: int = 1000):
        """
        Generate synthetic demo data for testing purposes

        Args:
            n_samples (int): Number of samples to generate

        Returns:
            tuple: (features, labels)
        """
        np.random.seed(self.random_state)
        features = np.random.rand(n_samples, 4)
        labels = np.random.rand(n_samples) * 10
        return features, labels