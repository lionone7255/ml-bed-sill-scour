"""
Performance Metrics Calculation
"""

import numpy as np


class EvaluationMetrics:
    """
    Contains static methods for calculating evaluation metrics
    """

    @staticmethod
    def r_squared(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate coefficient of determination (R²)

        Args:
            y_pred (np.ndarray): Predicted values
            y_true (np.ndarray): Ground truth values

        Returns:
            float: R² score
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

    @staticmethod
    def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error

        Args:
            y_pred (np.ndarray): Predicted values
            y_true (np.ndarray): Ground truth values

        Returns:
            float: RMSE value
        """
        return np.sqrt(np.mean((y_pred - y_true) ** 2))