"""
Result Visualization Utilities
"""

import matplotlib.pyplot as plt


class ResultVisualizer:
    """
    Helper class for creating standardized visualizations
    """

    @staticmethod
    def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str):
        """
        Create prediction vs actual scatter plot

        Args:
            y_true (np.ndarray): Ground truth values
            y_pred (np.ndarray): Predicted values
            title (str): Plot title
        """
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, edgecolor='k', alpha=0.6)
        plt.plot([0, 10], [0, 10], 'k--')
        plt.title(title)
        plt.xlabel('Observed ($y_s/H_s$)')
        plt.ylabel('Predicted ($y_s/H_s$)')
        plt.grid(True, linestyle='--', alpha=0.5)