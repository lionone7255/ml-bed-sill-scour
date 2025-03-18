"""
Machine Learning Model Definitions
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


class ModelFactory:
    """
    Factory class for creating configured ML models

    Supported Models:
    - 'RF': Random Forest
    - 'GBDT': Gradient Boosted Decision Trees
    - 'XGB': XGBoost
    """

    @staticmethod
    def get_model(model_type: str):
        """
        Create preconfigured ML model

        Args:
            model_type (str): Model identifier

        Returns:
            sklearn/xgboost model object

        Raises:
            ValueError: For unsupported model types
        """
        configs = {
            'RF': RandomForestRegressor(
                n_estimators=450,
                max_depth=10,
                max_features=3,
                random_state=42
            ),
            'GBDT': GradientBoostingRegressor(
                n_estimators=450,
                max_depth=9,
                max_features=3,
                random_state=42
            ),
            'XGB': XGBRegressor(
                n_estimators=300,
                max_depth=3,
                random_state=42
            )
        }

        if model_type not in configs:
            raise ValueError(f"Unsupported model type: {model_type}")

        return configs[model_type]