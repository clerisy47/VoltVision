import logging
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from zenml import step
from zenml.steps.sklearn import SKLearnBaseStep

class ModelTraining:
    """
    Model training class which trains XGBoost and SARIMA models on the provided data.
    """

    def __init__(self, X, y, time_series_index) -> None:
        """Initialize the model training class."""
        self.X = X
        self.y = y
        self.time_series_index = time_series_index

    def train_xgboost_model(self) -> XGBRegressor:
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        xgb_model = XGBRegressor()

        xgb_model.fit(X_train, y_train)

        xgb_score = xgb_model.score(X_test, y_test)
        logging.info(f"XGBoost Model Accuracy: {xgb_score}")

        return xgb_model

    def train_sarima_model(self) -> SARIMAX:
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        sarima_model = SARIMAX(endog=y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

        sarima_results = sarima_model.fit(disp=False)

        sarima_score = sarima_results.aic
        logging.info(f"SARIMA Model AIC: {sarima_score}")

        return sarima_results


@step
def train_models(X, y, time_series_index) -> dict:
    """
    Args:
        X: pd.DataFrame or np.ndarray
            Features for training the models.
        y: pd.Series or np.ndarray
            Target variable for training the models.
        time_series_index: pd.DatetimeIndex
            Datetime index for time series models.
    Returns:
        models: dict
            Dictionary containing trained models (XGBoost and SARIMA).
    """
    try:
        model_training = ModelTraining(X, y, time_series_index)
        xgb_model = model_training.train_xgboost_model()
        sarima_model = model_training.train_sarima_model()

        models = {'xgboost': xgb_model, 'sarima': sarima_model}
        return models
    except Exception as e:
        logging.error(e)
        raise e
