import logging

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE, r2_score as R2Score, mean_absolute_error as MAE
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker
from typing import Tuple


@step(experiment_tracker=experiment_tracker.name)
def evaluation(
    model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:

    """
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """
    try:


        prediction = model.predict(x_test)

        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)

        r2_class = R2Score()
        r2_score = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)
        
        return r2_score, rmse
    except Exception as e:
        logging.error(e)
        raise e