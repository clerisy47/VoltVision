import os
import pickle
from typing import Any, Type, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

DEFAULT_FILENAME = "Demand and Price Forecasting"


class cs_materializer(BaseMaterializer):
    """
    Custom materializer for Demand Price Forecasting
    """

    ASSOCIATED_TYPES = (
        str,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        CatBoostRegressor,
        RandomForestRegressor,
        XGBRegressor,
    )

    def handle_input(
        self, data_type: Type[Any]
    ) -> Union[
        str,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        CatBoostRegressor,
        RandomForestRegressor,
        XGBRegressor,
    ]:
        """
        It loads the model from the artifact and returns it.

        Args:
            data_type: The type of the model to be loaded
        """
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            obj = pickle.load(fid)
        return obj

    def handle_return(
        self,
        obj: Union[
            str,
            np.ndarray,
            pd.Series,
            pd.DataFrame,
            CatBoostRegressor,
            RandomForestRegressor,
            XGBRegressor
        ],
    ) -> None:
        """
        It saves the model to the artifact store.

        Args:
            model: The model to be saved
        """

        super().handle_return(obj)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(obj, fid)