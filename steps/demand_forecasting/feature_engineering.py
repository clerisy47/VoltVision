import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from zenml import step

class FeatureEngineering:
    """
    handle Na values
    """

    def __init__(self, df) -> None:
            self.df = df

    def get_data(self) :
         return self.df
    
    def handle_categorical_data( self, categoryColumn) :
        dummies = pd.get_dummies(self.df[categoryColumn], prefix='overcast')
        self.df = pd.concat([(self.df), dummies], axis=1)

    def check(self):
         print((self.df).iloc[:,1])

    def normalize_data(self, x1, x2) :
        scaler = MinMaxScaler()
        X = scaler.fit_transform((self.df).iloc[:,x1:x2])
        y = scaler.fit_transform((self.df).iloc[:,1].values.reshape(-1,1))
        return X,y

    def train_test_split(self, X, y, train_size):
        X_train = X[:int(train_size * len(X))]
        X_test = X[int(train_size * len(X)):]
        y_train = y[:int(train_size * len(y))]
        y_test = y[int(train_size * len(y)):]
        return X_train, X_test, y_train, y_test

@step
def featureEngineering(df) -> pd.DataFrame:
    """
    Args:
        df: pd.DataFrame
    Returns:
        df: pd.DataFrame
    """
    try:
        # df = assign from somewhere
       featureEngineering = FeatureEngineering(df)
       featureEngineering.handle_categorical_data('conditions')
       normalized_data = featureEngineering.normalize_data(3,7)
       splitted_data = featureEngineering.train_test_split(*normalized_data, 0.8)
       return splitted_data
     

    except Exception as e:
        logging.error(e)
        raise e