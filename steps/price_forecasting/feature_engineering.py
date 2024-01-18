import logging
from data_preprocessing import dataPreprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineering:
    """
    handle Na values
    """

    def __init__(self, df) -> None:
            self.df = df

    def get_data(self) :
         return self.df
    


    def check(self):
         print((self.df).iloc[:,1])

    def normalize_data(self, columnName) :
        scaler = MinMaxScaler()
        self.df = scaler.fit_transform(self.df[columnName].values.reshape(-1,1))

    def train_test_split(self, train_size):
        df_train = self.df[:int(train_size * len(self.df))]
        df_test = self.df[int(train_size * len(self.df)):]
        return df_train, df_test
    
    def create_dataset(self, dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

              
def featureEngineering() -> pd.DataFrame:
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        # df = assign from somewhere
       featureEngineering = FeatureEngineering(dataPreprocessing())
       featureEngineering.normalize_data("Prices\n(EUR/MWh)")
       df_train, df_test = featureEngineering.train_test_split(0.8)
       X_train, y_train = featureEngineering.create_dataset(df_train, 15)
       X_test, y_test = featureEngineering.create_dataset(df_test, 15)
       return X_train,y_train, X_test, y_test
     

    except Exception as e:
        logging.error(e)
        raise e
    

print(featureEngineering())