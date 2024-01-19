import logging
from zenml import step
import pandas as pd
import numpy as np
import scipy


class DataPreprocessing:
    """
    handle Na values
    """

    def __init__(self, df) -> None:
            self.df = df

    def get_data(self) :
         return self.df

    # Adding suitable value for severrisk
    def fillnan(self, columnname):
         mask = (self.df)[columnname] <= 0
         (self.df).loc[mask, columnname] = np.nan

    # Dropping empty rows
    def remove_unreliable_rows (self):
        (self.df).dropna(how='all', inplace=True)

    #Interpolate the data
    def interpolate_data (self, columnName): 
        (self.df)[columnName].interpolate(inplace=True)

    def remove_outliers(self, columnName, threshold) :
        zscore = scipy.stats.zscore((self.df)[columnName])
        self.df = (self.df)[abs(zscore)<threshold]
    

 
def dataPreprocessing(df) -> pd.DataFrame:
    """
    Args:
        df: pd.DataFrame
    Returns:
        df: pd.DataFrame
    """
    try:
        datapreprocessing = DataPreprocessing(df)
        datapreprocessing.remove_unreliable_rows() # new df is not returned but org df is modified.
        datapreprocessing.fillnan("Prices\n(EUR/MWh)")
        datapreprocessing.interpolate_data("Prices\n(EUR/MWh)")
        datapreprocessing.remove_outliers("Prices\n(EUR/MWh)", 5)

        return datapreprocessing.get_data()
    except Exception as e:
        logging.error(e)
        raise e
    

print(dataPreprocessing())