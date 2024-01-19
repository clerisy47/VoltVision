import logging
import pandas as pd
from zenml import step

class DataPreprocessing:
    """
    handle Na values
    """

    def __init__(self, df) -> None:
            self.df = df

    def get_data(self) :
         return self.df

    # Adding suitable value for severrisk
    def fillna(self, columnname):
          (self.df)[columnname].fillna(0, inplace=True)

    # Dropping empty cols
    def remove_unreliable_cols (self):
        index = (self.df).shape[0]
        for column in (self.df).columns:
            count_non_null = (self.df)[column].notna().sum()
            if(count_non_null < index/3) : #index/3 = 33% of data notnull and remaining null then gets dropped
               self.drop_col(column)

    #Interpolate the data
    def interpolate_data (self): 
        x = 3
        y = (self.df).shape[1] - 2
        for column in (self.df).columns[x:y]:
            (self.df)[column] = (self.df)[column].interpolate(method='linear', limit_direction='forward', axis=0)
         
    #Drop redundant cols
    def drop_col(self, columnName) :
        if columnName in (self.df).columns:  # Check if the column exists before dropping
            self.df.drop(columnName, axis=1, inplace=True)


    #Correlation analysis
    def correlate(self, x, y):
           checkDatatypeone = pd.api.types.is_float_dtype(self.df[x]) or pd.api.types.is_integer_dtype(self.df[x])
           checkDatatypetwo = pd.api.types.is_float_dtype(self.df[y]) or pd.api.types.is_integer_dtype(self.df[y])
           if(checkDatatypeone and checkDatatypetwo):
            return self.df[x].corr(self.df[y])
           else:
            return "false"

    # To drop anything below correlation 0.1
    def correlate_every_col(self): 
         for column in (self.df).columns:
              if(pd.api.types.is_float_dtype(self.df[column]) or pd.api.types.is_integer_dtype(self.df[column])):
                if(self.correlate('Demand (MW)',column) != "false" and self.correlate('Demand (MW)',column) < 0.1):
                   self.drop_col(column) 

              
    # To drop redundant cols like Temperature and feelsLike ...  
    def check_intercorrelation(self):
         columms_to_drop = []
         for columnone in (self.df).columns :
              for columntwo in (self.df).columns: 
                   if(columnone != columntwo):
                            if(self.correlate(columnone, columntwo) != "false" and self.correlate(columnone, columntwo) > 0.95):
                                    if(self.correlate('Demand (MW)',columnone) < self.correlate('Demand (MW)', columntwo)):
                                        columms_to_drop.append(columnone)
         for column in columms_to_drop:
              self.drop_col(column)
              

@step
def dataPreprocessing(df) -> pd.DataFrame:
    """
    Args:
        df: pd.DataFrame
    Returns:
        df: pd.DataFrame
    """
    try:
        # df = assign from somewhere
        datapreprocessing = DataPreprocessing(df)
        datapreprocessing.fillna("severerisk")
        datapreprocessing.remove_unreliable_cols() # new df is not returned but org df is modified.
        datapreprocessing.interpolate_data()
        datapreprocessing.drop_col('precipprob')  # removing redundant cols
        datapreprocessing.drop_col('windgust')
        datapreprocessing.correlate_every_col()
        datapreprocessing.check_intercorrelation()
        return datapreprocessing.get_data()
    except Exception as e:
        logging.error(e)
        raise e