import logging

import pandas as pd



class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        pass

    def get_data(self) -> pd.DataFrame:
        df_demand = pd.read_csv('../../dataset/Demand Forecasting/Demand Forecasting Demand Data upto Feb 21.csv', sep=',')
        df_weather = pd.read_csv('../../dataset/Demand Forecasting/Demand Forecasting Weather Data upto Feb 28.csv', sep=',')
        df_merged=pd.merge(left=df_demand,right=df_weather, on='datetime')
        return df_merged



def ingest_data() -> pd.DataFrame:
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e