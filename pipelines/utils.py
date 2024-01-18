import logging

import pandas as pd
from steps.demand_forecasting.data_preprocessing import DataPreprocessing
from steps.demand_forecasting.data_ingestion import ingest_data

def get_data_for_test():
    try:
        
        return DataPreprocessing(ingest_data()).get_data()
    except Exception as e:
        logging.error(e)
        raise e