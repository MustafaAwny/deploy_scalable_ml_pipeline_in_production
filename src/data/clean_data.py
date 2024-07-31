import logging
import pandas as pd


def load_data(path):
    try:
        df = pd.read_csv(path)
        logging.info('Data imported successfully')
        return df
    except BaseException:
        logging.info('Reading data failed')


def cleaned_data(df):
    try:
        df.columns = df.columns.str.strip()
        df.drop(["fnlgt", "education-num", "capital-gain", "capital-loss"],
                 axis=1, inplace=True)
        
        df.to_csv('census_cleaned.csv')
        logging.info('Data cleaned successfully')
        return df
    
    except BaseException:
        logging.info('Data is not cleaned')


df = load_data('census.csv')
cleaned_data(df)