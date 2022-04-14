import pandas as pd
import datetime

import sys, os

#sys.path.insert(1, '/Users/temporaryadmin/Documents/GitHub/qc_data_insights/utils')
sys.path.insert(1, '..')
sys.path.insert(1, '../../')
from utils import utils

###############################################################################
######### local ###############################################################
###############################################################################

def save_to_local_parquet(df, file_path):
    df.to_parquet(file_path, index=False, engine='auto', compression='snappy')
    print(f"saved to: '{file_path}' with size {utils.get_file_size(file_path)}")
    

def load_from_local_parquet(file_path):
    print(f"loading '{file_path}' with size {utils.get_file_size(file_path)} to df", end='... ')
    df = pd.read_parquet(file_path)
    print('done')
    
    return df

#############################
def bq_schema_to_dict(bq_schema:list) -> dict:
    "convert bq schema to dict with col_name:dtype key-value pairs"
    type_dict = {'INT64':'int64',
                 'INTEGER':'int64',
                 'FLOAT64':'float64',
                 'FLOAT':'float64',
                 'NUMERIC':'float64',
                 'BIGNUMERIC':'float64',
                 'STRING':'object',
                 'DATE':'datetime64'
                }
    
    schema_dict = {}
    for col in bq_schema: schema_dict[col.name] = type_dict[col.field_type]
        
    return schema_dict


def read_bigquery(query:str, bqclient, parse_dates:list=None, location:str = 'US', 
                  verbose=False) -> pd.DataFrame:
    """
    Load a query from BigQuery into dataframe.
    query: query-string or file path
    """

    if query.endswith('.sql'):  query_string = open(query, 'r').read().replace('%', '%%')
    else:                       query_string = query
    
    if verbose:
        print('running query...', end=' ')
        progress_bar_type = 'tqdm_notebook'
    else:
        progress_bar_type = None

    job = bqclient.query(query_string, location=location)
    if verbose: print('job done, downloading...', end=' ')
    
    result = job.result()
    schema_dict = bq_schema_to_dict(result.schema)
    df = result.to_dataframe(progress_bar_type=progress_bar_type, 
                             dtypes=schema_dict,
                             create_bqstorage_client=True)
    if verbose: print('done with shape', utils.df_shape(df))

    if parse_dates is not None:
        for date_col in parse_dates:
            try:    df[date_col] = pd.to_datetime(df[date_col], errors='raise').dt.tz_localize(None)
            except: print('ERROR converting to datetime for column:', date_col)

    return df