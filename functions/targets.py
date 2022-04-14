import numpy as np
import pandas as pd

from typing import Tuple, List

def create_acquisition_general(df) -> Tuple[pd.DataFrame, str]:
    """
    creates two columns:
    - new binary column for classification model
    - new boolean column for printing
    
    returns dataframe and target as str
    """
    df['target_acquisition_qc_general_0_1'] = np.where( df['order_count_lifetime_vert_dmart']>0, 1, 0)

    return df, 'target_acquisition_qc_general_0_1'


def create_acquisition_organic(df) -> Tuple[pd.DataFrame, str]:
    """
    If acquired, with voucher/incentive 0, without 1. If not acquired then no value (NaN)
    """
    target = 'target_acquisition_qc_organic_0_1'

    # if customer's first QC order was without voucher, then organic (1), Nan if not acquired
    df.loc[ (df['first_order_voucher_vert_qc'].fillna(0)==0)
           &(df['order_count_lifetime_vert_qc']>0), target] = 1
    df.loc[ (df['first_order_voucher_vert_qc'].fillna(0)>0 )
           &(df['order_count_lifetime_vert_qc']>0), target] = 0
    
    return df, target


def create_acquisition_order_freq_4w_qc(df) -> Tuple[pd.DataFrame, str]:
    """
    Expected order frequency per 4 weeks from entire QC vertical after acquisition.
    Created from average order frequency since first order in QC.
    """
    target = 'target_order_freq_4w_vert_qc'

    df[target] = df['order_freq_4w_vert_qc'].copy()
    
    return df, target


def create_acquisition_gmv_avg_qc(df) -> Tuple[pd.DataFrame, str]:
    "Expected average order value over 4 weeks after acquisition"
    target = 'target_gmv_avg_lifetime_vert_qc'

    df[target] = df['gmv_avg_lifetime_vert_qc'].copy()
    
    return df, target