import datetime
import numpy as np
from sklearn import metrics



def mape(y_true, y_pred) -> float:
    "calculate mean average percentage error"
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def regression_metrics(y_true, y_pred) -> dict:
    "create dictionary with regression metrics for passed arrays/series"
    reg_metrics = {}
    
    reg_metrics['mse']  = round(metrics.mean_squared_error(   y_true, y_pred),                4)
    reg_metrics['rmse'] = round(metrics.mean_squared_error(   y_true, y_pred, squared=False), 4)
    reg_metrics['mae']  = round(metrics.median_absolute_error(y_true, y_pred),                4)
    reg_metrics['r2']   = round(metrics.r2_score(             y_true, y_pred),                4)
    
    if y_true.min()<=0: pass
    else: reg_metrics['mape'] = round(mape(y_true, y_pred), 4)
    
    reg_metrics['y_pred_lower_equal_0'] = (y_pred <= 0).sum()
    reg_metrics['y_pred_max']           = round(y_pred.max(), 2)
    reg_metrics['std']                  = round(y_pred.std(), 4)
    
    return reg_metrics


def binary_metrics(y_true, y_pred) -> dict:
    "create dictionary with binary classification metrics"
    y_pred_01 = np.round(y_pred, 0)

    bin_metrics = {}
    bin_metrics['roc auc'] =   round(metrics.roc_auc_score(  y_true=y_true, y_score= y_pred),    4)
    bin_metrics['f1'] =        round(metrics.f1_score(       y_true=y_true, y_pred = y_pred_01), 4)
    bin_metrics['precision'] = round(metrics.precision_score(y_true=y_true, y_pred = y_pred_01, zero_division=0), 4)
    bin_metrics['recall'] =    round(metrics.recall_score(   y_true=y_true, y_pred = y_pred_01), 4)
    bin_metrics['accuracy'] =  round(metrics.accuracy_score( y_true=y_true, y_pred = y_pred_01), 4)
    bin_metrics['log loss'] =  round(metrics.log_loss(       y_true=y_true, y_pred = y_pred),    4)
    
    return bin_metrics


def regression_metrics_text(y_true, y_pred, timestamp=True) -> str:
    "return string of metrics for regression predictions"
    metrics_dict = regression_metrics(y_true, y_pred)
    
    text = ''
    for key, value in metrics_dict.items():
        text += f'{key}: {value:.4f}\t'
        
    if timestamp: text+= str(datetime.datetime.now())[:19]
    
    return text


def binary_metrics_text(y_true, y_pred, timestamp=True) -> str:
    "return string of metrics for regression predictions"
    metrics_dict = binary_metrics(y_true, y_pred)
    
    text = ''
    for key, value in metrics_dict.items():
        text += f'{key}: {value:.4f} | '
        
    if timestamp: text+= str(datetime.datetime.now())[:19]
    
    return text