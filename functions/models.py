import pandas as pd
import numpy as np
from sklearn import model_selection
from fastprogress.fastprogress import progress_bar

from typing import Tuple



def predict_out_of_fold_sklearn(df, train_index, predict_index, target:str, features:list,
                                n_splits:int = 5, preds_oof_col_suffix:str ='_model_1',
                                est=None, predict_method='predict',
                                model_init_params:dict = None, model_fit_params:dict = None,
                                verbose=True) -> Tuple[pd.DataFrame, list]:
    """
    Creates out of fold predictions using sklearn estimators
    
    train_index: index of df with training and validation data
    predict_index: index used for generating predictions only
    predict_method: 'predict' or 'predict_proba'
    
    returns: source df with added predictions, models
    """

    models_trained = []

    # train_oof_preds = np.zeros(len(train_index))
    oof_preds_col = 'preds_oof_'+preds_oof_col_suffix
    df[oof_preds_col] = np.nan
    predict_preds_oof = pd.DataFrame(index=predict_index)

    if verbose: print('creating folds...')
    folds = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=41
                                 ).split(df.loc[train_index, features],
                                         df.loc[train_index, target])

    for n_fold, (fold_train_index, fold_valid_index) in enumerate(progress_bar(list(folds)), start=1): # returns index independent of dataframe index
        # split into train and validation set
        x_train = df.loc[train_index, features].iloc[fold_train_index]
        x_valid = df.loc[train_index, features].iloc[fold_valid_index]
        y_train = df.loc[train_index, target  ].iloc[fold_train_index]
        y_valid = df.loc[train_index, target  ].iloc[fold_valid_index]

        print('train', x_train.shape, 'valid:', x_valid.shape, end='\t')

        # init model
        model = est(**model_init_params)
        try:  # try to add eval set for early stopping
            if model.__repr__()[:4] == 'LGBM':
                model_fit_params['eval_set'] = (x_valid, y_valid)
                model_fit_params['eval_names'] = ('\t valid')
                model_fit_params['early_stopping_rounds'] = 20
            if model.__repr__().startswith('<catboost'):
                model_fit_params['eval_set'] = (x_valid, y_valid)
                model_fit_params['early_stopping_rounds'] = 20
        except: pass

        if model_fit_params is not None:
            model.fit(x_train, y_train, **model_fit_params)
        else: 
            model.fit(x_train, y_train)
        models_trained.append(model)

        # predictions
        if predict_method=='predict':         train_preds_oof = model.predict(x_valid)
        elif predict_method=='predict_proba': train_preds_oof = model.predict_proba(x_valid)[:,1]
        df.loc[train_index[fold_valid_index], oof_preds_col] = train_preds_oof
        
        if predict_index is not None:
            if predict_method=='predict':         predict_preds = model.predict(df.loc[predict_index, features])
            elif predict_method=='predict_proba': predict_preds = model.predict_proba(df.loc[predict_index, features])[:,1]
            predict_preds_oof[f'pred_fold_{n_fold}'] = predict_preds
        
        if verbose:
            print(f'preds mean: {train_preds_oof.mean():.4f}', end=' | ')
            try: print('score:', dict(model.best_score_))
            except: print()
            if 'verbose' in model_fit_params: print()

    if predict_index is not None:
        predict_preds_oof = predict_preds_oof.mean(axis=1)
        df.loc[predict_index, oof_preds_col] = predict_preds_oof

    return df, models_trained