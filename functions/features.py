import pandas as pd
import numpy as np
from sklearn import preprocessing
import eli5
import warnings

from IPython.display import display


###############################################################################
#### following functions copied from helper library sloppy ####################
###############################################################################

def get_features_list(df, dtype='number', contains:list=[], contains_not:list=[], sort_results = True, verbose=True) -> list:
    """
    Returns list of continous or categorical features from DataFrame.
    dtype: number, int, float, etc
    contains: must contain all strings in list
    contains_not: must not contain any of strings in list
    """

    if dtype is not None:
        column_list = [col for col in df.select_dtypes(dtype).columns]
    else:
        column_list = [col for col in df.columns]

    for s in contains:     column_list = [col for col in column_list if col.find(s)> -1]
    for s in contains_not: column_list = [col for col in column_list if col.find(s)== -1]

    if sort_results: column_list = sorted(column_list)
        
    if verbose:
        print('found columns:', len(column_list))
        diff = len(column_list) - len(list(set(column_list)))
        if diff>0: print('found', diff, 'duplicate column names')
    
    return column_list


################### feature importances #######################################
def feat_importances_from_models(models:list, features:list) -> pd.DataFrame:
    """Create sorted feature importances as DataFrame from 1-n models"""
    model_importances = pd.DataFrame({'feature':features})

    for counter, m in enumerate(models):
        model_importances[f'imp_{counter}'] = m.feature_importances_
        model_importances[f'imp_{counter}'] = model_importances[f'imp_{counter}'] / model_importances[f'imp_{counter}'].sum()

    model_importances['imp_mean'] = model_importances.mean(axis=1)
    #model_importances['pctg']     = np.round(100 * model_importances['imp_mean'] / model_importances['imp_mean'].sum(), 4)

    model_importances = model_importances.sort_values('imp_mean', ascending=False).reset_index(drop=True)
    model_importances.index = np.array(model_importances.index) + 1

    return model_importances


def permutation_importance(est, x_valid, y_valid, n_iter=1, rank=False, verbose=True) -> pd.DataFrame:
    """
    Creates a df with all features and their importance from a permutation-shuffle run.
    n_iter: number of random shuffles for each feature, higher: more accurate, slower
    rank:   add column with rank of importance
    """
    if verbose: print('calculating permutation importance...', end=' ')

    perm_imp = (eli5
                .sklearn.PermutationImportance(est, cv='prefit', random_state=42, n_iter=n_iter)
                .fit(x_valid, y_valid)
               )

    perm_imp_df = (pd.DataFrame({'feature'     : x_valid.columns.to_list(),
                                 'perm_imp'    : perm_imp.feature_importances_,
                                 'perm_imp_std': perm_imp.feature_importances_std_,
                                 }
                               )
                   .sort_values('perm_imp', ascending=False)
                   .reset_index(drop=True)
                   )

    if rank:
        perm_imp_df['perm_imp_rank'] = perm_imp_df['perm_imp'].rank(method='average',
                                                                    ascending=False
                                                                   ).astype(int)
    if verbose: print('done, features with positive importance:', len(perm_imp_df.query('perm_imp > 0')), 'out of', len(perm_imp_df))

    return perm_imp_df


################### feature engineering #######################################

def create_high_cardinality_bins(df, columns:list, min_count:int = 20, verbose=True) -> pd.DataFrame:
    """Create new columns with bin-value for high cardinality values, e.g. post codes."""

    new_columns = []

    df['tmp'] = 1

    print('replacing high cardinility categories:')
    print(f'{"columns".ljust(52)}| rows < min count ({min_count})')

    for col in columns:
        new_column_name = f'{col}__min_count_{min_count}'
        new_columns.append(new_column_name)

        print(f'- {col.ljust(50)}', end='|        ')
        col_counts = df.groupby(col)['tmp'].transform("count")
        df[new_column_name] = np.where(col_counts < min_count, 'OTHER_HIGH_CARDINALITY', df[col])

        below_min_count = len(col_counts[col_counts<min_count])
        print(str(below_min_count).rjust(14))

    df = df.drop('tmp', axis=1)

    return df, new_columns


def convert_to_pd_catg(df, columns: list, verbose=True) -> pd.DataFrame:
    """
    Converts all columns to pandas categorical type.
    Enables additional functions and more memory-efficient data handling.
    """
    if verbose: print('converting to categorical:')
    for col in columns:
        try:
            if verbose: print(f'- {col}', end=' ')
            df[col] = df[col].astype('category')
            if verbose: print('ok')
        except:
            print(' error')

    return df


def create_count_encoding(df, columns:list, scaler:'sklearn.preprocessing. ...' = None,
                          verbose=True, drop_orig_cols=False) -> pd.DataFrame:
    """
    Expects a DataFrame with no missing values in specified columns.
    Creates new columns for every column combination (one or more columns to be combined).

    :df:                    DataFrame
    :column_combinations:   list of single or multiple columns,
                            eg.: ['country', 'product', ['country', 'product']]
    :scaler:                sklearn scaler for normalization
    :drop_orig_cols:        drop original columns after count-encoding
    """

    # create temporary column with no missing values, used for counting
    df['tmp'] = 1

    new_columns = []

    if verbose: print('adding categorical counts...')
    for col in columns:
        # set name suffix for new column

        new_column_name = 'ft_' + col + '__count'
        if verbose: print(f'- {new_column_name.ljust(60)}', end = ' ')

        # groupby count transform
        counts = df.groupby(col)['tmp'].transform('count').values.reshape(-1, 1)#.astype(int)

        if scaler:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                counts = scaler.fit_transform(counts); # suppress warnings
                scaler_str = str(type(scaler)).split('.')[-1].split('Scaler')[0].split('Transformer')[0].lower()
                new_column_name = f'{new_column_name}_scaled_{scaler_str}'

        df[new_column_name] = counts

        if verbose: print('unique', str( df[new_column_name].nunique() ).rjust(5),
                          '| min',  str( df[new_column_name].min()     ).rjust(5),
                          '| max',  str( df[new_column_name].max()     ).rjust(5))

        if drop_orig_cols: df = df.drop(col, axis=1)

        new_columns.append(new_column_name)

    df = df.drop('tmp', axis=1)

    return df, new_columns


def create_label_encoding(df, columns:list, drop_orig_cols = False, verbose = True):
    """
    Add numerical labels for categorical values.
    Values under a specified low total count are grouped together as '0'
    """
    #max_col_length = len(max(columns, key=len))

    new_columns = []

    df['tmp'] = 1

    if verbose: print('adding label encoding...')
    # set name suffix for new column
    for col in columns:
        new_column_name = 'ft_' + col + '__label'
        new_columns.append(new_column_name)

        if verbose: print('-', new_column_name.ljust(50), end=' ')

        column_values = df[col].copy().values
        label_encoder = preprocessing.LabelEncoder()
        df[new_column_name] = label_encoder.fit_transform(column_values)

        if verbose: print('unique:', str(df[new_column_name].nunique()).ljust(7))

        if drop_orig_cols: df = df.drop(col, axis=1)

    df = df.drop('tmp', axis=1)

    return df, new_columns


def create_one_hot_encoding(df, columns: list, min_pctg_to_keep=0.03, return_new_cols=True, verbose=True):
    """
    Adds one-hot encoded columns for each categorical column
    """
    max_col_length = len(max(columns, key=len))

    new_columns = []

    print('creating one-hot columns:')
    for column in columns:
        #new_columns = [column + "_" + i for i in full[column].unique()] #only use the columns that appear in the test set and add prefix like in get_dummies
        if verbose: print('-', column.ljust(max_col_length), end=' ')

        if df[column].nunique() > 500:
            print('too many unique values', df[column].nunique())
        else:
            one_hot_df = pd.get_dummies(df[column], prefix=f'ft_{column}__one_hot_')
            orig_col_number = len(one_hot_df.columns)

            keep_cols = (one_hot_df.sum()/len(one_hot_df))>=min_pctg_to_keep
            one_hot_df = one_hot_df.loc[:, keep_cols]

            if verbose: print(f'keep {len(one_hot_df.columns)}/{orig_col_number} one-hot columns')

            # drop columns if they already exist, in case function is called twice
            df = df.drop(one_hot_df.columns, axis=1, errors='ignore')
            df = pd.concat((df, one_hot_df), axis = 1)

            new_columns.extend(list(one_hot_df.columns))

    new_columns = list(set(new_columns))

    return df, new_columns