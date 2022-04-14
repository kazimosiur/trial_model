# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <br>
# <font size="20">Customer Insights - Predictions - QC Acquisiton</font>
# 
# 1. load data from BigQuery
# 3. definition of features
# 2. per use case: definition of target variables
# 4. predictions for acquisitions
# 5. export data to Google Storage bucket
# 6. data available through BigQuery view
# 
# [Confluence Documentation](https://confluence.deliveryhero.com/display/DINV/Customer+Segmentation)
# %% [markdown]
# # general
# %% [markdown]
# ## libraries

# %%
import pandas as pd
import numpy as np

from sklearn import preprocessing, linear_model, ensemble, metrics, model_selection
import lightgbm

import shap

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('whitegrid')
#from jupyterthemes import jtplot #;jtplot.style()

import datetime, dateutil
import os, sys, yaml, gc, psutil, argparse
from tqdm.auto import tqdm

from typing import Tuple, List

from IPython.display import display
from pprint import PrettyPrinter
pprint = PrettyPrinter(indent=1, width=160, compact=True).pprint

# main functions for this project
sys.path.insert(1, '../')
import functions as f

sys.path.insert(1, '../../')
from utils import utils
utils.set_pd_options()


# %%
import watermark
print(watermark.watermark(
    python=True, hostname=True, machine=True,
    packages='pandas,numpy,pyarrow,lightgbm,shap,google.cloud.bigquery,google.cloud.bigquery_storage,google.cloud.storage'))

# %% [markdown]
# ## connections

# %%
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file(os.path.expanduser('~')+'/Desktop/Work/google_cloud_data-insights-team.json')

from google.cloud import bigquery
bqclient = bigquery.Client(credentials=credentials, project=credentials.project_id)
print(f'connected to BigQuery, project: {bqclient.project}')


# %%
from google.cloud import storage
storage.blob._DEFAULT_CHUNKSIZE  = 5*1024*1024  # workaround for 60s timeout
storage.blob._MAX_MULTIPART_SIZE = 5*1024*1024
storage_client = storage.Client(credentials=credentials, project=credentials.project_id)
bucket_gs = storage_client.get_bucket('darkstores-data-eng-us')
print(f'connected to Google Storage, bucket: {bucket_gs.name}')

# %% [markdown]
# ## parameters

# %%
IS_NOTEBOOK = utils.is_notebook()

if IS_NOTEBOOK:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    get_ipython().run_line_magic('config', 'Completer.use_jedi = False')
    shap.initjs() # initiate javascript notebook support for shap plots

today            = datetime.date.today()
this_week_monday = today + datetime.timedelta(days=-today.weekday(),   weeks=0)
last_week_sunday = today + datetime.timedelta(days=-today.weekday()-1, weeks=0)
today_string            = str(today)[:10]
this_week_monday_string = str(this_week_monday)
last_week_sunday_string = str(last_week_sunday)

print('today:               ', today_string)
print('this week monday:    ', this_week_monday)
print('last week sunday:    ', last_week_sunday)


# %%
RUN_PERMUATION_IMP = False
N_CORES_ASSIGNED = psutil.cpu_count() - 2 # leave 2 cores free on local machine, increase to full capacity on server

# google storage directory
DIR_TOPIC = 'customer_insights/acquisition/'
DIR_PLOTS = DIR_TOPIC+'plots'

# local directory
DIR_TMP       = os.path.join(os.path.expanduser('~'), 'tmp')
DIR_TMP_PLOTS = os.path.join(DIR_TMP, 'plots')

os.makedirs(DIR_TMP,       exist_ok=True)
os.makedirs(DIR_TMP_PLOTS, exist_ok=True)


# %%
parser = argparse.ArgumentParser()
parser.add_argument("-geid", "--global_entity_id", type=str, default='OTHER')
if IS_NOTEBOOK: args = parser.parse_args("")
else:           args = parser.parse_args()

    
if args.global_entity_id=='OTHER':  GLOBAL_ENTITY_ID = 'IN_EG'
else:                               GLOBAL_ENTITY_ID = args.global_entity_id

GLOBAL_ENTITY_ID = GLOBAL_ENTITY_ID
DATE_UNTIL = last_week_sunday_string
#DATE_UNTIL = '2021-06-27'
print(f'\n\n{"-"*40}\n{GLOBAL_ENTITY_ID}, {DATE_UNTIL}\n{"-"*40}\n\n')

DATE_UNTIL_MINUS_7  = datetime.date.fromisoformat(DATE_UNTIL) - datetime.timedelta(days=7)
DATE_UNTIL_MINUS_28 = datetime.date.fromisoformat(DATE_UNTIL) - datetime.timedelta(days=28)

try:
    del df, export, x_train, x_valid, y_train, y_valid
    gc.collect()
except:
    pass

# %% [markdown]
# ## functions
# should be moved into separate file

# %%
def shap_summary_plot(shap_values, feature_df, top_n:int, target:str):
    "create shap summary plot, show and save to GS bucket"
    
    shap.summary_plot(shap_values,
                      feature_df, 
                      max_display=top_n, 
                      show=False)

    target_str = target.replace("target_", "")
    plt.title(f'{target_str}\n{GLOBAL_ENTITY_ID}, data until {DATE_UNTIL}, top {top_n} features')    
    
    plot_dir  = os.path.join(DIR_TMP_PLOTS, GLOBAL_ENTITY_ID, target)
    plot_path = os.path.join(DIR_TMP_PLOTS, GLOBAL_ENTITY_ID, target, f'shap_{GLOBAL_ENTITY_ID}_{DATE_UNTIL}_{target}.png')
    
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_path, format='png', dpi=120, bbox_inches='tight')
    utils.upload_gs_blob(bucket_gs, plot_path, 
                         f'{DIR_PLOTS}/entity={GLOBAL_ENTITY_ID}/date={DATE_UNTIL}/target={target}/',
                         verbose=False)


# %%
def shap_forces_single_row(shap_values_row, feature_names, color=True, top_n=None):
    shap_importance = (pd.DataFrame({'feature':feature_names,
                                     'shap_importance':shap_values_row})
                        .reset_index(drop=True)
                      )
    shap_importance['shap_importance_abs'] = shap_importance['shap_importance'].abs()
    shap_importance['shap_importance_rank'] = shap_importance['shap_importance_abs'].rank(method='average', ascending=False).astype(int)
    
    shap_importance = (shap_importance
                       .sort_values('shap_importance_rank', ascending=True)
                       .drop(columns='shap_importance_abs')
                       .reset_index(drop=True)
                      )

    if top_n is not None: shap_importance = shap_importance.head(top_n)
    if color: return shap_importance.style.background_gradient(subset='shap_importance', cmap='bwr', low=-0.4, high=0.4)
    else:     return shap_importance

# %% [markdown]
# # input
# %% [markdown]
# ## load data

# %%
query = utils.sql_query_from_placeholders('sql/features_acquisition_qubik.sql',
                                          {'GLOBAL_ENTITY_ID':GLOBAL_ENTITY_ID, 'DATE_UNTIL':DATE_UNTIL})
# removed date_until from the sql query - now just takes calculations from yesterday's date
# print(query)


# %%
data = f.loaders.read_bigquery(query, bqclient, verbose=True)


# %%
df = data.copy()

# %% [markdown]
# ## dtype changes

# %%
date_cols = [c for c in df.columns if c.find('date')>=0]

for c in date_cols:
    try:    df[c] = pd.to_datetime(df[c])
    except: print('could not convert to dates:', c)


# %%
# # BQ transfers data as objects instead of int/float in many cases
# to_num_cols = df.drop(columns=['global_entity_id', 'analytical_customer_id']).select_dtypes('object').columns
# if len(to_num_cols)>0:
#     for c in tqdm(to_num_cols):
#         try:
#             df[c] = pd.to_numeric(df[c])
#         except Exception as e:
#             print(c.ljust(40), e)


# %%
#if IS_NOTEBOOK: 
(utils
 .df_info(df.iloc[:, -50:])
 .style.background_gradient(subset=['isnull_%'], cmap='Reds')
)

# %% [markdown]
# # features
# %% [markdown]
# ## categorical

# %%
df.select_dtypes(['object', 'category']).columns


# %%
# feat_catg = ['country']


# %%
# df, cols_one_hot = f.features.create_one_hot_encoding(df, columns=['visit_last_platform_device'], 
#                                                       min_pctg_to_keep=0.005)
# print('\nfeatures one hot:', cols_one_hot)

# %% [markdown]
# ## continuous

# %%
# unique prefixes
if IS_NOTEBOOK:
    print(sorted(list(set([col.split('past')[0].split('nv_')[0].split('rs_')[0] 
                           for col in df.columns]))))


# %%
features_geo = [
    'coverage_vendors_qc', 'coverage_dmart', 'coverage_localstore', 
    'coverage_convenience', 'coverage_groceries', 'coverage_supermarket',
    'coverage_vendors_qc_in_2000m', 'coverage_vendors_qc_in_4000m', 'coverage_vendors_qc_in_6000m',
    'dist_nearest_vendor_qc', 'dist_nearest_dmart', 'dist_nearest_localstore']


# %%
# if IS_NOTEBOOK:
#     df['orders_of_visits_l12w'] = (df['order_count_l12w_vert_rs'] / df['visit_count_l12w']).clip(-1, 2)
#     df['orders_of_visits_l12w'].hist(bins=25);


# %%
features_qubik = [
    ### cause overfit for some reason
    #'customer_id_countd', 'visit_vendors_viewed_l04w', 'visit_vendors_available_l04w', 'visit_addresses_unique_l04w', 'visit_count_l04w',
    'order_rate_online_payment_l04w',
     'visit_count_l12w', 'visit_count_l04w_vs_l12w',
    'visit_session_dur_sum_l04w', 'visit_session_dur_avg_l04w',
    'visit_interact_speed_avg_l04w', 'visit_cart_abandon_rate_l04w', 'visit_search_fail_rate_l04w', 'visit_voucher_error_rate_l04w',
    'rating_avg_l04w', 
    'aos_score_last', 'aos_rating_last',
    'ccc_sessions_l04w',
    #'visit_channel_first', 'visit_channel_last', 'visit_last_platform_device', 
]


# %%
utils.display_df(df.loc[:, features_qubik].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T)


# %%
utils.df_info(df.loc[:, features_qubik]).style.background_gradient()


# %%
features_cont = f.features.get_features_list(df, 'number', contains=['vert_rs'])
#features_cont

# %% [markdown]
# ## final features

# %%
feat_one_hot = f.features.get_features_list(df, contains=['one_hot'])


# %%
# exclude one-hot for now
features_all = (
    features_cont
    +feat_one_hot
    +features_geo
    +features_qubik
)
len(features_all)

# %% [markdown]
# ## check

# %%
# df.loc[:, ['dist_nearest_dmart', 'dist_nearest_localstore', 'dist_nearest_vendor_qc']].describe().astype(int)

# %% [markdown]
# # predictions
# %% [markdown]
# - generate predictions with a single model
# - get regular and permutation feature importances
# - remove useless features
# - train out-of-fold models with reduced feature set
# %% [markdown]
# ## general acquisition

# %%
# acquired, if at least one QC order
print('='*12, 'general trial probability', '='*12)
df, target = f.targets.create_acquisition_general(df)
utils.display_value_counts(df[target])


# %%
list(df.columns)


# %%
# # check for differences between target groups, check to avoid overfitting, target leakage, etc
# (df
#  .groupby(target)
#  .agg({'days_since_first_order_vert_rs':['mean', 'min', 'max'],
#        'days_since_last_order_vert_rs':['mean', 'min', 'max'],
#        'voucher_dh_sum_vert_rs_lifetime':['mean', 'min', 'max'],
#        'order_amount_gmv_eur_avg_vert_rs_lifetime':['mean', 'min', 'max'],
#        'delivery_fee_sum_vert_rs_lifetime':['mean', 'min', 'max'],
#        'discount_other_sum_vert_rs_lifetime':['mean', 'min', 'max']
#       })
#  .reset_index()
#  .T
# )

# %% [markdown]
# ### benchmark model

# %%
# generate simplest possible 'predictions' as a baseline
df['preds_acquisition_qc_general_benchmark'] = df[target].mean()

print('Benchmark model:', f.metrics.binary_metrics_text(df[target], df['preds_acquisition_qc_general_benchmark']))

# %% [markdown]
# ### single train/valid model

# %%
# split with simple 80-20 method
x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
    df.loc[:, features_all],
    df.loc[:, target],
    stratify=df.loc[:, target],
    test_size=0.2, random_state=42)

print(f'x train: {utils.df_shape(x_train).rjust(17)} | y_train:', f'{len(y_train):>9,} | mean {y_train.mean():,.6f}')
print(f'x valid: {utils.df_shape(x_valid).rjust(17)} | y_valid:', f'{len(y_valid):>9,} | mean {y_valid.mean():,.6f}')


# %%
# basic parameters, not too deep, some regularization with min_child_samples and 80% feature use
params_init = {
    'objective':'binary',  'learning_rate': 0.08, 'n_estimators': 2000, 'min_split_gain': 0.003,
    'num_leaves': 2**7-1, 'max_depth': 7, 'min_child_samples': 200, 'colsample_bytree': 0.8, 
    'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'n_jobs': N_CORES_ASSIGNED, 'random_state': 42, 
    'verbose': -1, #'force_col_wise':True
}
model_lgb = lightgbm.LGBMClassifier(**params_init)

model_lgb.fit(X=x_train, y=y_train,
              eval_set=(x_valid, y_valid),
              early_stopping_rounds=20, verbose=100);

y_pred = model_lgb.predict_proba(x_valid)[:,1]

print()
print('LightGBM model: ', f.metrics.binary_metrics_text(y_valid, y_pred))
print('Benchmark model:', f.metrics.binary_metrics_text(df[target], df['preds_acquisition_qc_general_benchmark']))


# %%
if IS_NOTEBOOK:
    plt.title(f'acquisition probability into QC - {GLOBAL_ENTITY_ID}, data until {DATE_UNTIL}')
    sns.histplot(pd.Series(y_pred), bins=50);


# %%
feat_imp = f.features.feat_importances_from_models([model_lgb], features_all)
feat_imp['feature'] = feat_imp['feature'].apply(lambda s: s.replace('ft_', '').split('__fillna_')[0])

if IS_NOTEBOOK:
    display(feat_imp
            .loc[#feat_imp['feature'].str.contains('device')
                 :
                 , ['feature', 'imp_mean']]
            #.query('imp_mean > 0.02')
            .head(30)
            .style.bar(color='#93a2be')
           )

# %% [markdown]
# ### shapley values

# %%
# takes around 30s
sample_size = 500
print(f'creating TreeExplainer')
explainer = shap.TreeExplainer(
    model=model_lgb,
    data=x_valid[:sample_size], # will run a lot slower when passing data
    #data=x_valid_target.loc[x_valid_target['pred']>0.99, features_all],
    #feature_perturbation='interventional',
    model_output='probability',
    #model_output='raw_value'
)

shap_values = explainer.shap_values(X=x_valid[:sample_size], y=None)

shap_values_1 = explainer(x_valid[:sample_size])


# %%
shap_values_df = pd.DataFrame(shap_values, columns=features_all)


# %%
try:    shap_summary_plot(shap_values, x_valid[:sample_size], 20, target)
except: print('ERROR plotting shap values')


# %%
# for feat in feat_imp['feature'].head(10):
#     shap.plots.scatter(shap_values_1[:,feat])

# %% [markdown]
# ### overfit check

# %%
x_valid_target = x_valid.copy()
x_valid_target['target'] = y_valid
x_valid_target['pred']   = y_pred


# %%
x_valid_target['pred'].hist(bins=50);


# %%
y_valid.sum()


# %%
overfit = x_valid_target.loc[
    x_valid_target['pred']>0.999
    ,['pred', 'target', 'days_since_first_order_vert_rs', 'gmv_sum_l04w_vert_rs', 
      'order_count_l04w_vert_rs', 'order_count_l16w_vert_rs', 'rating_avg_l04w',
      'dist_nearest_dmart', 
     ]
]

normal = x_valid_target.loc[
    (x_valid_target['pred']>0.2) & (x_valid_target['pred']<0.21)
    ,['pred', 'target', 'days_since_first_order_vert_rs', 'gmv_sum_l04w_vert_rs',
      'order_count_l04w_vert_rs', 'order_count_l16w_vert_rs', 'rating_avg_l04w',
      'dist_nearest_dmart', 
     ]
]
print(overfit.shape, normal.shape)


# %%
# overfit.sample(25)


# %%
normal.head(5)

# %% [markdown]
# ### permutation importance
# currently requires no missing values, see [PR](https://github.com/eli5-org/eli5/pull/5)

# %%
if RUN_PERMUATION_IMP:
    # takes a few minutes, doesn't work with NaN, unfortunately
    print('permutation importance: calculating...')
    perm_imp = f.features.permutation_importance(model_lgb, x_valid[:100_000].fillna(-1), y_valid[:100_000], n_iter=1)

    feature_selection = list(perm_imp.query('perm_imp>0')['feature'].values)
    # features_relevant
    print(f'features: {len(features_relevant)}/{len(features_all)}')
    
    perm_imp.style.background_gradient()
else:
    print('permutation importance: deactivated')
    feature_selection = features_all

# %% [markdown]
# ### out of fold predictions
# %% [markdown]
# #### Logistic Regression
# has some bugs with the solver currently

# %%
# params_init = {
#     'random_state':42
# }

# model = linear_model.LogisticRegression(**params_init)

# y_pred_classes = model_selection.cross_val_predict(
#     estimator=model, 
#     method='predict_proba',
#     X=df[features_relevant].fillna(-1), # works best with -1, need to find a better imputation
#     y=df[target], 
#     cv=8, n_jobs=4, verbose=1)


# %%
# y_pred_1 = y_pred_classes[:,1]
# df['preds_oof_acquisition_qc__0_1__logreg'] = y_pred_1

# print(f.metrics.binary_metrics_text(df[target], df['preds_oof_acquisition_qc__0_1__logreg']))


# %%
# params_init = {
#     'random_state':42
# }

# df, models = sl.learn.predict_out_of_fold_sklearn(
#     df.fillna(-1), train_index=df.index, 
#     predict_index=None, # complete oof-predictions, no pred-only data
#     target='target_acquisition_qc__0_1', 
#     features=features_relevant,
#     n_splits=5,
#     preds_oof_col_suffix='acquisition_qc__0_1__logreg', 
#     est=linear_model.LogisticRegression,
#     predict_method='predict_proba',
#     model_init_params=params_init
# )

# %% [markdown]
# #### LightGBM
# %% [markdown]
# Create out of fold predictions with custom functions.
# Uses less memory and returns the used models for further usage.

# %%
print('generating predictions for trial probability...')
params_init = {
    'objective':'binary',  'learning_rate': 0.10, 'n_estimators': 2000, 'min_split_gain': 0.005,
    'num_leaves': 2**7-1, 'max_depth': 7, 'min_child_samples': 200, 'colsample_bytree': 0.8, 
    'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'n_jobs': N_CORES_ASSIGNED, 'random_state': 42, 
    'verbose': -1, #'force_col_wise':True
}

df, models = f.models.predict_out_of_fold_sklearn(
    df, train_index=df.index, 
    predict_index=None, # complete oof-predictions, no pred-only data
    target=target, 
    features=feature_selection,
    n_splits=5,
    preds_oof_col_suffix='acquisition_qc_general_0_1_lgb', 
    est=lightgbm.LGBMClassifier, predict_method='predict_proba',
    model_init_params=params_init,
    model_fit_params={'verbose':50})


# %%
df[target]
# df['preds_oof_acquisition_qc_general_0_1_lgb']


# %%
print(f.metrics.binary_metrics_text(df[target], df['preds_oof_acquisition_qc_general_0_1_lgb']))
# metrics_acq_gen = pd.DataFrame(data=f.metrics.binary_metrics(df[target], df['preds_oof_acquisition_qc_general_0_1_lgb']), index=[0])
# display(metrics_acq_gen)

# pd.crosstab(index=  df['target_acquisition_qc_general_0_1'],
#             columns=df['preds_oof_acquisition_qc_general_0_1_lgb'].apply(round),
#                          values='analytical_customer_id', aggfunc='count')


# %%
feat_imp = f.features.feat_importances_from_models(models, feature_selection)
feat_imp['feature'] = feat_imp['feature'].apply(lambda s: s.replace('ft_', '').split('__fillna_')[0])

utils.df_to_gs(feat_imp, bucket_gs, f'{DIR_TOPIC}feat_imp/entity={GLOBAL_ENTITY_ID}/target={target}/date={DATE_UNTIL}/feat_imp.parquet', verbose=False)

if IS_NOTEBOOK:
    display(feat_imp
            .loc[:, ['feature', 'imp_mean']]
            #.query('imp_mean > 0.02')
            .head(10)
            .style.bar(color='#93a2be')
           )


# %%
if IS_NOTEBOOK:
    plt.title(f'acquisition probability into QC - {GLOBAL_ENTITY_ID}, data until {DATE_UNTIL}')
    sns.histplot(df['preds_oof_acquisition_qc_general_0_1_lgb'], bins=50);

# %% [markdown]
# ### final

# %%
# combine predictions from multiple models.. add other predictions if more models are used

#df['preds_oof_acquisition_qc__0_1__final'] = ( df['preds_oof_acquisition_qc__0_1__lgb'] * 0.9
#                                              +df['preds_oof_acquisition_qc__0_1__logreg'] * 0.1)
df['preds_'+'acquisition_qc_general_0_1'+'_final'] = ( df['preds_oof_acquisition_qc_general_0_1_lgb']#*0.5
                                                      #+df['preds_oof_acquisition_qc_0_1_lgb']*0.5
                                                     )

#print('LogReg:    ', f.metrics.binary_metrics_text(df[target], df['preds_oof_acquisition_qc__0_1__logreg']))
print('LightGBM 1:', f.metrics.binary_metrics_text(df[target], df['preds_oof_acquisition_qc_general_0_1_lgb']))
#print('LightGBM 2:', f.metrics.binary_metrics_text(df[target], df['preds_oof_acquisition_qc_0_1_lgb2']))
#print('combined:  ', f.metrics.binary_metrics_text(df[target], df['preds_'+'acquisition_qc_general_0_1'+'_final']))

# %% [markdown]
# ### plots

# %%
if IS_NOTEBOOK:
    plt.title(f'distribution actual vs predicted: {GLOBAL_ENTITY_ID}, data until {DATE_UNTIL}')
    sns.histplot(df, x='preds_oof_acquisition_qc_general_0_1_lgb', bins=50, hue=target, kde=False);


# %%
p33 = df.loc[df['target_acquisition_qc_general_0_1']==0,'preds_oof_acquisition_qc_general_0_1_lgb'].quantile(.333)
p66 = df.loc[df['target_acquisition_qc_general_0_1']==0,'preds_oof_acquisition_qc_general_0_1_lgb'].quantile(.666)
print(p33)
print(p66)


# %%
if IS_NOTEBOOK:
    plt.title(f'distribution actual vs predicted: {GLOBAL_ENTITY_ID}, data until {DATE_UNTIL}')
    sns.histplot(df.loc[df['target_acquisition_qc_general_0_1']==0,:], x='preds_oof_acquisition_qc_general_0_1_lgb', bins=50, hue=target, kde=False, legend=False);


# %%
df['preds_oof_acquisition_qc_general_0_1_lgb_percentiles'] = np.random.randint(1,10000, size=len(df))/10000

if IS_NOTEBOOK:
    plt.title(f'distribution actual vs predicted: {GLOBAL_ENTITY_ID}, data until {DATE_UNTIL}')
    sns.histplot(df.loc[df['target_acquisition_qc_general_0_1']==0,:], x='preds_oof_acquisition_qc_general_0_1_lgb_percentiles', bins=50, hue=target, kde=False, legend=False);


# %%
# sns.histplot(df.sample(10000), x='order_hour_avg_l16w_vert_rs', bins=24*2, 
#              hue='target_acquisition_qc_general_0_1', kde=False);

# %% [markdown]
# ## organic acquisition

# %%
print('='*12, 'organic vs paid trial', '='*12)
df, target = f.targets.create_acquisition_organic(df)

utils.display_value_counts(df[target])


# %%
index_train = df.loc[df[target].notnull(), :].index
index_pred  = df.loc[df[target].isnull(),  :].index
print(f'train: {len(index_train):,} pred: {len(index_pred):,}')

# %% [markdown]
# ### benchmark model

# %%
# generate simplest possible 'predictions' as a baseline
avg = df.loc[df['order_count_lifetime_vert_qc']>0, target].mean()
df['preds_acquisition_qc_organic_0_1_benchmark'] = avg

print('benchmark model:', f.metrics.binary_metrics_text(
    df.loc[df['order_count_lifetime_vert_qc']>0, target],
    df.loc[df['order_count_lifetime_vert_qc']>0, 'preds_acquisition_qc_organic_0_1_benchmark']))

# %% [markdown]
# ### single train/valid model

# %%
# split with simple 80-20 method
x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
    df.loc[index_train, features_all],
    df.loc[index_train, target],
    stratify=df.loc[index_train, target],
    test_size=0.2, random_state=42)

print(f'x train: {utils.df_shape(x_train).rjust(16)} | y_train:', f'{len(y_train):>9,} | mean {y_train.mean():,.6f}')
print(f'x valid: {utils.df_shape(x_valid).rjust(16)} | y_valid:', f'{len(y_valid):>9,} | mean {y_valid.mean():,.6f}')


# %%
# basic parameters, not too deep, some regularization with min_child_samples and 80% feature use
params_init = {
    'objective':'binary',  'learning_rate': 0.05, 'n_estimators': 2000,
    'num_leaves': 2**7-1, 'max_depth': 7, 'min_child_samples': 100, 'colsample_bytree': 0.8, 
    'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'n_jobs': N_CORES_ASSIGNED, 'random_state': 42, 'verbose': -1
}
model_lgb = lightgbm.LGBMClassifier(**params_init)

model_lgb.fit(X=x_train, y=y_train,
              eval_set=(x_valid, y_valid),
              early_stopping_rounds=20, verbose=False);

y_pred = model_lgb.predict_proba(x_valid)[:,1]
y_pred_01 = np.round(y_pred, 0)

print('LightGBM model: ', f.metrics.binary_metrics_text(y_valid, y_pred))
print('benchmark model:', f.metrics.binary_metrics_text(
    df.loc[df['order_count_lifetime_vert_qc']>0, target],
    df.loc[df['order_count_lifetime_vert_qc']>0, 'preds_acquisition_qc_organic_0_1_benchmark']))

# %% [markdown]
# ### shapley values

# %%
sample_size = 500
print(f'creating TreeExplainers')
explainer = shap.TreeExplainer(
    model=model_lgb,
    data=x_valid[:sample_size], # will run a lot slower when passing data
    #feature_perturbation='interventional',
    model_output='probability',
    #model_output='raw_value'
)

shap_values = explainer.shap_values(X=x_valid[:sample_size], y=None)

try: shap_summary_plot(shap_values, x_valid[:sample_size], 20, target)
except: print('ERROR plotting shap values')

# %% [markdown]
# ### out of fold
# %% [markdown]
# #### LightGBM
# %% [markdown]
# Create out of fold predictions with custom functions.
# Uses less memory and returns the used models for further usage.

# %%
print('generating out-of-fold predictions for organic vs paid probability...')
params_init = {
    'objective':'binary',  'learning_rate': 0.05, 'n_estimators': 1000,
    'num_leaves': 2**7-1, 'max_depth': 7, 'min_child_samples': 100, 'colsample_bytree': 0.8, 
    'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'n_jobs': N_CORES_ASSIGNED, 'random_state': 42, 'verbose': -1
}

# train on 5-split actual data, predict on out-of-fold actual data + open customers
df, models = f.models.predict_out_of_fold_sklearn(
    df,
    train_index=index_train, 
    predict_index=index_pred,
    target=target, 
    features=features_all,
    n_splits=5,
    preds_oof_col_suffix='acquisition_qc_organic_0_1_lgb', 
    est=lightgbm.LGBMClassifier, predict_method='predict_proba',
    model_init_params=params_init,
    model_fit_params={'verbose':50})


# %%
metrics_acq_org = pd.DataFrame(data=f.metrics.binary_metrics(df.loc[index_train, target],
                                                             df.loc[index_train, 'preds_oof_acquisition_qc_organic_0_1_lgb']), index=[0])
display(metrics_acq_org)
display(pd.crosstab(index=  df.loc[index_train, 'target_acquisition_qc_organic_0_1'],
                    columns=df.loc[index_train, 'preds_oof_acquisition_qc_organic_0_1_lgb'].apply(round),
                    values='analytical_customer_id', aggfunc='count'))


# %%
if IS_NOTEBOOK: sns.histplot(df, x='preds_oof_acquisition_qc_organic_0_1_lgb', bins=100, 
                             #hue=target,
                             kde=False);


# %%
feat_imp = f.features.feat_importances_from_models(models, features_all)
feat_imp['feature'] = feat_imp['feature'].apply(lambda s: s.replace('ft_', '').split('__fillna_')[0])

utils.df_to_gs(feat_imp, bucket_gs, f'{DIR_TOPIC}feat_imp/entity={GLOBAL_ENTITY_ID}/target={target}/date={DATE_UNTIL}/feat_imp.parquet', verbose=False)

if IS_NOTEBOOK: 
    display(feat_imp
            .loc[:, ['feature', 'imp_mean']]
            #.query('imp_mean > 0.02')
            .head(10)
            .style.bar(color='#93a2be')
           )


# %%
# final prediction, add other models (ensemble) in the future
df['preds_'+'acquisition_qc_organic_0_1'+'_final'] = (df['preds_oof_acquisition_qc_organic_0_1_lgb']
                                                      #+df['preds_oof_acquisition_qc_organic_0_1_logreg']
                                                     )#/2
print('rows with missing predictions:', df['preds_'+'acquisition_qc_organic_0_1'+'_final'].isnull().sum())

# %% [markdown]
# ## order frequency
# Predict monthly order frequency after acquisition

# %%
print('='*12, 'order frequency', '='*12)
df, target = f.targets.create_acquisition_order_freq_4w_qc(df)

print('\ndistribution of actual frequencies:', end='')
display(df[target].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))


# %%
# Create train and prediction index. Train on QC customers, active >=7d. Predict on non-active customers.
index_train = df.loc[ (df['order_count_lifetime_vert_qc']> 1)
                     &(df['first_order_date_vert_qc']<pd.Timestamp(DATE_UNTIL_MINUS_28))
                     &(df[target].notnull()),  :].index

index_pred  = df.loc[(df['order_count_lifetime_vert_qc']<=1),  :].index

print(f'train: {len(index_train):,} | predict: {len(index_pred):,}')
print('missing targets:', df.loc[index_train, target].isnull().sum())


# %%
p99 = df[target].quantile(0.99)
if IS_NOTEBOOK: 
    plt.title(f'{target}, {GLOBAL_ENTITY_ID}, data until {DATE_UNTIL}')
    df.loc[index_train, target].clip(0, p99).hist(bins=int(p99));

# %% [markdown]
# ### benchmark model

# %%
# generate simplest possible 'predictions' as a baseline
avg = df.loc[df[target].notnull(), target].mean()
df['preds_order_freq_4w_vert_qc_benchmark'] = avg

print('benchmark model:', f.metrics.regression_metrics_text(
    df.loc[df[target].notnull(), target],
    df.loc[df[target].notnull(), 'preds_order_freq_4w_vert_qc_benchmark']))

# %% [markdown]
# ### single train/valid model

# %%
# if IS_NOTEBOOK:
#     display(df.loc[#index_train
#         df[target]==3.5
#         , ['first_order_date_vert_qc', 
#            'order_count_lifetime_vert_qc', 'order_freq_4w_vert_qc']].sample(10))                     


# %%
# split with simple 80-20 method
x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
    df.loc[index_train, features_all],
    df.loc[index_train, target],
    test_size=0.2, random_state=42)

print(f'x train: {utils.df_shape(x_train).rjust(16)} | y_train:', f'{len(y_train):>9,} | mean {y_train.mean():,.6f}')
print(f'x valid: {utils.df_shape(x_valid).rjust(16)} | y_valid:', f'{len(y_valid):>9,} | mean {y_valid.mean():,.6f}')


# %%
# basic parameters, not too deep, some regularization with min_child_samples and 80% feature use
params_init = {
    'learning_rate': 0.02, 'n_estimators': 2000,
    'num_leaves': 2**7-1, 'max_depth': 7, 'min_child_samples': 100, 'colsample_bytree': 0.8, 
    'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'n_jobs': N_CORES_ASSIGNED, 'random_state': 42, 
    'verbose': -1
}
model_lgb = lightgbm.LGBMRegressor(**params_init)

model_lgb.fit(X=x_train, y=y_train,
              eval_set=(x_valid, y_valid),
              early_stopping_rounds=20, verbose=50);

y_pred = model_lgb.predict(x_valid).clip(0.01, None)
best_iter = model_lgb.best_iteration_

print('LightGBM model: ', f.metrics.regression_metrics_text(y_valid, y_pred))
print('benchmark model:', f.metrics.regression_metrics_text(
    df.loc[df[target].notnull(), target],
    df.loc[df[target].notnull(), 'preds_order_freq_4w_vert_qc_benchmark']))

metrics_freq = pd.DataFrame(data=f.metrics.regression_metrics(y_valid, y_pred), index=[0])
#display(metrics_freq)


# %%
feat_imp = f.features.feat_importances_from_models([model_lgb], features_all)
feat_imp['feature'] = feat_imp['feature'].apply(lambda s: s.replace('ft_', '').split('__fillna_')[0])

utils.df_to_gs(feat_imp, bucket_gs, f'{DIR_TOPIC}feat_imp/entity={GLOBAL_ENTITY_ID}/target={target}/date={DATE_UNTIL}/feat_imp.parquet', verbose=False)

if IS_NOTEBOOK:
    display(feat_imp
            .loc[:, ['feature', 'imp_mean']]
            #.query('imp_mean > 0.02')
            .head(10)
            .style.bar(color='#93a2be')
           )

# %% [markdown]
# ### shapley values

# %%
sample_size = 500
print(f'creating TreeExplainer')
explainer = shap.TreeExplainer(
    model=model_lgb,
    data=x_valid[:sample_size], # will run a lot slower when passing data
    #feature_perturbation='interventional',
    #model_output='predict_proba',
    #model_output='raw_value'
)

shap_values = explainer.shap_values(X=x_valid[:sample_size], y=None,
                                    check_additivity=False # creates errors otherwise
                                   )


# %%
try: shap_summary_plot(shap_values, x_valid[:sample_size], 20, target)
except: print('ERROR plotting shap values')

# %% [markdown]
# ### on non-acquired customers

# %%
# basic parameters, same iterations as previous model
print(f'training on {len(index_train):,} acquired and predicting on {len(index_pred):,} non-acquired customers')
params_init = {
    'learning_rate': 0.02, 'n_estimators': best_iter+10,
    'num_leaves': 2**7-1, 'max_depth': 7, 'min_child_samples': 100, 'colsample_bytree': 0.8, 
    'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'n_jobs': N_CORES_ASSIGNED, 'random_state': 42, 
    'verbose': -1
}
model_lgb = lightgbm.LGBMRegressor(**params_init)

model_lgb.fit(X=df.loc[index_train, features_all],
              y=df.loc[index_train, target],
              verbose=50);

y_pred = model_lgb.predict(df.loc[index_pred, features_all])


# %%
if IS_NOTEBOOK:
    p99 = pd.Series(y_pred).quantile(0.99)
    pd.Series(y_pred).clip(0,p99).hist(bins=100);


# %%
df.loc[index_pred, 'preds_'+'order_freq_4w_vert_qc'] = y_pred

# insert additional logic (ensembles) later
df['preds_'+'order_freq_4w_vert_qc'+'_final'] = df['preds_'+'order_freq_4w_vert_qc']

# %% [markdown]
# ## order value - check 0 values
# Predict average order value after acquisition

# %%
print('='*12, 'average order value', '='*12)

if df.loc[df['order_count_lifetime_vert_qc']>0,
          'gmv_avg_lifetime_vert_qc'].isnull().sum():
    raise Exception('missing GMV AVG on vert QC')

df, target = f.targets.create_acquisition_gmv_avg_qc(df)

print('\ndistribution of actual frequencies:', end='')
display(df[target].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))


# %%
index_train = df.loc[ (df['order_count_lifetime_vert_qc']> 1)
                     &(df['first_order_date_vert_qc']<pd.Timestamp(DATE_UNTIL_MINUS_28))
                     &(df[target].notnull()),  :].index

index_pred  = df.loc[(df['order_count_lifetime_vert_qc']<=1),  :].index

print(f'train: {len(index_train):,} | predict: {len(index_pred):,}')
print('missing targets:', df.loc[index_train, target].isnull().sum())


# %%
p99 = df[target].quantile(0.99)
if IS_NOTEBOOK: df.loc[index_train, target].clip(0, p99).hist(bins=int(p99)*2);

# %% [markdown]
# ### benchmark model

# %%
# generate simplest possible 'predictions' as a baseline
avg = df.loc[df[target].notnull(), target].mean()
df['preds_gmv_avg_lifetime_vert_qc_benchmark'] = avg

print('benchmark model:', f.metrics.regression_metrics_text(
    df.loc[df[target].notnull(), target],
    df.loc[df[target].notnull(), 'preds_gmv_avg_lifetime_vert_qc_benchmark']))

# %% [markdown]
# ### single train/valid model

# %%
# split with simple 80-20 method
x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
    df.loc[index_train, features_all],
    df.loc[index_train, target],
    test_size=0.2, random_state=42)

print(f'x train: {utils.df_shape(x_train).rjust(16)} | y_train:', f'{len(y_train):>9,} | mean {y_train.mean():,.6f}')
print(f'x valid: {utils.df_shape(x_valid).rjust(16)} | y_valid:', f'{len(y_valid):>9,} | mean {y_valid.mean():,.6f}')


# %%
# basic parameters, not too deep, some regularization with min_child_samples and 80% feature use
params_init = {
    'learning_rate': 0.02, 'n_estimators': 2000,
    'num_leaves': 2**7-1, 'max_depth': 7, 'min_child_samples': 100, 'colsample_bytree': 0.8, 
    'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'n_jobs': N_CORES_ASSIGNED, 'random_state': 42, 
    'verbose': -100
}
model_lgb = lightgbm.LGBMRegressor(**params_init)

model_lgb.fit(X=x_train, y=y_train,
              eval_set=(x_valid, y_valid),
              early_stopping_rounds=20, verbose=50);

y_pred = model_lgb.predict(x_valid)
best_iter = model_lgb.best_iteration_

print('best iteration:', best_iter)
print('LightGBM model: ', f.metrics.regression_metrics_text(y_valid, y_pred))
print('benchmark model:', f.metrics.regression_metrics_text(
    df.loc[df[target].notnull(), target],
    df.loc[df[target].notnull(), 'preds_gmv_avg_lifetime_vert_qc_benchmark']))

metrics_aov = pd.DataFrame(data=f.metrics.regression_metrics(y_valid, y_pred), index=[0])
#display(metrics_aov)


# %%
# LightGBM model:  mse: 83.3652	rmse: 9.1305	mae: 4.8708	r2: 0.1230	y_pred_<=0: 0.0000	y_pred_max: 53.8100	std: 3.4669	2021-05-24 18:02:26
# LightGBM model:  mse: 81.8974	rmse: 9.0497	mae: 4.7790	r2: 0.1384	y_pred_<=0: 0.0000	y_pred_max: 47.2400	std: 3.5837	2021-05-24 19:50:59


# %%
feat_imp = f.features.feat_importances_from_models([model_lgb], features_all)
feat_imp['feature'] = feat_imp['feature'].apply(lambda s: s.replace('ft_', '').split('__fillna_')[0])

utils.df_to_gs(feat_imp, bucket_gs, f'{DIR_TOPIC}feat_imp/entity={GLOBAL_ENTITY_ID}/target={target}/date={DATE_UNTIL}/feat_imp.parquet', verbose=False)

if IS_NOTEBOOK:
    display(feat_imp
            .loc[:, ['feature', 'imp_mean']]
            #.query('imp_mean > 0.02')
            .head(10)
            .style.bar(color='#93a2be')
           )

# %% [markdown]
# ### shapley values

# %%
sample_size = 500
print(f'creating TreeExplainer')
explainer = shap.TreeExplainer(
    model=model_lgb,
    data=x_valid[:sample_size], # will run a lot slower when passing data
    #feature_perturbation='interventional',
    #model_output='predict_proba',
    #model_output='raw_value',
    check_additivity=False
)


# %%
# there are some weird behaviours.. check if this works, then continue with the next one as well
works = False
try:
    shap_values = explainer.shap_values(X=x_valid[:sample_size], y=None)
    works = True
except Exception as e:
    print(e)
#shap_values_1 = explainer(x_valid[:sample_size])


# %%
if works:
    try:
        shap_summary_plot(shap_values, x_valid[:sample_size], 20, target)
    except:
        print('ERROR plotting shap values')
else:
    print('cant')


# %%
# for feat in feat_imp['feature'].head(10):
#     shap.plots.scatter(shap_values_1[:,feat])

# %% [markdown]
# ### on non-acquired customers

# %%
# basic parameters, same iterations as previous model
print(f'training on {len(index_train):,} acquired and predicting on {len(index_pred):,} non-acquired customers')
params_init = {
    'learning_rate': 0.02, 'n_estimators': best_iter+10,
    'num_leaves': 2**7-1, 'max_depth': 7, 'min_child_samples': 100, 'colsample_bytree': 0.8, 
    'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'n_jobs': N_CORES_ASSIGNED, 'random_state': 42, 
    'verbose': -1
}
model_lgb = lightgbm.LGBMRegressor(**params_init)

model_lgb.fit(X=df.loc[index_train, features_all],
              y=df.loc[index_train, target],
              verbose=50);

y_pred = model_lgb.predict(df.loc[index_pred, features_all])

print(f'y_pred: mean {y_pred.mean():.2f}, std {y_pred.std():.2f}, min {y_pred.min():.2f}, max {y_pred.max():.2f}')


# %%
# regular predictions and percentiles
df.loc[index_pred, 'preds_'+'gmv_avg_lifetime_vert_qc'] = y_pred

# insert additional logic (ensemble) later
df['preds_'+'gmv_avg_lifetime_vert_qc'+'_final'] = df['preds_'+'gmv_avg_lifetime_vert_qc']


# %%
if IS_NOTEBOOK: 
    p995 = pd.Series(y_pred).quantile(0.995)    
    pd.Series(y_pred).clip(0,p995).hist(bins=int(p995)*2);

# %% [markdown]
# # export
# %% [markdown]
# ## metrics

# %%
metrics_all = pd.concat(objs=[metrics_acq_gen, metrics_acq_org, metrics_freq, metrics_aov])
metrics_all.index = ['target_acquisition_qc_general_0_1', 'target_acquisition_qc_organic_0_1', 'target_order_freq_4w_vert_qc', 'target_gmv_avg_lifetime_vert_qc']
metrics_all.index.name = 'target'
metrics_all = metrics_all.reset_index()
display(metrics_all)


# %%
utils.df_to_gs(metrics_all, bucket_gs, f'{DIR_TOPIC}metrics/entity={GLOBAL_ENTITY_ID}/date={DATE_UNTIL}/metrics_all_targets.parquet', verbose=False)

# %% [markdown]
# ## table

# %%
target_cols = sorted([col for col in df.columns if col.startswith('target')])
pred_cols   = sorted([col for col in df.columns if col.startswith('preds') and col.endswith('final') and col!='preds_benchmark'])
print(f'found {len(target_cols)} targets, {(len(pred_cols))} prediction columns')

if IS_NOTEBOOK: 
    print(target_cols)
    print(pred_cols)


# %%
export = (df
          .loc[:, [
              'analytical_customer_id', *target_cols, *pred_cols,
           ]]
          .sort_values('analytical_customer_id') # sort for BigQuery optimization
          .reset_index(drop=True)
         )


# %%
for p in ['target_acquisition_qc_organic_0_1', 'target_gmv_avg_lifetime_vert_qc', 'target_order_freq_4w_vert_qc',
          'preds_acquisition_qc_general_0_1_final', 'preds_acquisition_qc_organic_0_1_final',
          'preds_gmv_avg_lifetime_vert_qc_final', 'preds_order_freq_4w_vert_qc_final']:
    export[p] = export[p].astype('float32')


# %%
if IS_NOTEBOOK: display(utils.df_info(export))


# %%
utils.df_to_gs(
    export, bucket_gs, 
    f'{DIR_TOPIC}predictions/entity={GLOBAL_ENTITY_ID}/date={DATE_UNTIL}/preds.parquet')


# %%
gc.collect()


