import pandas as pd
import numpy as np
from lib import load_data
#from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import datetime
from scipy.stats import multivariate_normal, norm

config = {
    'only_click_touch_type': False,
    'standard_scale': False
}

data, data_valid, labels = load_data(valid_size=0.2)
data.drop(data[data['Fraud_reasons'] == 'mix'].index, inplace=True)
data_valid.drop(data_valid[data_valid['Fraud_reasons'] == 'mix'].index, inplace=True)

data.loc[data['Fraud_reasons'].isnull(), 'Fraud_reasons'] = 'ok'
data_valid.loc[data_valid['Fraud_reasons'].isnull(), 'Fraud_reasons'] = 'ok'
fraud_reasons = data['Fraud_reasons'].unique()

data['install_time'] = pd.to_datetime(data['install_time'])
data_valid['install_time'] = pd.to_datetime(data_valid['install_time'])
data['attributed_touch_time'] = pd.to_datetime(data['attributed_touch_time'])
data_valid['attributed_touch_time'] = pd.to_datetime(data_valid['attributed_touch_time'])
data['contributor_1_touch_time'] = pd.to_datetime(data['contributor_1_touch_time'], errors='coerce')
data['contributor_2_touch_time'] = pd.to_datetime(data['contributor_2_touch_time'], errors='coerce')
data['contributor_3_touch_time'] = pd.to_datetime(data['contributor_3_touch_time'], errors='coerce')
data_valid['contributor_1_touch_time'] = pd.to_datetime(data_valid['contributor_1_touch_time'], errors='coerce')
data_valid['contributor_2_touch_time'] = pd.to_datetime(data_valid['contributor_2_touch_time'], errors='coerce')
data_valid['contributor_3_touch_time'] = pd.to_datetime(data_valid['contributor_3_touch_time'], errors='coerce')

android_versions = pd.read_csv('android_versions.csv', delimiter=',')
android_versions['release_date'] = pd.to_datetime(android_versions['release_date'])
android_versions['version'] = android_versions['version'].apply(lambda x: int(x.replace('.', '00')))
data = data.merge(android_versions, how='left', left_on='app_version', right_on='version')
data_valid = data_valid.merge(android_versions, how='left', left_on='app_version', right_on='version')

ios_app_id = 'id966165025'
android_app_id = 'ng.jiji.app'
data.loc[data['app_id'] == ios_app_id, 'release_date'] = np.datetime64('NaT')
data_valid.loc[data_valid['app_id'] == ios_app_id, 'release_date'] = np.datetime64('NaT')

extracted_features_columns = np.array([])

def estimate_gaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma

def multivariate_gaussian(dataset, mu, sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)

def estimate_gaussian_series(series):
    mu = series[~series.isnull()].mean()
    sigma = series[~series.isnull()].std()
    return mu, sigma

def univariate_gaussian(series, mu, sigma):
    p = norm(loc=mu, scale=sigma)
    return p.pdf(series)
    #return norm.rvs(loc=0, scale=1, size=len(series))

if config['only_click_touch_type']:
    data = data[data['attributed_touch_type'] == 'click']

def set_time_features(data, data_valid):
    data['install_time_sec'] = (data['install_time'] - pd.datetime.now()).dt.total_seconds()
    data['release_date_sec'] = (data['release_date'] - pd.datetime.now()).dt.total_seconds()

    data_valid['install_time_sec'] = (data_valid['install_time'] - pd.datetime.now()).dt.total_seconds()
    data_valid['release_date_sec'] = (data_valid['release_date'] - pd.datetime.now()).dt.total_seconds()

    data['tti'] = (data['install_time']
                   - data['attributed_touch_time']).dt.total_seconds()
    data_valid['tti'] = (pd.to_datetime(data_valid['install_time'])
                         - data_valid['attributed_touch_time']).dt.total_seconds()

    data['time_diff_contributor1'] = (data['attributed_touch_time']
                                      - data['contributor_1_touch_time']).dt.total_seconds()
    data['time_diff_contributor2'] = (data['contributor_1_touch_time']
                                      - data['contributor_2_touch_time']).dt.total_seconds()
    data['time_diff_contributor3'] = (data['contributor_2_touch_time']
                                      - data['contributor_3_touch_time']).dt.total_seconds()

    data_valid['time_diff_contributor1'] = (data_valid['attributed_touch_time']
                                            - data_valid['contributor_1_touch_time']).dt.total_seconds()
    data_valid['time_diff_contributor2'] = (data_valid['contributor_1_touch_time']
                                            - data_valid['contributor_2_touch_time']).dt.total_seconds()
    data_valid['time_diff_contributor3'] = (data_valid['contributor_2_touch_time']
                                            - data_valid['contributor_3_touch_time']).dt.total_seconds()

    extracted_features_columns = np.array(['tti', 'release_date_sec', 'install_time_sec',
                                          'time_diff_contributor1', 'time_diff_contributor2', 'time_diff_contributor3'])

    return data, data_valid, extracted_features_columns

def set_fraud_rate_feature(train_df, valid_df, column, mean_fraud, fraud_reason):
    rate_feature_column = column + '_' + fraud_reason + '_fraud_rate'
    ratio = train_df[['Fraud_reasons', column]].groupby([column]).agg(
        lambda x: (x['Fraud_reasons'] == fraud_reason).sum()/len(x))
    ratio.rename(columns={'Fraud_reasons': rate_feature_column}, inplace=True)

    train_df = pd.merge(train_df, ratio, how='left', left_on=column, right_index=True)
    train_df[rate_feature_column] = train_df[rate_feature_column].fillna(mean_fraud)
    # train_df[rate_feature_column] = train_df[rate_feature_column].fillna(0)

    valid_df = pd.merge(valid_df, ratio, how='left', left_on=column, right_index=True)
    valid_df[rate_feature_column] = valid_df[rate_feature_column].fillna(mean_fraud)
    # valid_df[rate_feature_column] = valid_df[rate_feature_column].fillna(0)
    return train_df, valid_df, rate_feature_column

fraud_rates_columns = ['language', 'publisher', 'operator', 'device_type',
                       'os_version', 'sdk_version', 'app_id', 'app_version', 'wifi']

data, data_valid, time_features = set_time_features(data, data_valid)
extracted_features_columns = np.concatenate((extracted_features_columns, time_features))

def set_categorical_fetures(data, data_valid):
    data['wifi'] = data['wifi'].astype(str)
    data_valid['wifi'] = data_valid['wifi'].astype(str)
    cat_features = np.array(['language', 'publisher', 'operator', 'device_type',
                             'os_version', 'sdk_version', 'app_id', 'app_version', 'wifi'])

    return data, data_valid, cat_features

data, data_valid, cat_features = set_categorical_fetures(data, data_valid)
extracted_features_columns = np.concatenate((extracted_features_columns, cat_features))

"""
def set_distribution_features(data, data_valid):
    # trustworthy_publisher_marker = data['publisher'].isin(['AX', 'AG'])
    trustworthy_publisher_marker = data['Fraud_reasons'] == 'ok'
    distribution_features = np.array([])
    features = ['tti', 'release_date_sec', 'install_time_sec',
                'time_diff_contributor1', 'time_diff_contributor2', 'time_diff_contributor3']

    fraud_reasons_without_ok = np.array([x for x in fraud_reasons if x != 'ok'])
    for feature in features:
        mu, sigma = estimate_gaussian_series(data.loc[trustworthy_publisher_marker, feature])
        data[feature + '_p_ok'] = univariate_gaussian(data[feature], mu, sigma)
        data_valid[feature + '_p_ok'] = univariate_gaussian(data_valid[feature], mu, sigma)
        distribution_features = np.concatenate((distribution_features, np.array([feature + '_p_ok'])))

        for fraud_reason in fraud_reasons_without_ok:
            mu, sigma = estimate_gaussian_series(data.loc[data['Fraud_reasons'] == fraud_reason, feature])
            data[feature + '_p_' + fraud_reason] = univariate_gaussian(data[feature], mu, sigma)
            data_valid[feature + '_p_' + fraud_reason] = univariate_gaussian(data_valid[feature], mu, sigma)
            distribution_features = np.concatenate((distribution_features, np.array([feature + '_p_' + fraud_reason])))

    return data, data_valid, distribution_features

data, data_valid, distribution_features = set_distribution_features(data, data_valid)
extracted_features_columns = np.concatenate((extracted_features_columns, distribution_features))
"""

extracted_features_columns = np.concatenate((extracted_features_columns, np.array(['Fraud_reasons'])))
extracted_features_train = data[extracted_features_columns]
extracted_features_train.to_csv('extracted_features_train.csv', sep=';', index=False, encoding='utf-8')

extracted_features_valid = data_valid[extracted_features_columns]
extracted_features_valid.to_csv('extracted_features_valid.csv', sep=';', index=False, encoding='utf-8')

