import pandas as pd
import numpy as np
from lib import load_data
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import datetime
from scipy.stats import multivariate_normal, norm

config = {
    'only_click_touch_type': False,
    'standard_scale': True
}

data, data_valid, labels = load_data(valid_size=0.2)
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

if config['only_click_touch_type']:
    data = data[data['attributed_touch_type'] == 'click']

def set_time_features(data, data_valid):
    data['tti'] = (data['install_time']
                               - data['attributed_touch_time']).dt.total_seconds()
    data_valid['tti'] = (pd.to_datetime(data_valid['install_time'])
                               - data_valid['attributed_touch_time']).dt.total_seconds()

    data['install_time_since_midnight_hours'] = (data['install_time'] -
                                               pd.to_datetime(data['install_time'].dt.date)) / np.timedelta64(1, 's')
    data_valid['install_time_since_midnight_hours'] = data_valid['install_time'].dt.hour

    # data['install_time_since_midnight_sec'] = (data['install_time'] -
    #                                            pd.to_datetime(data['install_time'].dt.date)) / np.timedelta64(1, 's')
    # data_valid['install_time_since_midnight_sec'] = (data_valid['install_time'] -
    #                                            pd.to_datetime(data_valid['install_time'].dt.date)) / np.timedelta64(1, 's')

    # data['attributed_touch_time_since_midnight_sec'] = (data['attributed_touch_time'] -
    #                                            pd.to_datetime(data['attributed_touch_time'].dt.date)) / np.timedelta64(1, 's')
    # data_valid['attributed_touch_time_since_midnight_sec'] = (data_valid['attributed_touch_time'] -
    #                                                     pd.to_datetime(
    #                                                         data_valid['attributed_touch_time'].dt.date)) / np.timedelta64(1,
    #                                                                                                                  's')

    # data['attributed_touch_time_date_hour'] = data['attributed_touch_time'].apply(
    #     lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour))
    # data_valid['attributed_touch_time_date_hour'] = data_valid['attributed_touch_time'].apply(
    #     lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour))
    # df = pd.DataFrame(index=data['attributed_touch_time_date_hour'].unique())
    # df['bot_frauds_per_hour'] = data.groupby('attributed_touch_time_date_hour')['Fraud_reasons'].agg({
    #     'Fraud_reasons': lambda x: (x == 'bots').sum()})
    # df['tti_frauds_per_hour'] = data.groupby('attributed_touch_time_date_hour')['Fraud_reasons'].agg({
    #     'Fraud_reasons': lambda x: (x == 'tti_fraud').sum()})
    # df['click_spamming_frauds_per_hour'] = data.groupby('attributed_touch_time_date_hour')['Fraud_reasons'].agg({
    #     'Fraud_reasons': lambda x: (x == 'click_spamming').sum()})
    # df['mix_frauds_per_hour'] = data.groupby('attributed_touch_time_date_hour')['Fraud_reasons'].agg({
    #     'Fraud_reasons': lambda x: (x == 'mix').sum()})
    # df['data_center_frauds_per_hour'] = data.groupby('attributed_touch_time_date_hour')['Fraud_reasons'].agg({
    #     'Fraud_reasons': lambda x: (x == 'Data center').sum()})
    #
    # # imitate real-life scenario -- we know only about events one hour ago
    # data['attributed_touch_time_date_hour'] = data['attributed_touch_time_date_hour'] - pd.DateOffset(hours=1)
    # data_valid['attributed_touch_time_date_hour'] = data_valid['attributed_touch_time_date_hour'] - pd.DateOffset(hours=1)
    # data = data.merge(df, left_on='attributed_touch_time_date_hour', right_index=True)
    # data_valid = data_valid.merge(df, left_on='attributed_touch_time_date_hour', right_index=True)

    # data['install_time_date_hour'] = data['install_time'].apply(
    #     lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour))
    # data_valid['install_time_date_hour'] = data_valid['install_time'].apply(
    #     lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour))
    # df = pd.DataFrame(index=data['install_time_date_hour'].unique())
    # df['bot_frauds_per_hour'] = data.groupby('install_time_date_hour')['Fraud_reasons'].agg({
    #     'Fraud_reasons': lambda x: (x == 'bots').sum()})
    # df['tti_frauds_per_hour'] = data.groupby('install_time_date_hour')['Fraud_reasons'].agg({
    #     'Fraud_reasons': lambda x: (x == 'tti_fraud').sum()})
    # df['click_spamming_frauds_per_hour'] = data.groupby('install_time_date_hour')['Fraud_reasons'].agg({
    #     'Fraud_reasons': lambda x: (x == 'click_spamming').sum()})
    # df['mix_frauds_per_hour'] = data.groupby('install_time_date_hour')['Fraud_reasons'].agg({
    #     'Fraud_reasons': lambda x: (x == 'mix').sum()})
    # df['data_center_frauds_per_hour'] = data.groupby('install_time_date_hour')['Fraud_reasons'].agg({
    #     'Fraud_reasons': lambda x: (x == 'Data center').sum()})
    #
    # # imitate real-life scenario -- we know only about events one hour ago
    # data['install_time_date_hour'] = data['install_time_date_hour'] - pd.DateOffset(hours=1)
    # data_valid['install_time_date_hour'] = data_valid['install_time_date_hour'] - pd.DateOffset(
    #     hours=1)
    # data = data.merge(df, left_on='install_time_date_hour', right_index=True)
    # data_valid = data_valid.merge(df, left_on='install_time_date_hour', right_index=True)

    data['install_time_weekday'] = data['install_time'].dt.weekday_name
    data_valid['install_time_weekday'] = data_valid['install_time'].dt.weekday_name

    # data['tti_contributor1'] = (data['install_time']
    #                             - data['contributor_1_touch_time']).dt.total_seconds()
    # data['tti_contributor2'] = (data['install_time']
    #                             - data['contributor_2_touch_time']).dt.total_seconds()
    # data['tti_contributor3'] = (data['install_time']
    #                             - data['contributor_3_touch_time']).dt.total_seconds()
    #
    # data_valid['tti_contributor1'] = (data_valid['install_time']
    #                             - data_valid['contributor_1_touch_time']).dt.total_seconds()
    # data_valid['tti_contributor2'] = (data_valid['install_time']
    #                             - data_valid['contributor_2_touch_time']).dt.total_seconds()
    # data_valid['tti_contributor3'] = (data_valid['install_time']
    #                             - data_valid['contributor_3_touch_time']).dt.total_seconds()

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

    data['day_diff_install_release'] = (data['install_time'] -
                                         data['release_date']).dt.total_seconds() // (3600*24)

    data_valid['day_diff_install_release'] = (data_valid['install_time'] -
                                              data_valid['release_date']).dt.total_seconds() // (3600*24)

    extracted_features_columns = np.array(['tti', 'install_time_since_midnight_hours',
                                           # 'bot_frauds_per_hour', 'tti_frauds_per_hour', 'click_spamming_frauds_per_hour','mix_frauds_per_hour', 'data_center_frauds_per_hour',
                                           'time_diff_contributor1', #, 'time_diff_contributor2', 'time_diff_contributor3',
                                           'install_time_weekday', 'day_diff_install_release',
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

def set_fraud_rate_features(data, data_valid, fraud_reason):
    mean_fraud = (data['Fraud_reasons'] == fraud_reason).mean()
    fraud_reason_features_columns = np.array([])
    for col in tqdm(fraud_rates_columns):
        data, data_valid, fraud_reason_features_column = set_fraud_rate_feature(data, data_valid, col, mean_fraud,
                                                                               fraud_reason)
        fraud_reason_features_columns = np.concatenate((fraud_reason_features_columns, np.array([fraud_reason_features_column])))

    return data, data_valid, fraud_reason_features_columns

data, data_valid, time_features = set_time_features(data, data_valid)
extracted_features_columns = np.concatenate((extracted_features_columns, time_features))

# for fraud_reason in labels['Fraud_reasons'].unique():
#     print(f'Extracting features for fraud reason: {fraud_reason}')
#     data, data_valid, fraud_reason_features_columns = set_fraud_rate_features(data, data_valid, fraud_reason)
#     data[fraud_reason + '_mean_rate'] = data[fraud_reason_features_columns].mean(axis=1)
#     data_valid[fraud_reason + '_mean_rate'] = data_valid[fraud_reason_features_columns].mean(axis=1)
#     fraud_reason_features_columns = np.concatenate((fraud_reason_features_columns, np.array([fraud_reason + '_mean_rate'])))
#     extracted_features_columns = np.concatenate((extracted_features_columns, fraud_reason_features_columns))

def set_categorical_fetures(data, data_valid):
    data['wifi'] = data['wifi'].astype(str)
    data_valid['wifi'] = data_valid['wifi'].astype(str)
    cat_features = np.array(['wifi'])

    return data, data_valid, cat_features

data, data_valid, cat_features = set_categorical_fetures(data, data_valid)
extracted_features_columns = np.concatenate((extracted_features_columns, cat_features))


def set_distribution_features(data, data_valid):
    trustworthy_publisher_marker = (data['publisher'] == 'AX') | (data['publisher'] == 'AG')
    distribution_features = np.array([])
    features = ['tti', 'day_diff_install_release', 'time_diff_contributor1', 'time_diff_contributor2', 'time_diff_contributor3']
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


if config['standard_scale']:
    scaler = StandardScaler()
    extracted_features_columns_float = data[extracted_features_columns].dtypes.index.values[
        (data[extracted_features_columns].dtypes == np.float) & (~data[extracted_features_columns].isnull().any().values)]
    scaler.fit(data[extracted_features_columns_float])
    data[extracted_features_columns_float] = scaler.transform(data[extracted_features_columns_float])
    data_valid[extracted_features_columns_float] = scaler.transform(data_valid[extracted_features_columns_float])

extracted_features_columns = np.concatenate((extracted_features_columns, np.array(['Fraud_reasons'])))
extracted_features_train = data[extracted_features_columns]
extracted_features_train.to_csv('extracted_features_train.csv', sep=';', index=False, encoding='utf-8')

extracted_features_valid = data_valid[extracted_features_columns]
extracted_features_valid.to_csv('extracted_features_valid.csv', sep=';', index=False, encoding='utf-8')

# time_features = np.concatenate((time_features, np.array(['Fraud_reasons'])))
# time_features_train = data[time_features]
# time_features_train.to_csv('time_features_train.csv', sep=';', index=False, encoding='utf-8')
# time_features_valid = data_valid[time_features]
# time_features_valid.to_csv('time_features_valid.csv', sep=';', index=False, encoding='utf-8')
