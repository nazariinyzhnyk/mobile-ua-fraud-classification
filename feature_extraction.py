import pandas as pd
import numpy as np
from lib import load_data

config = {
    'only_click_touch_type': False
}

data, data_valid, labels = load_data(valid_size=0.2)
data['is_fraud'] = ~data['Fraud_reasons'].isnull()
data_valid['is_fraud'] = ~data_valid['Fraud_reasons'].isnull()

if config['only_click_touch_type']:
    data = data[data['attributed_touch_type'] == 'click']

def set_time_features(data, data_valid):
    data['time_difference'] = (pd.to_datetime(data['install_time'])
                               - pd.to_datetime(data['attributed_touch_time'])).dt.total_seconds()
    data_valid['time_difference'] = (pd.to_datetime(data_valid['install_time'])
                               - pd.to_datetime(data_valid['attributed_touch_time'])).dt.total_seconds()
    return data, data_valid

data, data_valid = set_time_features(data, data_valid)

def set_fraud_rate_feature(train_df, valid_df, column, mean_fraud):
    rate_feature_column = column + '_fraud_rate'
    site_ratio = train_df[['is_fraud', column]].groupby([column]).mean().reset_index().rename(
        columns={'is_fraud': rate_feature_column})
    train_df = pd.merge(train_df, site_ratio, how='left', on=column)
    train_df[rate_feature_column] = train_df[rate_feature_column].fillna(mean_fraud)

    valid_df = pd.merge(valid_df, site_ratio, how='left', on=column)
    valid_df[rate_feature_column] = valid_df[rate_feature_column].fillna(mean_fraud)
    return train_df, valid_df

def set_fraud_rate_features(data, data_valid, fraud_reason):
    mean_fraud = data['is_fraud'].mean()
    data, data_valid = set_fraud_rate_feature(data, data_valid, 'site_id', mean_fraud)
    data, data_valid = set_fraud_rate_feature(data, data_valid, 'sub_site_id', mean_fraud)
    data, data_valid = set_fraud_rate_feature(data, data_valid, 'publisher', mean_fraud)
    data, data_valid = set_fraud_rate_feature(data, data_valid, 'operator', mean_fraud)
    data, data_valid = set_fraud_rate_feature(data, data_valid, 'city', mean_fraud)
    data, data_valid = set_fraud_rate_feature(data, data_valid, 'device_type', mean_fraud)
    data, data_valid = set_fraud_rate_feature(data, data_valid, 'os_version', mean_fraud)
    data, data_valid = set_fraud_rate_feature(data, data_valid, 'sdk_version', mean_fraud)

    return data, data_valid

for fraud_reason in labels['Fraud_reasons'].values:
    set_fraud_rate_features(data, data_valid, fraud_reason)

data, data_valid = set_fraud_rate_features(data, data_valid)


extracted_features_train = data[['is_fraud', 'site_id_fraud_rate', 'sub_site_id_fraud_rate', 'publisher_fraud_rate',
                      'operator_fraud_rate', 'city_fraud_rate', 'device_type_fraud_rate', 'os_version_fraud_rate',
                      'sdk_version_fraud_rate', 'time_difference']]
extracted_features_train.to_csv('extracted_features_train.csv', sep=';', index=False, encoding='utf-8')

extracted_features_valid = data_valid[['is_fraud', 'site_id_fraud_rate', 'sub_site_id_fraud_rate', 'publisher_fraud_rate',
                      'operator_fraud_rate', 'city_fraud_rate', 'device_type_fraud_rate', 'os_version_fraud_rate',
                      'sdk_version_fraud_rate', 'time_difference']]
extracted_features_valid.to_csv('extracted_features_valid.csv', sep=';', index=False, encoding='utf-8')
