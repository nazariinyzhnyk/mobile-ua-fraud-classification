import pandas as pd
import numpy as np
from lib import load_data
from tqdm import tqdm

config = {
    'only_click_touch_type': False
}

data, data_valid, labels = load_data(valid_size=0.2)
data['is_fraud'] = ~data['Fraud_reasons'].isnull()
data_valid['is_fraud'] = ~data_valid['Fraud_reasons'].isnull()
extracted_features_columnns = ['Fraud_reasons']

if config['only_click_touch_type']:
    data = data[data['attributed_touch_type'] == 'click']

def set_time_features(data, data_valid):
    data['time_difference'] = (pd.to_datetime(data['install_time'])
                               - pd.to_datetime(data['attributed_touch_time'])).dt.total_seconds()
    data_valid['time_difference'] = (pd.to_datetime(data_valid['install_time'])
                               - pd.to_datetime(data_valid['attributed_touch_time'])).dt.total_seconds()
    extracted_features_columnns.extend(['time_difference'])
    return data, data_valid

def set_fraud_rate_feature(train_df, valid_df, column, mean_fraud, fraud_reason):
    rate_feature_column = column + '_' + fraud_reason + '_fraud_rate'
    ratio = train_df[['Fraud_reasons', column]].groupby([column]).agg(
        lambda x: (x['Fraud_reasons'] == fraud_reason).sum()/len(x))
    ratio.rename(columns={'Fraud_reasons': rate_feature_column}, inplace=True)

    train_df = pd.merge(train_df, ratio, how='left', left_on=column, right_index=True)
    train_df[rate_feature_column] = train_df[rate_feature_column].fillna(mean_fraud)

    valid_df = pd.merge(valid_df, ratio, how='left', left_on=column, right_index=True)
    valid_df[rate_feature_column] = valid_df[rate_feature_column].fillna(mean_fraud)

    extracted_features_columnns.extend([rate_feature_column])

    return train_df, valid_df

fraud_rates_columns = ['site_id', 'sub_site_id', 'publisher', 'operator', 'city', 'device_type', 'os_version', 'sdk_version']
def set_fraud_rate_features(data, data_valid, fraud_reason):
    mean_fraud = (data['Fraud_reasons'] == fraud_reason).mean()
    for col in tqdm(fraud_rates_columns):
        data, data_valid = set_fraud_rate_feature(data, data_valid, col, mean_fraud, fraud_reason)

    return data, data_valid

data, data_valid = set_time_features(data, data_valid)

for fraud_reason in labels['Fraud_reasons'].unique():
    print(f'Extracting features for fraud reason: {fraud_reason}')
    data, data_valid = set_fraud_rate_features(data, data_valid, fraud_reason)

extracted_features_train = data[extracted_features_columnns]
extracted_features_train.to_csv('extracted_features_train.csv', sep=';', index=False, encoding='utf-8')

extracted_features_valid = data_valid[extracted_features_columnns]
extracted_features_valid.to_csv('extracted_features_valid.csv', sep=';', index=False, encoding='utf-8')
