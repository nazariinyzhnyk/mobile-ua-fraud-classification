import pandas as pd
import numpy as np
from lib import load_data
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

config = {
    'only_click_touch_type': False,
    'standard_scale': True
}

data, data_valid, labels = load_data(valid_size=0.2)
data['is_fraud'] = ~data['Fraud_reasons'].isnull()
data_valid['is_fraud'] = ~data_valid['Fraud_reasons'].isnull()
data['install_time'] = pd.to_datetime(data['install_time'])
data_valid['install_time'] = pd.to_datetime(data_valid['install_time'])
data['attributed_touch_time'] = pd.to_datetime(data['attributed_touch_time'])
data_valid['attributed_touch_time'] = pd.to_datetime(data_valid['attributed_touch_time'])
# data['contributor_1_touch_time'] = pd.to_datetime(data['contributor_1_touch_time'], errors='coerce')
# data['contributor_2_touch_time'] = pd.to_datetime(data['contributor_2_touch_time'], errors='coerce')
# data['contributor_3_touch_time'] = pd.to_datetime(data['contributor_3_touch_time'], errors='coerce')
# data_valid['contributor_1_touch_time'] = pd.to_datetime(data_valid['contributor_1_touch_time'], errors='coerce')
# data_valid['contributor_2_touch_time'] = pd.to_datetime(data_valid['contributor_2_touch_time'], errors='coerce')
# data_valid['contributor_3_touch_time'] = pd.to_datetime(data_valid['contributor_3_touch_time'], errors='coerce')

if config['only_click_touch_type']:
    data = data[data['attributed_touch_type'] == 'click']

extracted_features_columns = np.array([])
def set_time_features(data, data_valid):
    data['time_difference'] = (data['install_time']
                               - data['attributed_touch_time']).dt.total_seconds()
    data_valid['time_difference'] = (pd.to_datetime(data_valid['install_time'])
                               - data_valid['attributed_touch_time']).dt.total_seconds()
    data['install_time_since_midnight_sec'] = (data['install_time'] -
                                               pd.to_datetime(data['install_time'].dt.date)) / np.timedelta64(1, 's')
    data_valid['install_time_since_midnight_sec'] = (data_valid['install_time'] -
                                               pd.to_datetime(data_valid['install_time'].dt.date)) / np.timedelta64(1, 's')

    data['attributed_touch_time_since_midnight_sec'] = (data['attributed_touch_time'] -
                                               pd.to_datetime(data['attributed_touch_time'].dt.date)) / np.timedelta64(1, 's')
    data_valid['attributed_touch_time_since_midnight_sec'] = (data_valid['attributed_touch_time'] -
                                                        pd.to_datetime(
                                                            data_valid['attributed_touch_time'].dt.date)) / np.timedelta64(1,
                                                                                                                     's')
    # data['tti_contributor1'] = (data['install_time']
    #                             - data_valid['contributor_1_touch_time']).dt.total_seconds()
    # data['tti_contributor2'] = (data['install_time']
    #                             - data_valid['contributor_2_touch_time']).dt.total_seconds()
    # data['tti_contributor3'] = (data['install_time']
    #                             - data_valid['contributor_3_touch_time']).dt.total_seconds()
    #
    # data_valid['tti_contributor1'] = (data_valid['install_time']
    #                             - data_valid['contributor_1_touch_time']).dt.total_seconds()
    # data_valid['tti_contributor2'] = (data_valid['install_time']
    #                             - data_valid['contributor_2_touch_time']).dt.total_seconds()
    # data_valid['tti_contributor3'] = (data_valid['install_time']
    #                             - data_valid['contributor_3_touch_time']).dt.total_seconds()
    extracted_features_columns = np.array(['time_difference', 'install_time_since_midnight_sec', 'attributed_touch_time_since_midnight_sec'])
                                        # 'tti_contributor1', 'tti_contributor2', 'tti_contributor3'])
    return data, data_valid, extracted_features_columns

def set_fraud_rate_feature(train_df, valid_df, column, mean_fraud, fraud_reason):
    rate_feature_column = column + '_' + fraud_reason + '_fraud_rate'
    print(column)
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

fraud_rates_columns = ['site_id', 'sub_site_id', 'language', 'publisher', 'operator', 'city', 'device_type',
                       'os_version', 'sdk_version', 'app_id', 'app_version', 'country_code', 'wifi']
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

for fraud_reason in labels['Fraud_reasons'].unique():
    print(f'Extracting features for fraud reason: {fraud_reason}')
    data, data_valid, fraud_reason_features_columns = set_fraud_rate_features(data, data_valid, fraud_reason)
    data[fraud_reason + '_mean_rate'] = data[fraud_reason_features_columns].mean(axis=1)
    data_valid[fraud_reason + '_mean_rate'] = data_valid[fraud_reason_features_columns].mean(axis=1)
    fraud_reason_features_columns = np.concatenate((fraud_reason_features_columns, np.array([fraud_reason + '_mean_rate'])))
    extracted_features_columns = np.concatenate((extracted_features_columns, fraud_reason_features_columns))

if config['standard_scale']:
    scaler = StandardScaler()
    scaler.fit(data[extracted_features_columns])
    data[extracted_features_columns] = scaler.transform(data[extracted_features_columns])
    data_valid[extracted_features_columns] = scaler.transform(data_valid[extracted_features_columns])

extracted_features_columns = np.concatenate((extracted_features_columns, np.array(['Fraud_reasons'])))
extracted_features_train = data[extracted_features_columns]
extracted_features_train.to_csv('extracted_features_train.csv', sep=';', index=False, encoding='utf-8')

extracted_features_valid = data_valid[extracted_features_columns]
extracted_features_valid.to_csv('extracted_features_valid.csv', sep=';', index=False, encoding='utf-8')

time_features = np.concatenate((time_features, np.array(['Fraud_reasons'])))
time_features_train = data[time_features]
time_features_train.to_csv('time_features_train.csv', sep=';', index=False, encoding='utf-8')
time_features_valid = data_valid[time_features]
time_features_valid.to_csv('time_features_valid.csv', sep=';', index=False, encoding='utf-8')
