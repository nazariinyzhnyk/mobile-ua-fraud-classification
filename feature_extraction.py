import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lib import load_data

config = {
    'only_click_touch_type': False
}

data, data_valid, labels = load_data(valid_size=0.2)
labels = labels.rename(columns={'Publisher (media_source)': 'publisher_labels'})
data = data.rename(columns={'publisher': 'publisher_data'})

if config['only_click_touch_type']:
    data = data[data['attributed_touch_type'] == 'click']

data['fraud_flag'] = data['Appsflyer_id'].notnull().astype(int)

def set_time_features(data):
    data['time_difference'] = (pd.to_datetime(data['install_time'])
                               - pd.to_datetime(data['attributed_touch_time'])).dt.total_seconds()
    s = pd.datetime.now() - pd.to_datetime(data['install_time'])
    data['install_time'] = s.dt.total_seconds()

    scaler = MinMaxScaler()
    data_numeric = data[['time_difference', 'install_time']]
    data_numpy = scaler.fit_transform(data_numeric)
    scaled_data = pd.DataFrame(data_numpy, index=data_numeric.index, columns=data_numeric.columns)
    data[['time_difference', 'install_time']] = scaled_data

set_time_features(data)

def set_fraud_rate_features(data):
    mean_fraud = data['fraud_flag'].mean()
    site_ratio = data[['fraud_flag', 'site_id']].groupby(['site_id']).mean().reset_index().rename(
        columns={'fraud_flag': 'site_id_numeric'})
    data = pd.merge(data, site_ratio, how='left', left_on='site_id', right_on='site_id')
    data['site_id_numeric'] = data['site_id_numeric'].fillna(mean_fraud)

    site_ratio = data[['fraud_flag', 'publisher_data']].groupby(['publisher_data']).mean().reset_index().rename(
        columns={'fraud_flag': 'publisher_data_numeric'})
    data = pd.merge(data, site_ratio, how='left', left_on='publisher_data', right_on='publisher_data')
    data['publisher_data_numeric'] = data['publisher_data_numeric'].fillna(mean_fraud)

    site_ratio = data[['operator', 'fraud_flag']].groupby('operator').mean().reset_index().rename(
        columns={'fraud_flag': 'operator_numeric'})
    data = pd.merge(data, site_ratio, how='left', left_on='operator', right_on='operator')
    data['operator_numeric'] = data['operator_numeric'].fillna(mean_fraud)

    site_ratio = data[['city', 'fraud_flag']].groupby('city').mean().reset_index().rename(
        columns={'fraud_flag': 'city_numeric'})
    data = pd.merge(data, site_ratio, how='left', left_on='city', right_on='city')
    data['city_numeric'] = data['city_numeric'].fillna(mean_fraud)

    site_ratio = data[['device_type', 'fraud_flag']].groupby('device_type').mean().reset_index().rename(
        columns={'fraud_flag': 'device_type_numeric'})
    data = pd.merge(data, site_ratio, how='left', left_on='device_type', right_on='device_type')
    data['device_type_numeric'] = data['device_type_numeric'].fillna(mean_fraud)

    site_ratio = data[['os_version', 'fraud_flag']].groupby('os_version').mean().reset_index().rename(
        columns={'fraud_flag': 'os_version_numeric'})
    data = pd.merge(data, site_ratio, how='left', left_on='os_version', right_on='os_version')
    data['os_version_numeric'] = data['os_version_numeric'].fillna(mean_fraud)

    site_ratio = data[['sdk_version', 'fraud_flag']].groupby('sdk_version').mean().reset_index().rename(
        columns={'fraud_flag': 'sdk_version_numeric'})
    data = pd.merge(data, site_ratio, how='left', left_on='sdk_version', right_on='sdk_version')
    data['sdk_version_numeric'] = data['sdk_version_numeric'].fillna(mean_fraud)

set_fraud_rate_features(data)

data['app_id_numeric'] = np.where(data['app_id'] == 'ng.jiji.app', 1, 0)

data_to_model = data[['site_id_numeric', 'publisher_data_numeric', 'app_id_numeric', 'operator_numeric',
                      'city_numeric', 'device_type_numeric', 'os_version_numeric', 'sdk_version_numeric',
                      'install_time', 'time_difference', 'wifi', 'fraud_flag']]
data_to_model.to_csv('data_to_model.csv', sep=',')