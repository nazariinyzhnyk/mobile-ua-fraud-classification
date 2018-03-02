import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

labels = pd.read_csv('labels.csv', delimiter=';')
data = pd.read_csv('data_set .csv')
labels = labels.rename(columns={'Publisher (media_source)': 'publisher_labels'})
data = data.rename(columns={'publisher': 'publisher_data'})

# Тут есть почти 8000 просмотров, я их убрал, чтобы данные стали чище (можно добавить обратно)
data = data[data['attributed_touch_type'] == 'click']

# Присоединим флаги фродов к данным
data = pd.merge(data, labels[['Appsflyer_id', 'Fraud_reasons']], how='left', left_on='appsflyer_id', right_on='Appsflyer_id')
# 1 - есть фрод, 0 - нет
data['fraud_flag'] = data['Appsflyer_id'].notna().astype(int)
mean_fraud = data.fraud_flag.mean()

"""
    Сначала нужно разобратся с временем
"""
data['time_difference'] = (pd.to_datetime(data['install_time'])
                           - pd.to_datetime(data['attributed_touch_time'])).dt.total_seconds()

s = pd.datetime.now() - pd.to_datetime(data['install_time'])
data['install_time'] = s.dt.total_seconds()

scaler = MinMaxScaler()
data_numeric = data[['time_difference', 'install_time']]
data_numpy = scaler.fit_transform(data_numeric)
scaled_data = pd.DataFrame(data_numpy, index=data_numeric.index, columns=data_numeric.columns)
data[['time_difference', 'install_time']] = scaled_data
"""
"""

# уникальных значений site_id - 6000, поэтому тут приписывается каждому site_id рейт фрода и эта фича будет как числовая
# Так делаеться для многих категориальных фич, наверное этот код можно упростить
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

data['app_id_numeric'] = np.where(data['app_id'] == 'ng.jiji.app', 1, 0)


#print(data[['sdk_version', 'fraud_flag']].groupby('sdk_version').mean())
# фичи, используемые в модели
data_to_model = data[['site_id_numeric', 'publisher_data_numeric', 'app_id_numeric', 'operator_numeric',
                      'city_numeric', 'device_type_numeric', 'os_version_numeric', 'sdk_version_numeric',
                      'install_time', 'time_difference', 'wifi', 'fraud_flag']]

data_to_model.to_csv('data_to_model.csv', sep=',')

print(data_to_model.info())

#data = pd.get_dummies(data, columns=['category_id'], prefix=['category'])
