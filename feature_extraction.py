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
site_ratio = data[['fraud_flag', 'site_id']].groupby(['site_id']).mean().reset_index().rename(
    columns={'fraud_flag': 'site_id_numeric'})
data = pd.merge(data, site_ratio, how='left', left_on='site_id', right_on='site_id')

site_ratio = data[['fraud_flag', 'publisher_data']].groupby(['publisher_data']).mean().reset_index().rename(
    columns={'fraud_flag': 'publisher_data_numeric'})
data = pd.merge(data, site_ratio, how='left', left_on='publisher_data', right_on='publisher_data')


"""
    Закончились числовые фичи, дальше - категориальные
"""



# для скейлинга всех числовых фич
#scaler = MinMaxScaler()
#data_numeric = data[['views', 'likes', 'dislikes', 'comment_count']]
#data_numpy = scaler.fit_transform(data_numeric)
#scaled_data = pd.DataFrame(data_numpy, index=data_numeric.index, columns=data_numeric.columns)
#data[['views', 'likes', 'dislikes', 'comment_count']] = scaled_data

#data = pd.get_dummies(data, columns=['category_id'], prefix=['category'])
