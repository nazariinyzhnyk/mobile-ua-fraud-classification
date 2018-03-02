import pandas as pd
import numpy as np

labels = pd.read_csv('labels.csv', delimiter=';')
data = pd.read_csv('data_set .csv')
labels = labels.rename(columns={'Publisher (media_source)': 'publisher_labels'})
data = data.rename(columns={'publisher': 'publisher_data'})

# Тут есть почти 8000 просмотров, я их убрал, чтобы данные стали чище (можно добавить обратно)
data = data[data['attributed_touch_type'] == 'click']

# Присоединим флаги фродов к данным
data = pd.merge(data, labels[['Appsflyer_id', 'Fraud_reasons']], how='left', left_on='appsflyer_id', right_on='Appsflyer_id')
# 1 - есть фрод, 0 - нет
data['fraud_flag'] = data['Appsflyer_id'].notna()

data['time_difference'] = (pd.to_datetime(data['install_time'])
                           - pd.to_datetime(data['attributed_touch_time'])).dt.total_seconds()

