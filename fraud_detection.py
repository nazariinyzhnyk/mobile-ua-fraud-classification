import pandas as pd
import numpy as np

labels = pd.read_csv('labels.csv', delimiter=';')
data = pd.read_csv('data_set .csv')
labels = labels.rename(columns={'Publisher (media_source)': 'publisher_labels'})
data = data.rename(columns={'publisher': 'publisher_data'})

# Тут есть почти 8000 просмотров, я их убрал, чтобы данные стали чище (можно добавить обратно)
data = data[data['attributed_touch_type'] == 'click']

print(data.head())
print(data.info())
print(labels.info())
print(labels.head())

count_by_reasons = labels.Fraud_reasons.value_counts()
print(count_by_reasons)

count_by_publisher_data = data.publisher_data.value_counts()
count_by_publisher_labels = labels.publisher_labels.value_counts()
# В основном, бо больших паблишерах 40% фрода, по BR неизвестно, так как нету данных в data_set
# В условии сказано, что нельзя точно доверять labels
# AX и AG - в условии: считаем, что 1% фрода. По данным, в этом удостоверится нельзя
div = count_by_publisher_labels / count_by_publisher_data

# Отношения кол-ва фродов ко всем установкам по паблишерам
print(pd.concat([div, count_by_publisher_labels, count_by_publisher_data], axis=1, join='outer'))

# Присоединим флаги фродов к данным
data = pd.merge(data, labels[['Appsflyer_id', 'Fraud_reasons']], how='left', left_on='appsflyer_id', right_on='Appsflyer_id')
# 1 - есть фрод, 0 - нет
data['fraud_flag'] = data['Appsflyer_id'].notna()

data['time_difference'] = (pd.to_datetime(data['install_time'])
                           - pd.to_datetime(data['attributed_touch_time'])).dt.total_seconds()

# Отношение кол-ва фродов ко всем установкам по времени между взаимодействием и инсталом
# Если разница между временем 80 и 81 сек - ratio = 0.85 (очень высокий коэф фрода)
ratio = data[data['fraud_flag']].time_difference.value_counts() / data.time_difference.value_counts()
print(pd.concat([ratio[ratio > 0.6], data.time_difference.value_counts()], axis=1, join='outer').dropna())

"""
data_contributor_1_touch_time = data[data['contributor_1_touch_time']
                                     != '0000-00-00 00:00:00']
print(data_contributor_1_touch_time.info())

data_contributor_2_touch_time = data_contributor_1_touch_time[data_contributor_1_touch_time['contributor_2_touch_time']
                                                              != '0000-00-00 00:00:00']
print(data_contributor_2_touch_time.info())

data_contributor_3_touch_time = data_contributor_2_touch_time[data_contributor_2_touch_time['contributor_3_touch_time']
                                                              != '0000-00-00 00:00:00']
print(data_contributor_3_touch_time.info())
"""