import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from models import *
np.random.seed(42)

data_train = pd.read_csv('extracted_features_train.csv', sep=';')
data_valid = pd.read_csv('extracted_features_valid.csv', sep=';')
# bool_cols = data_train.dtypes[data_train.dtypes == bool].index.values
# data_train[bool_cols] = data_train[bool_cols].astype(str)
# data_valid[bool_cols] = data_valid[bool_cols].astype(str)

data_train.drop(data_train[data_train['Fraud_reasons'] == 'mix'].index, inplace=True)
data_valid.drop(data_valid[data_valid['Fraud_reasons'] == 'mix'].index, inplace=True)

data_train.loc[data_train['Fraud_reasons'].isnull(), 'Fraud_reasons'] = 'ok'
data_valid.loc[data_valid['Fraud_reasons'].isnull(), 'Fraud_reasons'] = 'ok'

label_encoder = LabelEncoder()
le = label_encoder.fit(data_train['Fraud_reasons'].values)
y_train = le.transform(data_train['Fraud_reasons'].values)
y_valid = le.transform(data_valid['Fraud_reasons'].values)
x_train = data_train.loc[:, data_train.columns != 'Fraud_reasons']
x_valid = data_valid.loc[:, data_valid.columns != 'Fraud_reasons']
x_valid = x_valid[x_train.columns]

# train_catboost(x_train, y_train, x_valid, y_valid, le)
train_lgb(x_train, y_train, x_valid, y_valid, le)
