import pandas as pd
# import xgboost as xgb
# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import operator
import matplotlib.pyplot as plt
import lightgbm as lgb

data_train = pd.read_csv('extracted_features_train.csv', sep=';')
data_valid = pd.read_csv('extracted_features_valid.csv', sep=';')

fraud_types = np.array(['click_spamming', 'tti_fraud', 'mix', 'Data center', 'bots'])
label_encoder = LabelEncoder()
data_train.loc[data_train['Fraud_reasons'].isnull(), 'Fraud_reasons'] = 'ok'
data_valid.loc[data_valid['Fraud_reasons'].isnull(), 'Fraud_reasons'] = 'ok'

le = label_encoder.fit(data_train['Fraud_reasons'].values)
y_train = le.transform(data_train['Fraud_reasons'].values)
y_valid = le.transform(data_valid['Fraud_reasons'].values)
x_train = data_train.loc[:, data_train.columns != 'Fraud_reasons']
x_valid = data_valid.loc[:, data_valid.columns != 'Fraud_reasons']
x_valid = x_valid[x_train.columns]

columns = ['install_time_weekday']
le_dict = {col: LabelEncoder() for col in columns}
for col in columns:
    x_train[col] = le_dict[col].fit_transform(x_train[col])
    x_valid[col] = le_dict[col].fit_transform(x_valid[col])



model = lgb.LGBMClassifier(objective='multiclassova', learning_rate=0.02, n_estimators=400, num_iterations=500,
                           num_leaves=100, max_depth=7)
model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=10)


# model = XGBClassifier(max_depth=7, learning_rate=0.2, n_estimators=200,
#                       objective='multi:softmax', eval_metric='mlogloss')
# # model = pickle.load(open("xgb.model", "rb"))
#
# model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=10)
# pickle.dump(model, open("xgb.model", "wb"))
# model = pickle.load(open("xgb.model", "rb"))

y_pred = model.predict(x_valid)

accuracy = accuracy_score(y_valid, y_pred)
precisions = precision_score(le.inverse_transform(y_valid), le.inverse_transform(y_pred), average=None, labels=le.classes_)
recalls = recall_score(le.inverse_transform(y_valid), le.inverse_transform(y_pred), average=None, labels=le.classes_)
print(f'Overall accuracy: {100 * accuracy:.2f}%')
for i in range(len(le.classes_)):
    print(f'Fraud reason: {le.classes_[i]}, Precision: {(100 * precisions[i]):.2f}%, Recall: {(100 * recalls[i]):.2f}%')

fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(model, max_num_features=20, height=0.5, ax=ax)
plt.savefig('feature_importance_xgb.png', bbox_inches='tight')

