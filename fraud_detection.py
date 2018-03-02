import pandas as pd
# import xgboost as xgb
# from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import Pool, CatBoostClassifier, cv, CatboostIpythonWidget

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data = pd.read_csv('data_to_model.csv', sep=';', dtype={'publisher_data': str, 'site_id': str})

#, dtype={'os_version': str, 'device_type': str, 'fraud_flag': int, 'wifi': np.float,
                                                  #       'time_difference': np.float}, encoding='utf-8')
print(data.dtypes)
target = data['fraud_flag']
data = data.loc[:, data.columns != 'fraud_flag']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
print(data.columns[categorical_features_indices])
X_train.fillna(-999, inplace=True)
X_test.fillna(-999, inplace=True)

model = CatBoostClassifier(
    iterations=100,
    loss_function='Logloss',
    random_seed=42,
    eval_metric='Accuracy'
)
# model.fit(
#     X_train, y_train,
#     cat_features=categorical_features_indices,
#     eval_set=(X_test, y_test),
#     logging_level='Verbose',  # you can uncomment this for text output
#     plot=False,
#     use_best_model=True
# )
# model.save_model('model.model')
model.load_model('model.model')
print((model.predict(X_test) == y_test).sum()/len(y_test))

# cv_data = cv(
#     model.get_params(),
#     Pool(data.values, label=target.values, cat_features=categorical_features_indices),
# )
#
# print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
#     np.max(cv_data['Accuracy_test_avg']),
#     cv_data['Accuracy_test_stddev'][np.argmax(cv_data['Accuracy_test_avg'])],
#     np.argmax(cv_data['Accuracy_test_avg'])
# ))
# print('Precise validation accuracy score: {}'.format(np.max(cv_data['Accuracy_test_avg'])))

#
# # pd.get_dummies(X_train['device_type'], prefix='vi', prefix_sep='_')
#
# # model = XGBClassifier()
# model = LGBMClassifier()
# model.fit(X_train.values, y_train.values)
#
# print(model)
#
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
#
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
