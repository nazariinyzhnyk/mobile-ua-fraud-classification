import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('data_to_model.csv')

target = data['fraud_flag']
data = data.loc[:, data.columns != 'fraud_flag']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=7)

model = XGBClassifier()
model.fit(X_train, y_train)

print(model)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
