import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(valid_size=0):
    labels = pd.read_csv('labels.csv', delimiter=';')
    data = pd.read_csv('data_set .csv')
    data = pd.merge(data, labels[['Appsflyer_id', 'Fraud_reasons']], how='left', left_on='appsflyer_id',
                    right_on='Appsflyer_id')

    data_train, data_valid = train_test_split(data, test_size=valid_size, random_state=42)
    return data_train, data_valid, labels