import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score

def encode_cat_features(x_train, x_valid):
    cat_columns_index = np.where(x_train.dtypes != float)[0]
    cat_columns = list(x_train.columns[cat_columns_index])
    for cat_column in cat_columns:
        x_train[cat_column] = x_train[cat_column].astype('category')
        x_valid[cat_column] = x_valid[cat_column].astype('category')
    return x_train, x_valid, cat_columns

def train_lgb(x_train, y_train, x_valid, y_valid, le):
    x_train, x_valid, categorical_features = encode_cat_features(x_train, x_valid)
    lgb_model = lgb.LGBMClassifier(objective='multiclass', learning_rate=0.02, n_estimators=400, num_iterations=1000,
                                   num_leaves=100, max_depth=7, categorical_feature=categorical_features)
    lgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=10,  categorical_feature=categorical_features)
    pickle.dump(lgb_model, open('lgb.model', 'wb'))
    lgb_model = pickle.load(open('lgb.model', 'rb'))
    print_metrics(lgb_model, x_valid, y_valid, le)

    fig, ax = plt.subplots(figsize=(12, 18))
    lgb.plot_importance(lgb_model, max_num_features=30, height=0.5, ax=ax)
    plt.savefig('feature_importance_lgb.png', bbox_inches='tight')
    plt.show()

def print_metrics(model, x_valid, y_valid, le):
    y_pred = model.predict(x_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    precisions = precision_score(le.inverse_transform(y_valid), le.inverse_transform(y_pred), average=None,
                                 labels=le.classes_)
    recalls = recall_score(le.inverse_transform(y_valid), le.inverse_transform(y_pred), average=None,
                           labels=le.classes_)
    print(f'Overall accuracy: {100 * accuracy:.2f}%')
    for i in range(len(le.classes_)):
        print(f'Fraud reason: {le.classes_[i]}, Precision: {(100 * precisions[i]):.2f}%, Recall: {(100 * recalls[i]):.2f}%')
