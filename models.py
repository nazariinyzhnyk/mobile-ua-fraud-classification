import numpy as np
import lightgbm as lgb
import catboost as ctb
# from xgboost import XGBClassifier
import matplotlib.pyplot as plt
# import xgboost as xgb
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

def get_cat_columns(df):
    cat_columns_index = np.where(df.dtypes != float)[0]
    cat_columns = df.columns[cat_columns_index].values

    return cat_columns_index, cat_columns

def encode_cat_features(x_train, x_valid):
    _, cat_columns = get_cat_columns(x_train)
    le_dict = {col: LabelEncoder() for col in cat_columns}
    for col in cat_columns:
        x_train[col] = le_dict[col].fit_transform(x_train[col])
        x_valid[col] = le_dict[col].fit_transform(x_valid[col])

    return x_train, x_valid

def train_catboost(x_train, y_train, x_valid, y_valid, le):
    cat_columns_index, _ = get_cat_columns(x_train)
    catboost_model = ctb.CatBoostClassifier(loss_function='MultiClass', eval_metric='Accuracy', use_best_model=True, random_seed=42)
    catboost_model.fit(x_train, y_train, cat_features=cat_columns_index, eval_set=(x_valid, y_valid))
    pickle.dump(catboost_model, open("catboost.model", "wb"))
    print_metrics(catboost_model, x_valid, y_valid, le)

def train_lgb(x_train, y_train, x_valid, y_valid, le):
    x_train, x_valid = encode_cat_features(x_train, x_valid)
    lgb_model = lgb.LGBMClassifier(objective='multiclassova', learning_rate=0.02, n_estimators=400, num_iterations=500,
                               num_leaves=100, max_depth=7)
    lgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=10)
    pickle.dump(lgb_model, open("lgb.model", "wb"))
    print_metrics(lgb_model, x_valid, y_valid, le)

    fig, ax = plt.subplots(figsize=(12, 18))
    lgb.plot_importance(lgb_model, max_num_features=20, height=0.5, ax=ax)
    plt.savefig('feature_importance_lgb.png', bbox_inches='tight')

# def train_xgb(x_train, y_train, x_valid, y_valid):
#     x_train, x_valid = encode_cat_features(x_train, x_valid)
#     xgb_model = xgb.XGBClassifier(max_depth=7, learning_rate=0.2, n_estimators=200,
#                           objective='multi:softmax', eval_metric='mlogloss')
#     model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=10)
#     pickle.dump(xgb_model, open("catboost.model", "wb"))
#     print_metrics(xgb_model, x_valid, y_valid, le)
#     fig, ax = plt.subplots(figsize=(12, 18))
#     xgb.plot_importance(xgb_model, max_num_features=20, height=0.5, ax=ax)
#     plt.savefig('feature_importance_xgb.png', bbox_inches='tight')

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


