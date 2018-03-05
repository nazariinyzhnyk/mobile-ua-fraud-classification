import keras
from lib import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from keras.constraints import non_neg
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.utils import np_utils

def get_callbacks(model_name='model'):
    model_checkpoint = ModelCheckpoint(model_name + '.model', monitor='val_categorical_accuracy', save_best_only=True, save_weights_only=False,
                                       verbose=1)
    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=5, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=1, verbose=1)

    return [model_checkpoint, early_stopping, reduce_lr]

data_train = pd.read_csv('extracted_features_train.csv', sep=';')
data_valid = pd.read_csv('extracted_features_valid.csv', sep=';')

fraud_types = np.array(['click_spamming', 'tti_fraud', 'mix', 'Data center', 'bots'])
classes = np.concatenate((fraud_types, ['ok']))
print(data_train.dtypes)
label_encoder = LabelEncoder()
data_train.loc[data_train['Fraud_reasons'].isnull(), 'Fraud_reasons'] = 'ok'
data_valid.loc[data_valid['Fraud_reasons'].isnull(), 'Fraud_reasons'] = 'ok'

le = label_encoder.fit(data_train['Fraud_reasons'].values)
y_train = le.transform(data_train['Fraud_reasons'].values)
y_train = np_utils.to_categorical(y_train)
y_valid = le.transform(data_valid['Fraud_reasons'].values)
y_valid = np_utils.to_categorical(y_valid)
x_train = data_train.loc[:, data_train.columns != 'Fraud_reasons']
x_valid = data_valid.loc[:, data_valid.columns != 'Fraud_reasons']
x_valid = x_valid[x_train.columns]

input = keras.layers.Input(shape=(len(x_valid.columns),))
x = keras.layers.Dense(128, activation='relu')(input)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(len(classes), activation='softmax')(x)
#
model = keras.Model([
    input,
], x)
opt = keras.optimizers.Adam(lr=0.002)
model.compile(opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()
history = model.fit([x_train.values], y_train, validation_data=([x_valid.values],y_valid),
                    batch_size=256, callbacks=get_callbacks('nn'), epochs=50, verbose=2)
model = load_model('nn.model')
predictions = model.predict(x_valid)