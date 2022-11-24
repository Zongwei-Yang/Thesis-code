# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:16:17 2022

@author: Administrator
"""

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, Bidirectional, Multiply, LSTM
from keras.layers.core import *
from keras.models import *
from sklearn.metrics import mean_absolute_error
from keras import backend as K
from tensorflow.python.keras.layers import CuDNNLSTM

from my_utils.read_write import pdReadCsv
import numpy as np

SINGLE_ATTENTION_VECTOR = False
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_KERAS"] = '1'

# Attention mechanism
def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = inputs
    a = Dense(input_dim, activation='softmax')(a)
    # Replace the dimension input according to the given schema (dim)  For example, (2,1) is the first and second dimension of displacement input
    a_probs = Permute((1, 2), name='attention_vec')(a)
    # Layer that multiplies (element-wise) a list of inputs.
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

# Create timing data block
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y

# Establish CNN-BiLSTM and add attention mechanism
def attention_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
    # 卷积层和dropout层
    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # , padding = 'same'
    x = Dropout(0.3)(x)
    # For GPU you can use CuDNNLSTM cpu LSTM
    lstm_out = Bidirectional(CuDNNLSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    # It is used to compress the data of the input layer into one-dimensional data, generally between the convolution layer and the full connection layer
    attention_mul = Flatten()(attention_mul)
	
    # output = Dense(1, activation='sigmoid')(attention_mul)  分类
    output = Dense(1, activation='linear')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

# Normalization
def fit_size(x, y):
    from sklearn import preprocessing
    x_MinMax = preprocessing.MinMaxScaler()
    y_MinMax = preprocessing.MinMaxScaler()
    x = x_MinMax.fit_transform(x)
    y = y_MinMax.fit_transform(y)
    return x, y, y_MinMax


def flatten(X):
    flattened_X = np.empty((X.shape[0], X.shape[2]))
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return (flattened_X)


src = r'E:\dat'
path = r'E:\dat'
trials_path = r'E:\dat'
train_path = src + r'merpre.csv'
df = pdReadCsv(train_path, ',')
df = df.replace("--", '0')
df.fillna(0, inplace=True)
INPUT_DIMS = 43
TIME_STEPS = 12
lstm_units = 64


def load_data(df_train):
    X_train = df_train.drop(['Per'], axis=1)
    y_train = df_train['wap'].values.reshape(-1, 1)
    return X_train, y_train, X_train, y_train


groups = df.groupby(['Per'])
for name, group in groups:
    X_train, y_train, X_test, y_test = load_data(group)
    # Normalization
    train_x, train_y, train_y_MinMax = fit_size(X_train, y_train)
    test_x, test_y, test_y_MinMax = fit_size(X_test, y_test)

    train_X, _ = create_dataset(train_x, TIME_STEPS)
    _, train_Y = create_dataset(train_y, TIME_STEPS)
    print(train_X.shape, train_Y.shape)

    m = attention_model()
    m.summary()
    m.compile(loss='mae', optimizer='Adam', metrics=['mae'])
    model_path = r'me_pre\\'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),  # Combine EarlyStopping mechanism to improve model training efficiency and prevent over fitting. When the loss of two iterations is not improved, Keras stops training
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]
    m.fit(train_X, train_Y, batch_size=32, epochs=111, shuffle=True, verbose=1,
          validation_split=0.1, callbacks=callbacks)
    # m.fit(train_X, train_Y, epochs=111, batch_size=32)
    test_X, _ = create_dataset(test_x, TIME_STEPS)
    _, test_Y = create_dataset(test_y, TIME_STEPS)

    pred_y = m.predict(test_X)
    inv_pred_y = test_y_MinMax.inverse_transform(pred_y)
    inv_test_Y = test_y_MinMax.inverse_transform(test_Y)
    mae = int(mean_absolute_error(inv_test_Y, inv_pred_y))
    print('test_mae : ', mae)
    mae = str(mae)
    print(name)
    m.save(model_path + name[0] + '_' + name[1] + '_' + name[2] + '_' + mae + '.h5')