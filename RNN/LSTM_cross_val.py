from pathlib import Path
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
import datetime
from tensorflow.keras.callbacks import EarlyStopping

""" LSTM_cross_val.py for cross validation of RNN model. """
__author__ = "Ludvig Eriksson, Johan Lamm, Ellinor Lundblad"
__maintainer__ = "Johan Lamm"
__email__ = "lammj@student.chalmers.se"

# Path to data set for training, testing and validation.
data_path = "../Data/Full_data.csv"

# ========================================
# Parameters to run cross validation on (change these for selection of different parameters for cross validation)
# ========================================
lay = 1
n = 128
a = 'sigmoid'
metrics = ['mape', "mse"]
epochs = 500
batch_size = 32
learning_rate = 0.0001

# ========================================
# Some data preparation
# ========================================
df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
df = df.dropna()
df.columns = df.columns.map(str)

df = df.astype('float32')


outdata = pd.DataFrame()
# Cross validation loop with different split for different selections of i
for i in range(10):
    valstart = int(len(df)*i/10)
    valend = valstart+96*7
    start = int(len(df)*((i + 1) % 10)/10)
    end = int(len(df)*((i + 1) % 10 +1)/10)-1
    temp = df.drop(df.index[start:end])
    train = temp.drop(df.index[valstart:valend])

    validation = df[valstart:valend]
    test = df[start:end]

    # Remove future cols from input set and add to output/target sets instead
    futurecols = []
    for i in range(0, 96):
        futurecols.append('f'+str(15*(i+1))+'m')

    target = pd.concat([train.pop(x) for x in futurecols],
                       axis=1)  # Pop future cols from train set and add to target set
    real = pd.concat([test.pop(x) for x in futurecols], axis=1)  # Pop future cols from test set and add to real set
    val_target = pd.concat([validation.pop(x) for x in futurecols],
                           axis=1)  # Pop future cols from validation set and add
    # to val_target set

    # Save the model index for later usage
    index = test.index

    # Reshape the pandas dataframes to numpy instead to enable rescaling with MinMaxScaler()
    train = train.to_numpy()
    train = train.reshape(train.shape[0], train.shape[1])
    test = test.to_numpy()
    test = test.reshape(test.shape[0], test.shape[1])
    validation = validation.to_numpy()
    validation = validation.reshape(validation.shape[0], validation.shape[1])
    target = target.to_numpy()
    target = target.reshape(target.shape[0], target.shape[1])
    real = real.to_numpy()
    real = real.reshape(real.shape[0], real.shape[1])
    val_target = val_target.to_numpy()
    val_target = val_target.reshape(val_target.shape[0], val_target.shape[1])

    # Rescale data with MinMax scaler.
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train)  # Fit the scaler to training data and transform.
    test = scaler.transform(test)  # Transform the test data
    validation = scaler.transform(validation)  # Transform the validation data
    target = scaler.fit_transform(target)  # Fit the scaler to the target data and transform
    val_target = scaler.transform(val_target)  # Transform validation target data
    real = scaler.transform(real)  # Transform real data

    no_steps = 96  # Set number of steps to look back in LSTM to 96 i.e number of quarts to look back.

    # Reshape the data to enable usage in LSTM modell. (3D shape needed)
    train = train.reshape(train.shape[0], 1, train.shape[1])
    test = test.reshape(test.shape[0], 1, test.shape[1])
    validation = validation.reshape(validation.shape[0], 1, validation.shape[1])
    # Set stop condition for training
    cb_es = EarlyStopping(monitor='val_mse', patience=50)

    model = tf.keras.models.Sequential()

    for L in range(lay):
        if L == range(lay)[-1]:# Last LSTM layer should not return sequences.
            model.add(LSTM(units=n, input_shape=(train.shape[1], train.shape[2]), return_sequences=False,
                           activation=a))
        else:
            model.add(LSTM(n, input_shape=(train.shape[1], train.shape[2]), return_sequences=True,
                           activation=a))
        model.add(Dropout(0.2)) # Adds a dropout layer of 20% after each LSTM layer
    model.add(Dense(n, activation="relu"))
    model.add(Dense(units=len(target[0])))
    opti = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opti,
                  loss='mse',
                  metrics=metrics)
    history = model.fit(train, target,
                        epochs=epochs,
                        validation_data=(validation, val_target),
                        batch_size=batch_size, callbacks=cb_es)


    predicted = model.predict(test)
    predicted = scaler.inverse_transform(predicted)
    predicted = pd.DataFrame(predicted, index=index)

    real = scaler.inverse_transform(real)
    real = pd.DataFrame(real, index=index)

    data = pd.concat([real, predicted], axis=1)
    print("Predicted")
    print(predicted)
    print("Real")
    print(real)
    outdata = pd.concat([data, outdata], axis=0) # Merge old outdata with recent data from latest prediction.


# Save the outdata to a csv file
outdata.to_csv('../Data/diff_LSTM_long_1.csv')


