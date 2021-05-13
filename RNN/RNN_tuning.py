from pathlib import Path
import tensorflow as tf
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Concatenate, LeakyReLU
import datetime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras

""" RNN_tuning.py used for optimization of parameters for RNN model. """
__author__ = "Ludvig Eriksson, Johan Lamm, Ellinor Lundblad"
__maintainer__ = "Ellinor Lundblad"
__email__ = "lundblad@student.chalmers.se"

# Disables GPU usage for computation, comment out i GPU is desired.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ========================================
# Log and data settings.
# ========================================
# Gets time for creation/writing of data file.
logtime = str(datetime.datetime.now().year) + str(datetime.datetime.now().month) + str(datetime.datetime.now().day) \
          + "_" + str(datetime.datetime.now().hour) + "_" + str(datetime.datetime.now().minute) + \
          "_" + str(datetime.datetime.now().second)

# Path to data set for training, testing and validation.
data_path = "../Data/Full_data.csv"
# Creation of directory (if it does not exists) and log-files used for analysis.
Path("logs").mkdir(parents=True, exist_ok=True)
log_txt = 'logs/RNN_log_'+str(logtime)+'.txt'
log_csv = 'logs/RNN_log_'+str(logtime)+'.csv'
log = open(log_txt, 'w+')


# ========================================
# Parameters to optimize over (change these for selection of different parameters for optimization)
# ========================================
hp_layers = [1]
hp_nodes = [128]
hp_activation = ['sigmoid']
metrics = ['mape', "mse"]
epochs = 1
batch_size = [32]
learning_rate = [0.0001]


# Create columns for log.csv-fil
log_cols = ['layers', 'nodes', 'activation', 'batch_size', 'learning rate', 'runtime', 'mape', 'mse']  # Add values to log last
df_log = pd.DataFrame(columns=log_cols)  # Creating dataframe for csv conversion

df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
df = df.dropna()
df.columns = df.columns.map(str)

df = df.astype('float32')

outdata = pd.DataFrame()


# Splitting data into train, test and validation below. Change i for different data split.
i = 4
valstart = int(len(df)*i/10)
valend = int(len(df)*(i + 1)/10)-1
start = int(len(df)*((i + 5) % 10)/10)
end = int(len(df)*((i + 5) % 10 +1)/10)-1
temp = df.drop(df.index[start:end])
train = temp.drop(df.index[valstart:valend])
validation = df[valstart:valend]
test = df[start:end]
print(train)
print(validation)
print(test)

# Remove future cols from input set and add to output/target sets instead
futurecols = []
for i in range(0, 96):
    futurecols.append('f'+str(15*(i+1))+'m')
print(futurecols)

target = pd.concat([train.pop(x) for x in futurecols], axis=1) # Pop future cols from train set and add to target set
real = pd.concat([test.pop(x) for x in futurecols], axis=1) # Pop future cols from test set and add to real set
val_target = pd.concat([validation.pop(x) for x in futurecols], axis=1) # Pop future cols from validation set and add
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
train = scaler.fit_transform(train) # Fit the scaler to training data and transform.
test = scaler.transform(test) # Transform the test data
validation = scaler.transform(validation) # Transform the validation data
target = scaler.fit_transform(target) # Fit the scaler to the target data and transform
val_target = scaler.transform(val_target) # Transform validation target data
real = scaler.transform(real) # Transform real data


no_steps = validation.shape[1] # Set number of steps to look back in LSTM to validation data shape.

# Reshape the data to enable usage in LSTM modell. (3D shape needed)
train = np.reshape(train, newshape=(train.shape[0], 1, train.shape[1]))
test = np.reshape(test, newshape=(test.shape[0], 1, test.shape[1]))
validation = np.reshape(validation, newshape=(validation.shape[0], 1, validation.shape[1]))

# Set stop condition for training
cb_es = EarlyStopping(monitor='val_mse', patience=10)

results = []
run_number = 1
hparams = []

# Loop over all batch sizes, learning rates, layers, nodes and activation functions provided in the setup earlier.
for size in batch_size:
    for lr in learning_rate:
        for lay in hp_layers:
            for n in hp_nodes:
                for a in hp_activation:
                    print('\nLayers:  %i' % lay)
                    print('Nodes: %i' % n)
                    print('Activation function: %s\n' % a)
                    print('Batch size: %i\n' % size)
                    print('Learning rate: %f\n' % lr)
                    start_time = datetime.datetime.now()
                    model = tf.keras.models.Sequential()

                    for L in range(lay):
                        if L == range(lay)[-1]: # Last LSTM layer should not return sequences.
                            model.add(LSTM(units=n, input_shape=(1, no_steps), return_sequences=False,
                                           activation=a, name='LSTM_lay_%i' %L))
                        else:
                            model.add(LSTM(n, input_shape=(1, no_steps), return_sequences=True,
                                           activation=a))
                        model.add(Dropout(0.2, name="Dropout_lay_%i" %L)) # Adds a dropout layer of 20% after each LSTM layer
                    model.add(Dense(n, activation="relu", name="Dense_lay_1"))
                    model.add(Dense(units=len(target[0]), name="Output_lay"))
                    opti = keras.optimizers.Adam(learning_rate=lr)
                    model.compile(optimizer=opti,
                                  loss='mse',
                                  metrics=metrics)
                    history = model.fit(train, target,
                                        epochs=epochs,
                                        validation_data=(validation, val_target),
                                        batch_size=size, callbacks=cb_es)
                    keras.utils.plot_model(model, "RNN_finished_model.png")
                    end_time = datetime.datetime.now()
                    run_time = str(end_time - start_time)
                    print("run time: " + str(run_time))
                    # Logging of values of the different parameters for both .txt and .csv files
                    csv_input = [lay, n, a, size, lr, run_time]
                    log.write('Testrun number %i:\n' % run_number)
                    log.write('    Number of layers: %s\n' % lay)
                    log.write('    Number of nodes: %i\n' % n)
                    log.write('    Activation function: %s\n' % a)
                    log.write('    Batch size: %i\n' % size)
                    log.write('    Learning rate: %f\n' % lr)
                    log.write('    Runtime: %s\n' % run_time)

                    # Add the logging values in files for each metric in the metrics-list
                    for m in metrics:
                        metric_value = history.history['val_' + m].pop()
                        log.write('    %s: %f\n' % (m, metric_value))
                        csv_input.append(metric_value)
                    log.write('\n\n')
                    run_number += 1

                    # Put the values in a list with later is converted to a dataframe
                    hparams.append(csv_input)

val_mse = history.history["val_mse"]
mse = history.history["mse"]
val_mse = pd.DataFrame(data={"val_mse":val_mse})
mse = pd.DataFrame(data={"mse":mse})
epoch_validation = pd.concat([val_mse, mse], axis=1)
epoch_validation.to_csv("logs/RNN_epochs_mse.csv")

# Close .txt-log file
log.close()

print(hparams)

# Convert the list of results to a dataframe
df_log = pd.DataFrame(np.array(hparams), columns=log_cols)