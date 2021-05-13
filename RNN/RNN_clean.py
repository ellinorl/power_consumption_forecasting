import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
import datetime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras

""" RNN_clean.py for training and testing of RNN model. """
__author__ = "Richard Johnsson, Lasse Kötz, Ellinor Lundblad"
__maintainer__ = "Ellinor Lundblad"
__email__ = "lundblad@student.chalmers.se"

class LSTM_clean():
    """ This class contains functions to run a finished LSTM model with specified parameters"""
    def __init__(self, layers=1, nodes=128, activation='sigmoid', metrics=['mape', 'mse'], epochs=500, batch_size=32, learning_rate=0.0001, patience=10):
        """All default parameters in this class are the best performing ones in our optimization on our specific dataset.
        They can be tuned for better performance on new data.
        :param layers: int , number of LSTM layers in our model
        :param nodes: int, number of nodes in our LSTM layers and extra Dense layer
        :param activation: str, activation function used in LSTM layers
        :param metrics: list, metrics to print in logfile
        :param: epochs: maximum number of epochs in model training
        :param: batch_size: batch size used in model training
        :param: learning_rate: learning rate for optimizer (Adam),
        :param: patience: number of non-improved epochs before ending training """

        self.layers = layers
        self.nodes = nodes
        self.activation = activation
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cb_es = EarlyStopping(monitor='val_mse', patience=patience)
        self.outdata = pd.DataFrame()

    def data_split(self, data_path:str, split:int):
        """ Splits the data into a preferred split
            :param data_path: str, path to .csv file with data
            :param spil: int, number between 0 and 9 used for splitting the data appropriately.
            :returns list of pandas dataframes, [train, validation, test]. """
        i = split
        df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
        df = df.dropna()
        df.columns = df.columns.map(str)
        df = df.astype('float32')

        valstart = int(len(df) * i / 10)
        valend = int(len(df) * (i + 1) / 10) - 1
        start = int(len(df) * ((i + 5) % 10) / 10)
        end = int(len(df) * ((i + 5) % 10 + 1) / 10) - 1

        temp = df.drop(df.index[start:end])
        train = temp.drop(df.index[valstart:valend])
        validation = df[valstart:valend]
        test = df[start:end]
        print(train)
        print(validation)
        print(test)
        return [train, validation, test]


    def data_cleanup(self, train_validation_test:list):
        """ Cleans data and prepares for input to LSTM model. Saves train, validation, test, target, val_target and real
        as class variables.
            :param train_validation_test, list of pandas datafram [train, validation, test] like the one returned in
            data_split."""

        if len(train_validation_test) != 3:  # Ensure that the list is of correct format
            raise ValueError("train_validation_test not correct format, please ensure it is a list of three elements")

        # Getting train, validation and test from input list
        train = train_validation_test[0]
        validation = train_validation_test[1]
        test = train_validation_test[2]

        # We want future values to be used as targets for the model, so they are separated from the input sets,
        # and saved in target, real, and val_target.
        futurecols = []
        for i in range(0, 96):
            futurecols.append('f' + str(15 * (i + 1)) + 'm')
        print(futurecols)

        # pop all columns with name "fxxm" from original datasets into seperate datasets
        target = pd.concat([train.pop(x) for x in futurecols], axis=1)
        real = pd.concat([test.pop(x) for x in futurecols], axis=1)
        val_target = pd.concat([validation.pop(x) for x in futurecols], axis=1)

        self.index = test.index  # Save datetime index of test for later usage when recreating pandas dataframe.

        # From pandas dataframe to numpy to prepare for rescaling with MinMaxScaler()
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

        # Rescaling of data with MinMaxScaler() for usage in LSTM model. Do not forget to inverse transform scaled data
        # after prediction.
        self.input_scaler = MinMaxScaler(feature_range=(0, 1))
        train = self.input_scaler.fit_transform(train) # fits scaler to training data and transforms training data
        test = self.input_scaler.transform(test)
        validation = self.input_scaler.transform(validation)
        self.output_scaler = MinMaxScaler(feature_range=(0,1))
        self.target = self.output_scaler.fit_transform(target) # refits scaler to target data and saves scaled target data to self.target
        self.val_target = self.output_scaler.transform(val_target) # saves transformed validation target
        self.real = self.output_scaler.transform(real) # saves transformed real (target for test)

        self.no_steps = validation.shape[1]  # no of steps back that is later used as input for LSTM

        # Reshaping of all data that is used as input for LSTM to 3D array (as that is the required shape).
        self.train = np.reshape(train, newshape=(train.shape[0], 1, train.shape[1]))
        self.test = np.reshape(test, newshape=(test.shape[0], 1, test.shape[1]))
        self.validation = np.reshape(validation, newshape=(validation.shape[0], 1, validation.shape[1]))

    def create_model(self):
        """ Creates and saves a keras mode to self.model. Uses the parameters supplied in __init__."""
        self.model = tf.keras.models.Sequential()
        for L in range(self.layers):
            if L == self.layers - 1: # Last LSTM layer should not return sequences.
                self.model.add(LSTM(units=self.nodes, input_shape=(1, self.no_steps), return_sequences=False,
                               activation=self.activation, name='LSTM_lay_%i' % L))
            else:
                self.model.add(LSTM(self.nodes, input_shape=(1, self.no_steps), return_sequences=True,
                               activation=self.activation))
            self.model.add(Dropout(0.2, name="Dropout_lay_%i" % L)) # Adds a dropout layer of 20% after each LSTM layer
        self.model.add(Dense(self.nodes, activation="relu", name="Dense_lay_1"))
        self.model.add(Dense(units=self.target.shape[1], name="Output_lay"))
        opti = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=opti,
                      loss='mse',
                      metrics=self.metrics)
        print(self.model.summary()) # Prints model summary

    def fit_model(self, training_and_validation_set=None):
        """ Fits self.model object to training set while comparing to valiation set.
        :param training_and_validation_set: type: list of four elements,
            training, training_target, validation, validation_target.
            If no training_and_validation_set is given, the model uses instance attributes self.train, self.target,
            self.validation and self.val_target. These are created in data_cleanup()."""
        if not training_and_validation_set: # No training_and_validation_set is given, use instance attributes instead.
            training_set = self.train
            target_set = self.target
            validation_set = self.validation
            val_target = self.val_target
        elif len(training_and_validation_set) != 4:
            raise ValueError("Please provide a valid traing_and_validation_set input variable.")
        else: # Get the sets from the training_and_validation_set list.
            training_set = training_and_validation_set[0]
            target_set = training_and_validation_set[1]
            validation_set = training_and_validation_set[2]
            val_target = training_and_validation_set[3]
        # Run model.fit with parameters epoch, batch_size and callbacks given in __init__.
        self.model.fit(training_set, target_set,
                                            epochs=self.epochs,
                                            validation_data=(validation_set, val_target),
                                            batch_size=self.batch_size, callbacks=self.cb_es)

    def time_stamp_string(self):
        """:returns str of timestamp for logfile saving"""
        return str(datetime.datetime.now().year) + str(datetime.datetime.now().month) + str(
            datetime.datetime.now().day) \
                  + "_" + str(datetime.datetime.now().hour) + "_" + str(datetime.datetime.now().minute) + \
                  "_" + str(datetime.datetime.now().second)

    def model_prediction_and_comparison(self, test_set=None, real=None):
        """ Makes a prediction for the coming 24h with 15-min time steps and compares it to the real value.
            :param test_set: numpy array in 3D with last 24h 15-min power consumption values and one-hot encoding for
            day of week and time of day (15-minute of day). If none is given, self.test from data_cleanup is used.
            :param real: numpy array in 2D with matching coming 24h 15-min power consumption values. For comparison with
             model prediction. If none is given, self.real from data_cleanup() is used.

             If no parameters are given, ensure the self.real and self.test exist by running data_cleanup()"""

        if real is None:
            real = self.real
        real = self.output_scaler.inverse_transform(real)
        predicted_labels = []
        real_labels = []
        for i in range(1, 97):  # Creates column labels for the pandas dataframe
            predicted_labels.append("predicted_{}".format(i))
            real_labels.append("real_{}".format(i))
        if test_set is None:  # If no test_set is passed, use self.test
            test_set = self.test
            index = self.index  # self.index is saved in data_cleanup(), which gives a datetime() index for test_set.
            predicted = self.model.predict(test_set)
            predicted = self.output_scaler.inverse_transform(predicted)
            predicted = pd.DataFrame(predicted, index=index, columns=predicted_labels)
            real = pd.DataFrame(real, index=index, columns=real_labels)
        else:
            predicted = self.model.predict(test_set)
            predicted = self.output_scaler.inverse_transform(predicted)
            predicted = pd.DataFrame(predicted, columns=predicted_labels)
            real = pd.DataFrame(real, columns=real_labels)

        outdata = pd.concat([real, predicted], axis=1)
        outdata = outdata.dropna()
        print(outdata)
        outdata.to_csv('logs/RNN_diff_%s.csv' % self.time_stamp_string())

    def model_prediction_only(self, to_predict):
        """ Makes a prediction for the coming 24h with 15-min time steps
            :param to_predict: numpy array in 3D with last 24h 15-min power consumption values and one-hot encoding for
            day of week and time of day (15-minute of day).
            Saves prediction to .csv file"""
        predicted_labels = []
        for i in range(1, 97):  # Creates column labels for the pandas dataframe
            predicted_labels.append("predicted_{}".format(i))
        predicted = self.model.predict(to_predict)  # Make model prediction
        predicted = self.output_scaler.inverse_transform(predicted)
        predicted = pd.DataFrame(predicted, columns=predicted_labels)
        predicted.to_csv('logs/RNN_predicted__%s.csv' % self.time_stamp_string())
        return predicted

if __name__ == "__main__":
    final_LSTM = LSTM_clean()
    train_val_test = final_LSTM.data_split("../Data/Full_data.csv", 4) # Creates data split
    final_LSTM.data_cleanup(train_val_test)
    final_LSTM.create_model()
    final_LSTM.fit_model()

    # For testing on given split below
    final_LSTM.model_prediction_and_comparison()

    # For testing on one sample time below
    print("LSTM test shape {}".format(final_LSTM.test.shape))
    print("LSTM test shape  new {}".format(final_LSTM.test[150][0].reshape(1, 1, 200).shape))
    print("LSTM real shape {}".format(final_LSTM.real.shape))
    print("LSTM real shape  new {}".format(final_LSTM.real[150].reshape(1, 96).shape))
    # Takes the 150th sample and reformats for prediction using the trained model (needs 3D array even
    # if only one time is used)
    final_LSTM.model_prediction_and_comparison(final_LSTM.test[150][0].reshape(1, 1, 200), final_LSTM.real[150].reshape(1, 96))
    final_LSTM.model_prediction_only(final_LSTM.test[150][0].reshape(1, 1, 200))