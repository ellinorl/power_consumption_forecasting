import tensorflow as tf
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping


""" ANN_clean.py for training and testing of ANN model. """
__author__ = "Ludvig Eriksson, Johan Lamm"
__maintainer__ = "Ludvig Eriksson"
__email__ = "MISSING"

class ANNModel:
    def __init__(self):
        """ Initialisation of the ANNModel class.
        Parameters:
            """
        # Name log-file after date and time as yyyymmdd_hhmm to ensure no overwriting occurs
        now = dt.datetime.now()
        log_name = "%i%02i%02i_%02i%02i" % (now.year, now.month, now.day, now.hour, now.minute)
        self.log_path = 'logs/ANN_log_' + log_name + '.csv'
        self.outdata_path = '../Data/ANN_result_' + log_name + '.csv'
        self.timings_path = 'logs/ANN_times_' + log_name + '.csv'
        Path("logs").mkdir(parents=True, exist_ok=True)

        self.log_cols = ['layers', 'nodes', 'activation', 'runtime', 'mape', 'mse']  # Add values to log last
        self.df_log = pd.DataFrame(columns=self.log_cols)  # Creating dataframe for csv conversion


    def data_preparation(self, val_i, data_path):
        """ Method for preparing the data for use in our AI-models.
        This includes splitting the data into three different datasets,
        one for training on, one for validating and one for testing,
        so that the testing data isn't seen by the model during training.

        Parameters:
            val_i (int): Choose starting point for the validation data, and is chosen as day number i in the data dataframe
            data_path (str): Path to the input data of the model.
            """

        df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
        df = df.dropna()
        df.columns = df.columns.map(str)
        df = df.astype('float32')
        self.df = df

        self.index, self.model, self.history = pd.DataFrame, keras.models.Sequential, keras.callbacks.History  # Index for dataframe, AI model and model fit history
        self.train, self.test, self.validation = pd.DataFrame, pd.DataFrame, pd.DataFrame  # Input data
        self.target, self.real, self.val_target = pd.DataFrame, pd.DataFrame, pd.DataFrame  # Output data

        valstart = int(len(self.df) * val_i / 10)
        valend = valstart + 96  # int(len(df)*(i + 1)/10)-1
        start = int(len(self.df) * ((val_i + 1) % 10) / 10)
        end = int(len(self.df) * ((val_i + 1) % 10 + 1) / 10) - 1

        temp = self.df.drop(self.df.index[start:end])
        train = temp.drop(self.df.index[valstart:valend])
        validation = self.df[valstart:valend]
        test = self.df[start:end]
        sample = test.sample(1)

        futurecols = ['f'+str(15*(i+1))+'m' for i in range(96)]

        target = pd.concat([train.pop(x) for x in futurecols], axis=1)
        real = pd.concat([test.pop(x) for x in futurecols], axis=1)
        val_target = pd.concat([validation.pop(x) for x in futurecols], axis=1)
        sample_target = pd.concat([sample.pop(x) for x in futurecols], axis=1)
        # Parameters for model and tensor generation
        # Spara undan indexet för modellen

        self.index = test.index
        self.train = np.array(train)
        self.test = np.array(test)
        self.sample = np.array(sample)
        self.validation = np.array(validation)
        self.target = np.array(target)
        self.real = np.array(real)
        self.val_target = np.array(val_target)
        self.sample_target = np.array(sample_target)

    def build_fit_model(self, l: int = 1, n: int = 128, a: str = "relu", lr: float = 0.0001, loss: str = "mse", met=["mape", "mse"], e: int=1000 , bs: int = 512, cb=EarlyStopping(monitor='val_mse', patience=25)):
        """ Method for construction, compilation, and fit of the Keras sequential model.

        Parameters:
            l (int): number of Dense layers in the model.
            n (int): number of nodes in the Dense layers.
            a (str): what activation function to use in the Dense layers.
            lr (float): learning rate for the Adams optimizer.
            loss (str): what loss function to use.
            met (list of str): a list of strings with the metrics desired for performance measuring.
            e (int): maximum number of epochs used for training the model.
            bs (int): batch size to use while training the model.
            cb (list of tensorflow.keras.Callbacks): a list of callbacks used while training.

        Returns:
            history: keras history object containing historical values of e.g. loss for each epoch of the .fit() run.
            fit_time (str): time to fit the model, in microseconds."""
        model = tf.keras.models.Sequential()
        for _ in range(l):
            model.add(tf.keras.layers.Dense(units=n, activation=a))
        model.add(tf.keras.layers.Dense(units=len(self.target[0])))

        model.compile(optimizer=keras.optimizers.Adam(lr=lr),
                      loss=loss,
                      metrics=met)

        start_time = dt.datetime.now()
        history = model.fit(self.train, self.target, bs, e, 1, cb, 0, (self.validation, self.val_target))
        end_time = dt.datetime.now()
        fit_time = str((end_time - start_time).total_seconds()*1000000)

        self.model = model

        return history, fit_time

    def predict(self, input, index=None):
        """ Method for predicting values based on input, and measuring the time it takes to predict.

        Parameters:
            input (numpy.ndarray): array of input data entries to predict based upon.
            index (pandas.DataFrame.index): Dataframe.index object to use for indexing of output, none if not specified.

        Returns:
            prediction (pandas.DataFrame): Pandas DataFrame with the predictions based of the input data,
            with index as specified
            prediction_time (str): time to predict, in microseconds."""
        start_time = dt.datetime.now()
        prediction = pd.DataFrame(self.model.predict(input), index=index)
        end_time = dt.datetime.now()
        prediction_time = str((end_time - start_time).total_seconds()*1000000)
        return prediction, prediction_time

    def optimize(self, layers: list, nodes: list, activations: list, loss: str, metrics: list, epochs: int, batch_size: int, callbacks: list, learning_rate: float):
        """ Method for fitting model with several different configurations, using Grid Search.
        When finished, the method saves the values in a .csv file according to the 'log_path' string.

        Parameters:
            layers (list of int): list of number of layers to iterate over.
            nodes (list of int): list of number of nodes to iterate over.
            activations (list of str): list of activation functions to iterate over.
            loss (str): loss function to use while training.
            metrics (list of str): metrics to measure during training.
            epochs (int): maximum number of epochs to train the model with.
            batch_size (int): batch size to use while training.
            callbacks (list of tensorflow.keras.Callbacks): list of callbacks to use during training.
            learning_rate (float): learning rate for the Adams optimizer."""
        tot_run_number = len(layers) * len(nodes) * len(activations) * 10
        run_number = 1
        for i in range(0, 10):
            self.data_preparation(i)
            results = []
            for L in layers:
                for N in nodes:
                    for A in activations:
                        print('\n\nRun number %i of %i.\n' % (run_number, tot_run_number))
                        opt = keras.optimizers.Adam(lr=learning_rate)
                        start_time = dt.datetime.now()
                        self.build_model(L, N, A, opt,
                                         loss, metrics)
                        history, fit_time = self.fit_model(epochs,
                                                           batch_size,
                                                           callbacks=callbacks)
                        end_time = dt.datetime.now()
                        run_time = end_time - start_time
                        csv_input = [L, N, A, batch_size, learning_rate, run_time] +\
                                    [history.history['val_' + m].pop() for m in metrics]
                        run_number += 1
                        results.append(csv_input)

        df_log = pd.DataFrame(np.array(results), columns=self.log_cols)
        df_log.to_csv(self.log_path, index=False)

    def run_model(self, l: int, n: int, a: str, lr: float, loss: str, met: list, e: int, bs: int, cb: list):
        """ Method for running the model one time with the specified parameters.

        Parameters:
            l (int): number of Dense layers in the model.
            n (int): number of nodes in each Dense layer.
            a (str): activation function to use in the Dense layers.
            lr (float): learning rate for the Adams optimizer.
            loss (str): loss function to use while training.
            met (list of str): metrics to measure during training.
            e (int): maximum number of epochs to use for training.
            bs (int): batch size for training the model.
            cb (list of tensorflow.keras.Callbacks): list of callbacks to use while training."""
        outdata, timings = pd.DataFrame(), []
        for i in range(0, 10):
            self.data_preparation(i)
            history, fit_time = self.build_fit_model(l, n, a, lr, loss, met, e, bs, cb)
            pred_all, t_pred_all = self.predict(self.test, self.index)
            pred_one, t_pred_one = self.predict(self.sample)
            timings.append([fit_time, t_pred_all, t_pred_one])

            pred_all = pd.DataFrame(pred_all, index=self.index)
            real_all = pd.DataFrame(self.real, index=self.index)
            data = pd.concat([pred_all, real_all], axis=1)
            outdata = pd.concat([outdata, data])

        timings = pd.DataFrame(timings, columns=['Time2Fit[us]', 'tPredAll[us]', 'tPredOne[us]'], index=None)
        timings.to_csv(self.timings_path)
        outdata.to_csv(self.outdata_path)

if __name__ == "__main__":
    ann = ANNModel('../Data/Full_data.csv')
    cb_es = EarlyStopping(monitor='val_mse', patience=10)
    ann.run_model(1, 128, 'relu', 0.0001, 'mse', ['mape', 'mse'], 5000, 128, cb_es)
