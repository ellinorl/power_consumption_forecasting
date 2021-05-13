from ANN import ANN_clean
from RNN import RNN_clean
from LinearRegression import linear_regression_input_comparison
from Data import Test_data
import numpy as np
import pandas as pd
from joblib import dump, load
from tensorflow import keras

""" AI_tool.py for training and prediction of the models. """
__author__ = "Ludvig Eriksson, Jakob Johansson, Richard Johnsson, Lasse Kötz, Johan Lamm, Ellinor Lundblad"
__maintainer__ = "Ellinor Lundblad"
__email__ = "lundblad@student.chalmers.se"

class AI_tool:
    """ This class is intended as a simple tool for training models and predicting 24h power consumption with 15 minute
    timesteps."""

    def __init__(self):
        """ This method creates instances of ANNModel, LSTM_clean, RegressionModel and DataPreprocessing."""
        self.ANN_model = ANN_clean.ANNModel()
        self.RNN_model = RNN_clean.LSTM_clean()
        self.LR_model = linear_regression_input_comparison.RegressionModel()
        self.test_data = Test_data.DataPreprocessing()

    def train_model(self, model_type: str, power_consumption_data_path: list):
        """ This method trains a specific model with the data provided.
            :parameter model_type: str, RNN,ANN, or LR
            :parameter power_consumption_data_path: list of all .csv files intended to be used in training, test and
            validation of the model.
            :returns a trained model."""
        power_input = self.test_data.create_powerinput(power_consumption_data_path)
        power_input.to_csv("Data/Power_data_1")
        cleaned_data = self.test_data.create_features(["Data/Power_data_1"])
        cleaned_data.to_csv("Data/Cleaned_data_1")
        if model_type == "RNN":
            train_val_test = self.RNN_model.data_split("Data/Cleaned_data_1", 4)
            self.RNN_model.data_cleanup(train_val_test)
            self.RNN_model.create_model()
            self.RNN_model.fit_model()
            return self.RNN_model.model
        elif model_type == "ANN":
            self.ANN_model.data_preparation(4, "Data/Cleaned_data_1")
            self.ANN_model.build_fit_model()
            return self.ANN_model.model
        elif model_type == "LR":
            self.LR_model.data_preparation(4, "Data/Cleaned_data_1")
            self.LR_model.fit_model()
            return self.LR_model.model
        else:
            raise ValueError("not valid model_type argument %s, please use RNN, ANN or LR" % model_type)

    def model_prediction_next_24h(self, model, to_predict_path: str, model_type: str):
        """ Makes a prediction of the enxt 24h from the data given in to_predict_path and saves it to a .csv file
            :parameter model: a model like the one returned from load_model() or train_model().
            :parameter to_predict_path: path to the .csv file where the data needed for our prediction is.
            An example of a correctly formatted .csv file can be found in Data/example_randomly_generated_day.csv.
            :parameter model_type: str, ANN, RNN or LR"""
        to_predict = pd.read_csv(to_predict_path, parse_dates=['Date'], index_col='Date')
        predicted_labels = []
        for i in range(1, 97):  # Creates column labels for the pandas dataframe
            predicted_labels.append("predicted_{}".format(i))
        df = pd.DataFrame()
        df["Consumption (kW)"] = to_predict['Consumption (kW)']
        for i in range(-96, 97):
            if i > 0:
                df["p" + str(i * 15) + 'm'] = to_predict['Consumption (kW)'].shift(i)
        # one hot encoding for quarters
        for i in range(0, 96):
            df["Q" + str(i)] = (to_predict.index.hour * 60 + to_predict.index.minute) // 15 == i
        # Weekdays
        for i in range(0, 7):
            df["Day" + str(i)] = to_predict.index.hour == i
        df = df.tail(1)
        last_value = df.to_numpy()

        if model_type == "RNN":
            input_scaler, output_scaler = self.load_scalers_RNN()
            last_value = input_scaler.transform(last_value)
            last_value = last_value.reshape(1, 1, 200)
            predicted = model.predict(last_value)  # Make model prediction
            predicted = output_scaler.inverse_transform(predicted)
            predicted = pd.DataFrame(predicted, columns=predicted_labels)
            predicted.to_csv('logs/RNN_predicted__%s.csv' % self.RNN_model.time_stamp_string())
        elif model_type == "ANN":
            last_value = last_value.astype('float32')
            last_value = np.asarray(last_value)
            predicted = model.predict(last_value)
            predicted = pd.DataFrame(predicted, columns=predicted_labels)
            predicted.to_csv('logs/ANN_predicted__%s.csv' % self.RNN_model.time_stamp_string())
        elif model_type == "LR":
            last_value = last_value.astype('float32')
            predicted = model.predict(last_value)
            predicted = pd.DataFrame(predicted, columns=predicted_labels)
            predicted.to_csv('logs/LR_predicted__%s.csv' % self.RNN_model.time_stamp_string())
        else:
            raise ValueError("not valid model_type argument %s, please use RNN, ANN or LR" % model_type)
        return predicted

    def save_model(self, model, model_type: str):
        """ Saves a model.
            :parameter model: a model like the one returned from train_model().
            :parameter model_type: str, ANN, RNN or LR"""
        if model_type == "LR":
            dump(model, "save_model_%s.joblib" % model_type)
        else:
            model.save("save_model_%s" % model_type)
            if model_type == "RNN":
                dump(self.RNN_model.input_scaler, "save_RNN_input_scaler.joblib")
                dump(self.RNN_model.output_scaler, "save_RNN_output_scaler.joblib")
            elif model_type != "ANN":
                raise ValueError("not valid model_type argument %s, please use RNN, ANN or LR" % model_type)

    def load_model(self, model_path: str, model_type: str):
        """ Returns a model object from the path provided.
            :parameter model_path: path to a saved model.
            :parameter model_type: str, ANN, RNN or LR.
            :returns a model of the given model_type found at the model path."""
        if model_type == "LR":
            return load(model_path)
        elif model_type == "ANN" or model_type == "RNN":
            return keras.models.load_model(model_path)
        else:
            raise ValueError("not valid model_type argument %s, please use RNN, ANN or LR" % model_type)

    def load_scalers_RNN(self):
        """ Loads saved scalers from a RNN training.
            :returns a tuple of the input and output scaler."""
        return load("final_RNN_input_scaler.joblib"), load("final_RNN_output_scaler.joblib")


if __name__ == '__main__':
    tool = AI_tool()
    # ----------------------------- LSTM model ------------------------------------
    model2 = tool.load_model("final_model_RNN", "RNN")
    print("--------------------- LSTM --------------------")
    print(tool.model_prediction_next_24h(model2, "Data/example_randomly_generated_day.csv", "RNN"))

    # -------------------- ANN model -------------------------
    model2 = tool.load_model("final_model_ANN", "ANN")
    print(tool.model_prediction_next_24h(model2, "Data/example_randomly_generated_day.csv",
                                         "ANN"))

    # --------------------------- LR model ----------------------------------
    model2 = tool.load_model("final_model_LR.joblib", "LR")
    print(tool.model_prediction_next_24h(model2, "Data/example_randomly_generated_day.csv", "LR"))
