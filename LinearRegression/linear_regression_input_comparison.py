import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import datetime
from Results import results
from matplotlib import rc

""" linear_regression_input_comparison.py for all training and testing of linear regression model. """
__author__ = "Ellinor Lundblad, Jakob Johansson"
__maintainer__ = "Ellinor Lundblad"
__email__ = "lundblad@student.chalmers.se"

class RegressionModel():
    """ Class used for linear regression model fitting and prediction, as well as comparison
     between different types of linear regression models."""
    def __init__(self, split=None):
        self.model = LinearRegression()
        self.predictions = None
        self.errors = []
        self.times = {}
        # Som settings to make the plots look nicer
        rc('font', **{'family': 'serif', 'sans-serif': ['computer modern sans-serif']})
        plt.rcParams.update({'font.size': 20})
        rc('text', usetex=True)
        self.colors = ['#173F5F','#F6D55C','#3CAEA3','#20639B', "#ED553B"]

    def data_preparation(self, split:int, data_path):
        # Split the dataset into training and test set
        df = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        if split == None:
            size = int(0.75 * len(df))
            self.X_train, self.X_test = df[0:size], df[size:len(df)]
        else:
            test_start = int(len(df) * split / 10)
            test_end = int(len(df) * (split + 1) / 10) - 1
            start = int(len(df) * ((split + 5) % 10) / 10)
            end = int(len(df) * ((split + 5) % 10 + 1) / 10) - 1
            self.X_train = df.drop(df.index[start:end])
            self.X_test = df[start:end]
            print("X_test shape", self.X_test.shape)
            print("X_train shape", self.X_train.shape)
            # self.X_train, self.X_test = df[start, end], df[test_start, test_end]

        futurecols = []
        for i in range(1, 97):
            futurecols.append("f" + str(i * 15) + "m")
        print(futurecols)

        # Split the test and training set into input (X) and target (y) values
        self.y_test = pd.concat([self.X_test.pop(x) for x in futurecols], axis=1)
        self.y_train = pd.concat([self.X_train.pop(x) for x in futurecols], axis=1)

    def fit_model(self):
        """ fits the model (given in self.model) to the training set """
        start_time = datetime.datetime.now()
        self.model.fit(self.X_train, self.y_train)
        end_time = datetime.datetime.now()
        self.fitting_time = str(end_time - start_time)

    def make_prediction(self):
        start_time = datetime.datetime.now()
        predicted = self.model.predict(self.X_test)
        self.prediction_time = str(datetime.datetime.now() - start_time)
        self.prediction = pd.DataFrame(predicted, index=self.X_test.index)
        self.outdata = pd.concat([self.y_test, self.prediction], axis=1)    # stores the prediction and actual value to a dataframe

    def saveOutdata(self, model_name=None):
        """ save self.outdata (model prediction + actual value) to a log file.
        Parameters:
            model_name: str
            the name of the model to save, helps keep track of the model files.
            if name is given the filepath is stored in self.files dictionary with name as key"""
        if model_name != None:
            file_path = 'logs/%sdiff_lr_%s.csv' % (model_name, self.get_time_stamp_string())
            self.outdata.to_csv(file_path)
            self.files[model_name] = file_path
            self.times[model_name] = [self.fitting_time, self.prediction_time]
        else:
            self.outdata.to_csv('logs/diff_lr_%s.csv' % self.get_time_stamp_string())
        return self.outdata

    def loop_models(self):
        """ Loops the over the appropriate model and saves their predictions to csv files """
        self.files = {}
        self.models = {"Ridge regression": Ridge(alpha=100), "Lasso regression": Lasso(alpha=100), "Linjär regression": LinearRegression(), "PLSR": PLSRegression(n_components=10), "PCR" : make_pipeline(StandardScaler(), PCA(n_components=20), LinearRegression())}
        for key, val in self.models.items():
            self.model = val
            self.fitOurModel()
            self.saveOutdata(model_name=key)
            #self.logErrors()
        print(self.times)

    def get_time_stamp_string(self):
        """ Returns an appropriate string format for timestamp now, for usage in log file names"""
        now = datetime.datetime.now()
        return '%i%i%i_%i%i%i' % (now.year, now.month, now.day, now.hour, now.minute, now.second)

    def plot_model_comparison(self):
        """ Plots the error of the models to compare. Only runnable after loop_models """
        mape = {}
        for index, model in enumerate(self.models.keys()):
            # Uses the result class to process the output \files and gets MAPE over every 15-min intervall
            mape[model] = results.Results(self.files[model]).APE.mean()
            plt.plot(mape[model], label=model, c=self.colors[index]) # Lineplot for each model MAPE
        ticks = np.linspace(0, 96, 25)
        labels = []
        for tick in ticks:
            label = int(tick / 4)
            label = '%i' % label
            labels.append(label)

        plt.title("Jämförelse av olika linjära regressionsmodellers MAPE")
        plt.xticks(ticks=np.linspace(0, 96, 25), labels=labels, size=12)
        plt.tight_layout()
        plt.xlabel("Timmar framåt för prognos")
        plt.ylabel("MAPE")
        plt.legend(fontsize=12)
        plt.show()

if __name__ == "__main__":
    regression_model = RegressionModel()
    regression_model.loop_models()
    regression_model.plot_model_comparison()

    """
    diff_from_cross_validation = pd.DataFrame()
    for i in range(10):
        reg_model = RegressionModel(split=i)
        reg_model.fitOurModel()
        diff_from_cross_validation = pd.concat([diff_from_cross_validation, reg_model.outdata])
    diff_from_cross_validation.to_csv("logs/linjarreg_cross_val_diff_%s.csv" % reg_model.get_time_stamp_string())
    """