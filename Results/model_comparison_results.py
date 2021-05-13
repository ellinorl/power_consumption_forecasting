from results import Results
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

""" model_comparison_results.py for comparison of model results. """
__author__ = "Ludvig Eriksson, Jakob Johansson, Richard Johnsson, Lasse Kötz, Johan Lamm, Ellinor Lundblad"
__maintainer__ = "Ellinor Lundblad"
__email__ = "lundblad@student.chalmers.se"

class AllResults():
    """ Class for creating comparison plots and all plots for the different models """
    def __init__(self, ANN_file, RNN_file, LR_file):
        """ :param ANN_file: string, file path to ANN diff csv file
            :param RNN_file: string, file path to RNN diff csv file
            :param LR_file: string, file path to LR diff csv file"""

        # Creates seperate objects of Result type class for individual plots and data computation
        self.ANN_res = Results(file_path=ANN_file, model_name="ANN")
        self.RNN_res = Results(file_path=RNN_file, model_name="RNN")
        self.LR_res = Results(file_path=LR_file, model_name="LR")
        self.models = [self.ANN_res, self.RNN_res, self.LR_res] # List of models for looping
        self.colors = {"ANN": "#ED553B", "RNN": "#3CAEA3", "LR": "#173F5F"} # Color dictionary for plots

    def MAPE_time_ahead_comparison(self):
        """ Plots comparison of prediction percentage error for all three models. """
        print('\n\n====================\nMAPE_time_ahead_comparison()\n====================')
        ANN_mape = self.ANN_res.APE.mean()
        RNN_mape = self.RNN_res.APE.mean()
        LR_mape = self.LR_res.APE.mean()
        plt.plot(ANN_mape, label="ANN", c=self.colors["ANN"])
        plt.plot(RNN_mape, label="RNN", c=self.colors["RNN"])
        plt.plot(LR_mape, label="Linjär regression", c=self.colors["LR"])

        # Code below makes plot look nicer
        ticks = np.linspace(0, 88, 12)
        labels = []
        for tick in ticks:
            label = int(tick / 4)
            label = '%i' % label
            labels.append(label)
        plt.xticks(ticks=ticks, labels=labels)

        plt.ylim(0, 0.15)
        ticks = np.linspace(0, 0.15, 6)
        labels = []
        for i in ticks:
            label = "{:.1f}".format(i * 100)
            labels.append(label)
        plt.yticks(ticks=ticks, labels=labels)

        plt.ylabel("MAPE [\%]")
        plt.xlabel("Tid framåt [h]")
        plt.title("MAPE jämförelse")
        plt.xlim(0, 95)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.show()

    def mape_time_ahead_all(self):
        print('\n\n====================\nmape_time_ahead_all()\n====================')
        for model in self.models:
            model.abs_confidenceInterval()

    def sample_day_all(self):
        print('\n\n====================\nsample_day_all()\n====================')
        for model in self.models:
            model.sample_day()

    def abs_confidence_interval_days(self):
        print('\n\n====================\nabs_confidence_interval_days()\n====================')
        for model in self.models:
            model.abs_confidenceInterval_days()

    def get_mape_all_three_midnight(self):
        """ :returns Pandas dataframe of mape over all hours after midnight"""
        print('\n\n====================\nget_mape_all_three_midnight()\n====================')
        mape = pd.DataFrame(columns=["ANN", "RNN", "LR"])
        for model in self.models:

            errors = model.errors.copy()
            pe = model.APE.copy()
            errors['min_from_midnight'] = errors.index.hour * 60 + errors.index.minute
            pe['min_from_midnight'] = errors.index.hour * 60 + errors.index.minute

            for i in range(0, 96):
                newdata = (pe.loc[(errors['min_from_midnight'] >= i * 15) & (pe['min_from_midnight'] <= i * 15 + 14)])
                newdata = newdata.drop(columns=['min_from_midnight'])
                mape[model.model] = newdata.mean(axis=0)
        return mape

    def plot_mape_from_midnight_comparison(self):
        """ Plots mape from midnight for all three models."""
        print('\n\n====================\nplot_mape_from_midnight_comparison()\n====================')
        mape = self.get_mape_all_three_midnight()
        for model in self.models:
            plt.plot(mape[model.model], label=model.model, c=self.colors[model.model])

        ticks = np.linspace(0, 88, 12)
        labels = []
        for tick in ticks:
            label = int(tick / 4)
            label = '%02i:00' % label
            labels.append(label)
        plt.xticks(ticks=ticks, labels=labels, rotation=45)

        plt.ylim(0, 0.2)
        ticks = np.linspace(0, 0.2, 6)
        labels = []
        for i in ticks:
            label = "{:.1f}".format(i * 100)
            labels.append(label)
        plt.yticks(ticks=ticks, labels=labels)

        plt.ylabel("MAPE [\%]")
        plt.xlabel("Tid på dygn [hh:mm]")
        plt.title("MAPE jämförelse")
        plt.xlim(0, 95)
        plt.legend(fontsize=14)

        plt.tight_layout()

        plt.show()

    def plot_histogram_all(self):
        print('\n\n====================\nplot_histogram_all()\n====================')
        for model in self.models:
            model.guassianDistribution()

    def plot_heatmap_all(self):
        print('\n\n====================\nplot_heatmap_all()\n====================')
        for model in self.models:
            model.heatMap()

    def get_mape_all(self):
        print('\n\n====================\nget_mape_all()\n====================')
        mape = {}
        for model in self.models:
            mape[model.model] = model.APE.mean()
        return mape

    def persistance_first_hour(self):
        print('\n\n====================\npersistance_first_hour()\n====================\n\n')
        self.LR_res.persistance_first_hour()


    def baseline_plot_all(self):
        baseline = [np.mean(abs(self.LR_res.APE_prev_day.iloc[:, i] * 100)) for i in range(len(self.LR_res.error_prev_day.iloc[0]))]
        print('Baseline:\n  medelvärde: %.3f\n         min: %.3f\n         max: %.3f' % (np.mean(baseline), np.min(baseline), np.max(baseline)))

        #plt.figure(figsize=(15, 8))
        plt.plot(baseline, '#F6D55C', label='Baseline')
        for model in self.models:
            prediction = [np.mean(model.APE.iloc[:, i] * 100) for i in range(len(model.abs_errors.iloc[0]))]
            plt.plot(prediction, c=self.colors[model.model], label=model.model)
        plt.legend(loc='lower right', fontsize=14)

        ticks = np.linspace(0, 88, 12)
        labels = []
        for tick in ticks:
            label = int(tick / 4)
            label = '%i' % label
            labels.append(label)

        plt.xticks(ticks=ticks, labels=labels)
        plt.title('Baseline vs Prognos')
        plt.xlabel('Timme framåt i tiden [h]')
        plt.ylabel('Fel i MAPE [\%]')
        plt.ylim([0, 20])
        plt.xlim([0, 96])
        plt.tight_layout()
        plt.show()

        print('\nFunction running: baseline_plot_all()\n')

if __name__ == '__main__':
    res = AllResults(ANN_file="../Data/diff_ann_long.csv", RNN_file="../Data/diff_LSTM_long_1.csv",
               LR_file="../LinearRegression/logs/linjarreg_cross_val_diff_2021513_15112.csv")
    mape = res.get_mape_all()
    print(mape)
    av_mape = {}
    for key, values in mape.items():
        av_mape[key] = np.mean(values)
    print(av_mape)
    #res.plot_heatmap_all()
    #res.baseline_plot_all()
    #res.persistance_first_hour()
    #res.MAPE_time_ahead_comparison()
    res.mape_time_ahead_all()
    res.abs_confidence_interval_days()
    #res.sample_day_all()
    res.plot_mape_from_midnight_comparison()
    res.plot_histogram_all()
