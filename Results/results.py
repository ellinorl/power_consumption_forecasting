import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from scipy.stats import norm
import numpy as np
import math
from matplotlib import rc
from scipy.optimize import curve_fit
import datetime

""" results.py for creation of results plots of all model prediction. """
__author__ = "Ludvig Eriksson, Jakob Johansson, Richard Johnsson, Lasse Kötz, Johan Lamm, Ellinor Lundblad"
__maintainer__ = "Ellinor Lundblad"
__email__ = "lundblad@student.chalmers.se"

colors = ['#ED553B','#F6D55C','#3CAEA3']
index = 2
model = "ANN" # "LR" or "ANN" or "LSTM"


class Results():
    rc('font', **{'family': 'serif','sans-serif': ['computer modern sans-serif']})
    plt.rcParams.update({'font.size': 24})
    rc('text', usetex=True)

    def __init__(self, file_path=None, model_name=None):
        """ Class for creating plots for a specific model.
            :param file_path: string, path of diff file for given model
            :param model_name: string, ANN, RNN or LR"""

        self.colors = {"ANN": "#ED553B", "RNN": "#3CAEA3", "LR": "#173F5F"}
        self.outdata = pd.read_csv("../Data/diff_LSTM_long_1.csv", parse_dates=True, index_col="Date")
        if file_path != None:
            self.outdata = pd.read_csv(file_path, parse_dates=True, index_col="Date")
        if model_name != None:
            self.model = model_name
        else:
            self.model = model
        self.outdata = self.outdata.dropna()
        self.errors = pd.DataFrame(index=self.outdata.index)
        self.mse = pd.DataFrame(index=self.outdata.index)
        self.real = pd.DataFrame(index=self.outdata.index)
        self.predicted = pd.DataFrame(index=self.outdata.index)
        self.percentage_error = pd.DataFrame(index=self.outdata.index)
        self.APE = pd.DataFrame(index=self.outdata.index)
        self.error_prev_day = pd.DataFrame(index=self.outdata.index[96*15:])
        self.APE_prev_day = pd.DataFrame(index=self.error_prev_day.index)
        self.diff_prediction_prev_day = pd.DataFrame(index=self.outdata.index[96*15:])
        self.hold_errors = pd.DataFrame(index=self.outdata.index)
        self.change_ape = pd.DataFrame(index=self.outdata.index)
        self.abs_errors = pd.DataFrame(index=self.outdata.index)
        self.abs_hold_errors = pd.DataFrame(index=self.outdata.index)

        for i in range(0, len(self.outdata.columns) // 2):
            self.errors[str(i)] = (self.outdata.iloc[:, i] - self.outdata.iloc[:, i + len(self.outdata.columns) // 2])
            self.abs_errors[str(i)] = abs(self.errors.iloc[:, i])
            self.mse[str(i)] = (self.outdata.iloc[:, i] - self.outdata.iloc[:, i + len(self.outdata.columns) // 2]) ** 2
            self.real[str(i)] = self.outdata.iloc[:, i]
            self.predicted[str(i)] = self.outdata.iloc[:, i + len(self.outdata.columns) // 2]
            self.percentage_error[str(i)] = (self.outdata.iloc[:, i] - self.outdata.iloc[:, i + len(self.outdata.columns) // 2]) / self.outdata.iloc[:, i]
            self.APE[str(i)] = abs((self.outdata.iloc[:, i] - self.outdata.iloc[:, i + len(self.outdata.columns) // 2]) / self.outdata.iloc[:, i])
            self.error_prev_day[str(i)] = self.real.iloc[:len(self.real)-96*15, i].values - self.real.iloc[96*15:, i].values
            self.APE_prev_day[str(i)] = self.error_prev_day.iloc[:, i].values / self.real.iloc[96*15:, i].values
            self.diff_prediction_prev_day[str(i)] = abs(self.errors.iloc[96*15:, i]) - abs(self.error_prev_day.iloc[:, i])

        for i in range(0, len(self.outdata.columns) // 2 - 1):
            self.hold_errors[str(i)] = self.outdata.iloc[:, i]-self.outdata.iloc[:, 0]
            self.abs_hold_errors[str(i)] = abs(self.hold_errors.iloc[:, i])
            self.change_ape[str(i)] = (abs(self.errors.iloc[:, i+1])-abs(self.hold_errors.iloc[:, i]) ) #/ abs(self.hold_errors.iloc[:, i])

    def guassianDistribution(self):
        error_array = self.percentage_error.to_numpy()
        #print(len(error_array))
        bins = int(math.sqrt(len(error_array)))
        error_array = error_array.ravel()
        plt.hist(error_array, bins=bins, density=True, color=self.colors[self.model], edgecolor='black')
        (mu, sigma) = norm.fit(error_array)

        x = np.linspace(-4 * sigma + mu, 4 * sigma + mu, bins)
        plt.xlim(-0.75, 0.75)
        y = norm.pdf(x, mu, sigma)
        plt.plot(x, y, c="black", linewidth=2)
        """ticks = []
        for i in x:
            tick = i * 100
            ticks.append(tick)
        plt.xticks(ticks=x, labels=ticks)"""
        plt.xlabel("Procentuellt fel")
        #print('$\mu={:.4f}\pm{:.4f},\ \sigma=%.3f$')
        plt.title('Histogram av procentuellt fel, $\mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
        plt.tight_layout()
        plt.show()

    def confidenceInterval(self):
        # intervall för en specifik prognos
        pred = []
        m_3_sigma = []
        m_1_sigma = []
        m_2_sigma = []
        p_1_sigma = []
        p_2_sigma = []
        p_3_sigma = []
        x = []

        for i in range(0, 96):
            x.append(i)
            m_3_sigma.append(np.percentile(self.APE.iloc[:, i], 100-0.13))
            m_2_sigma.append(np.percentile(self.APE.iloc[:, i], 100-(2.14+0.13)))
            m_1_sigma.append(np.percentile(self.APE.iloc[:, i], 100-(13.59+2.14+0.13)))
            p_1_sigma.append(np.percentile(self.APE.iloc[:, i], 13.59+2.14+0.13))
            p_2_sigma.append(np.percentile(self.APE.iloc[:, i], 2.14+0.13))
            p_3_sigma.append(np.percentile(self.APE.iloc[:, i], 0.13))


        mape = self.APE.mean()
        m_1_sigma = np.array(m_1_sigma)
        m_2_sigma = np.array(m_2_sigma)
        m_3_sigma = np.array(m_3_sigma)
        p_1_sigma = np.array(p_1_sigma)
        p_2_sigma = np.array(p_2_sigma)
        p_3_sigma = np.array(p_3_sigma)
        #plt.plot(real, color='r')
        plt.fill_between(x, p_3_sigma, m_3_sigma, color='#bbcfe0', label="$ \subset \mu \pm 3\sigma$")
        plt.fill_between(x, p_2_sigma, m_2_sigma, color='#8caecb', label="$\subset \mu \pm 2\sigma$")
        plt.fill_between(x, p_1_sigma, m_1_sigma, color='#6b97bc', label="$ \subset \mu \pm \sigma$")
        plt.plot(mape, c="#173F5F", label="$\mu $ (MAPE)")

        ticks = x[::4]
        labels = []
        for i in ticks:
            if i == 0:
                labels.append("0")
            else:
                labels.append(str(int(i/4)))
        plt.xticks(ticks=ticks, labels=labels)
        plt.ylim(0, 1)
        ticks = np.linspace(0, 1, 11)
        labels = []
        for i in ticks:
            label = "{:.1f}".format(i*100)
            labels.append(label)
        plt.yticks(ticks=ticks, labels=labels)
        plt.xlabel("Tid [h]")
        plt.ylabel("MAPE [\%]")

        plt.legend(fontsize=14)
        plt.title("MAPE och konfidensinterval för fel")

        plt.show()

    def abs_confidenceInterval(self):
        # intervall för en specifik prognos
        pred = []
        m_3_sigma = []
        m_1_sigma = []
        m_2_sigma = []
        p_1_sigma = []
        p_2_sigma = []
        p_3_sigma = []
        x = []

        for i in range(0, 96):
            x.append(i)
            # m_3_sigma.append(np.percentile(self.APE.iloc[:, i], 100-0.13))
            m_2_sigma.append(np.percentile(self.APE.iloc[:, i], 100 - (2.14 + 0.13)))
            m_1_sigma.append(np.percentile(self.APE.iloc[:, i], 100 - (13.59 + 2.14 + 0.13)))
            p_1_sigma.append(np.percentile(self.APE.iloc[:, i], 13.59 + 2.14 + 0.13))
            p_2_sigma.append(np.percentile(self.APE.iloc[:, i], 2.14 + 0.13))
            # p_3_sigma.append(np.percentile(self.APE.iloc[:, i], 0.13))

        mape = self.APE.mean()
        m_1_sigma = np.array(m_1_sigma)
        m_2_sigma = np.array(m_2_sigma)
        # m_3_sigma = np.array(m_3_sigma)
        p_1_sigma = np.array(p_1_sigma)
        p_2_sigma = np.array(p_2_sigma)
        # p_3_sigma = np.array(p_3_sigma)

        # plt.plot(real, color='r')
        # plt.fill_between(x, p_3_sigma, m_3_sigma, color='#bbcfe0', label="$ \subset \mu \pm 3\sigma$")
        plt.fill_between(x, p_2_sigma, m_2_sigma, color=self.lighten_color(self.colors[self.model], 0.33),
                         label="$\subset \mu \pm 2\sigma$")
        plt.fill_between(x, p_1_sigma, m_1_sigma, color=self.lighten_color(self.colors[self.model], 0.66),
                         label="$ \subset \mu \pm \sigma$")
        plt.plot(mape, c=self.colors[self.model], label="$\mu $ (MAPE)")

        #print(np.mean(mape))

        ticks = x[::8]
        labels = []
        for i in ticks:
            if i == 0:
                labels.append("0")
            else:
                labels.append(str(int(i / 4)))
        plt.xticks(ticks=ticks, labels=labels)
        print(ticks)

        plt.ylim(0, 0.5)
        ticks = np.linspace(0, 0.5, 6)
        labels = []
        for i in ticks:
            label = "{:.1f}".format(i * 100)
            labels.append(label)
        plt.yticks(ticks=ticks, labels=labels)
        plt.xlabel("Tid framåt [h]")
        plt.ylabel("MAPE [\%]")
        plt.xlim(0, 95)
        plt.legend(fontsize=14)
        plt.title("MAPE och varians för fel")
        plt.tight_layout()
        plt.show()

    def abs_confidenceInterval_short(self):
        # intervall för en specifik prognos
        tid_framat = 5
        pred = []
        m_3_sigma = []
        m_1_sigma = []
        m_2_sigma = []
        p_1_sigma = []
        p_2_sigma = []
        p_3_sigma = []
        x = []

        for i in range(0, tid_framat):
            x.append(i)
            # m_3_sigma.append(np.percentile(self.APE.iloc[:, i], 100-0.13))
            m_2_sigma.append(np.percentile(self.APE.iloc[:, i], 100 - (2.14 + 0.13)))
            m_1_sigma.append(np.percentile(self.APE.iloc[:, i], 100 - (13.59 + 2.14 + 0.13)))
            p_1_sigma.append(np.percentile(self.APE.iloc[:, i], 13.59 + 2.14 + 0.13))
            p_2_sigma.append(np.percentile(self.APE.iloc[:, i], 2.14 + 0.13))
            # p_3_sigma.append(np.percentile(self.APE.iloc[:, i], 0.13))

        mape = self.APE.mean()
        m_1_sigma = np.array(m_1_sigma)
        m_2_sigma = np.array(m_2_sigma)
        # m_3_sigma = np.array(m_3_sigma)
        p_1_sigma = np.array(p_1_sigma)
        p_2_sigma = np.array(p_2_sigma)
        # p_3_sigma = np.array(p_3_sigma)

        # plt.plot(real, color='r')
        # plt.fill_between(x, p_3_sigma, m_3_sigma, color='#bbcfe0', label="$ \subset \mu \pm 3\sigma$")
        plt.fill_between(x, p_2_sigma, m_2_sigma, color=self.lighten_color(self.colors[self.model], 0.33),
                         label="$\subset \mu \pm 2\sigma$")
        plt.fill_between(x, p_1_sigma, m_1_sigma, color=self.lighten_color(self.colors[self.model], 0.66),
                         label="$ \subset \mu \pm \sigma$")
        plt.plot(mape.iloc[0:tid_framat], c=self.colors[self.model], label="$\mu $ (MAPE)")
        print(mape[0:4])

        ticks = x
        labels = []
        for i in ticks:
            if i == 0:
                labels.append("0")
            else:
                labels.append(str(int(i * 15)))

        plt.xticks(ticks=ticks, labels=labels)
        plt.ylim(0, 0.5)
        ticks = np.linspace(0, 0.5, 6)
        labels = []
        for i in ticks:
            label = "{:.1f}".format(i * 100)
            labels.append(label)
        plt.yticks(ticks=ticks, labels=labels)
        plt.xlabel("Tid [min]")
        plt.ylabel("MAPE [%]")
        plt.xlim(0, 4)
        plt.legend(fontsize=14)
        plt.title("MAPE och varians för fel")
        plt.grid()
        plt.show()

    def abs_confidenceInterval_days(self):
        errors = self.errors.copy()
        pe = self.APE.copy()
        errors['min_from_midnight'] = errors.index.hour * 60 + errors.index.minute
        pe['min_from_midnight'] = errors.index.hour * 60 + errors.index.minute

        for i in range(0, 96):
            newdata = (pe.loc[(errors['min_from_midnight'] >= i * 15) & (pe['min_from_midnight'] <= i * 15 + 14)])
            newdata = newdata.drop(columns=['min_from_midnight'])

            mape = newdata.mean(axis=0)
        ##----------------------------
        # intervall för en specifik prognos

        m_3_sigma = []
        m_1_sigma = []
        m_2_sigma = []
        p_1_sigma = []
        p_2_sigma = []
        x = []

        for i in range(0, 96):
            x.append(i)
            m_2_sigma.append(np.percentile(newdata.iloc[:, i], 100 - (2.14 + 0.13)))
            m_1_sigma.append(np.percentile(newdata.iloc[:, i], 100 - (13.59 + 2.14 + 0.13)))
            p_1_sigma.append(np.percentile(newdata.iloc[:, i], 13.59 + 2.14 + 0.13))
            p_2_sigma.append(np.percentile(newdata.iloc[:, i], 2.14 + 0.13))

        m_1_sigma = np.array(m_1_sigma)
        m_2_sigma = np.array(m_2_sigma)
        p_1_sigma = np.array(p_1_sigma)
        p_2_sigma = np.array(p_2_sigma)

        # plt.plot(real, color='r')
        # plt.fill_between(x, p_3_sigma, m_3_sigma, color='#bbcfe0', label="$ \subset \mu \pm 3\sigma$")
        plt.fill_between(x, p_2_sigma, m_2_sigma, color=self.lighten_color(self.colors[self.model], 0.33),
                         label="$\subset \mu \pm 2\sigma$")
        plt.fill_between(x, p_1_sigma, m_1_sigma, color=self.lighten_color(self.colors[self.model], 0.66),
                         label="$ \subset \mu \pm \sigma$")
        ##----------------------------
        plt.plot(mape, c=self.colors[self.model], label="$\mu $ (MAPE)")

        ticks = x[::8]
        labels = []
        for i in ticks:
            if i == 0:
                labels.append("00:00")
            else:
                labels.append('%02i:00' % int(i / 4))
        plt.xticks(ticks=ticks, labels=labels, rotation=45)
        plt.ylim(0, 0.5)
        ticks = np.linspace(0, 0.5, 6)
        labels = []
        for i in ticks:
            label = "{:.1f}".format(i * 100)
            labels.append(label)
        plt.yticks(ticks=ticks, labels=labels)
        plt.xlabel("Tid på dygn [hh:mm]")
        plt.ylabel("MAPE [\%]")
        plt.xlim(0, 95)
        plt.legend(loc='upper left', fontsize=14)
        plt.title("MAPE och varians för fel")
        plt.tight_layout()
        plt.show()

    def sample_day(self):
        real = self.real[
            (self.real.index.year == 2019) & (self.real.index.month == 2) & (self.predicted.index.day == 12)]
        prediction_day = self.predicted.copy()
        prediction_day["Date"] = pd.to_datetime(self.predicted.index)
        #print(prediction_day)
        #print(prediction_day["Date"])
        prediction_day = self.predicted[
            (prediction_day["Date"].dt.year == 2019) & (prediction_day["Date"].dt.month == 2) & (
                        prediction_day["Date"].dt.day == 12) & (prediction_day["Date"].dt.hour == 0) & (
                        prediction_day["Date"].dt.minute == 0) & (prediction_day["Date"].dt.second == 0)]

        plt.plot(np.array(real.iloc[0]), c='black', label='Faktisk elförbrukning')
        plt.plot(np.array(prediction_day.iloc[0]), c=self.colors[self.model], label="Prognostiserad elförbrukning")
        #print(self.real.columns)
        cols = self.real.columns[::8]
        labels = []
        ticks = []
        for i in cols:
            if i == 0:
                labels.append("00:00")
            else:
                labels.append('%02i:00' % (int(int(i) / 4)))
            ticks.append(int(i))
        #print(ticks)
        #print(labels)
        plt.title("Exempelprognos för elförbrukning")
        plt.xticks(ticks=ticks, labels=labels, rotation=45)
        plt.xlim(-0.5, 95.5)
        plt.ylabel("Elförbrukning [kW]")
        plt.xlabel("Tid på dygn [hh:mm]")
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.show()

    def change_confidenceInterval(self):
        # intervall för en specifik prognos
        pred = []
        # m_3_sigma = []
        m_1_sigma = []
        m_2_sigma = []
        p_1_sigma = []
        p_2_sigma = []
        # p_3_sigma = []
        x = []

        for i in range(0, 95):
            x.append(i)
            # m_3_sigma.append(np.percentile(self.change_ape.iloc[:, i], 100-0.13))
            m_2_sigma.append(np.percentile(self.change_ape.iloc[:, i], 100 - (2.14 + 0.13)))
            m_1_sigma.append(np.percentile(self.change_ape.iloc[:, i], 100 - (13.59 + 2.14 + 0.13)))
            p_1_sigma.append(np.percentile(self.change_ape.iloc[:, i], 13.59 + 2.14 + 0.13))
            p_2_sigma.append(np.percentile(self.change_ape.iloc[:, i], 2.14 + 0.13))
            # p_3_sigma.append(np.percentile(self.change_ape.iloc[:, i], 0.13))

        mape = self.change_ape.mean()
        abs_errors = self.abs_errors.mean()
        abs_hold_errors = self.abs_hold_errors.mean()

        m_1_sigma = np.array(m_1_sigma)
        m_2_sigma = np.array(m_2_sigma)
        # m_3_sigma = np.array(m_3_sigma)
        p_1_sigma = np.array(p_1_sigma)
        p_2_sigma = np.array(p_2_sigma)
        # p_3_sigma = np.array(p_3_sigma)

        # plt.plot(real, color='r')
        # plt.fill_between(x, p_3_sigma, m_3_sigma, color='#bbcfe0', label="$ \subset \mu \pm 3\sigma$")
        plt.fill_between(x, p_2_sigma, m_2_sigma, color=self.lighten_color(self.colors[self.model], 0.33),
                         label="$\subset \mu \pm 2\sigma$")
        plt.fill_between(x, p_1_sigma, m_1_sigma, color=self.lighten_color(self.colors[self.model], 0.66),
                         label="$ \subset \mu \pm \sigma$")
        plt.plot(mape, c=self.colors[self.model], label="$\mu $ (MAPE)")
        # plt.plot(abs_errors, c="red", label="$\mu $ (ABSERROR)")
        # plt.plot(abs_hold_errors, c="green", label="$\mu $ (ABSHOLDERROR)")
        plt.plot()
        ticks = x[::8]
        labels = []
        for i in ticks:
            if i == 0:
                labels.append("0")
            else:
                labels.append(str(int(i / 4)))
        plt.xticks(ticks=ticks, labels=labels)
        plt.ylim(-5, 5)
        plt.xlim(0, 94)
        ticks = np.linspace(-5, 5, 11)
        labels = []
        for i in ticks:
            label = "{:.1f}".format(i)
            labels.append(label)
        plt.yticks(ticks=ticks, labels=labels)
        plt.xlabel("Tid [h]")
        plt.ylabel("Differans [kw]")
        # plt.legend()
        plt.title("Differans av absolut fel för prognos och hold")

        plt.show()

    def heatMap(self):
        errors = self.errors.copy()
        pe = self.APE.copy()
        errors['min_from_midnight'] = errors.index.hour * 60 + errors.index.minute
        pe['min_from_midnight'] = errors.index.hour * 60 + errors.index.minute

        plt.rcParams.update({'font.size': 24})
        time_errors = []

        newarray = np.zeros((96, 96 * 2))

        for i in range(0, 96):
            newdata = (pe.loc[(errors['min_from_midnight'] >= i * 15) & (pe['min_from_midnight'] <= i * 15 + 14)])
            newdata = newdata.drop(columns=['min_from_midnight'])

            mape = newdata.mean(axis=0)
            print(mape)
            for j in range(0, 96):
                newarray[i][j + i] = (mape[j])*100

        for i in range(7):
            x = [i * 16, 96 + i * 16]
            y = [0, 96]
            plt.plot(x, y, c='grey')

        # fel som både tid på dygn och tid framåt.
        plt.imshow(newarray, cmap=self.get_continuous_cmap(['#FFFFFF', '#F6D55C', '#000000']), aspect='auto')
        plt.ylabel("Tid för prognos [hh:mm]")
        plt.yticks([0, 12, 24, 36, 48, 60, 72, 84],
                   ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'])
        plt.xlabel("Tid prognos gäller för [hh:mm]")
        plt.xticks([0, 24, 48, 72, 96, 120, 144, 168],
                   ['00:00', '06:00', '12:00', '18:00', '24:00', '06:00',
                    '12:00', '18:00'], rotation = 45)
        plt.grid()
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('MAPE [\%]')

        ax2 = plt.twiny()
        ax2.set_xlabel("Prognoshorisont [h]")
        ax2.set_xticks([0, 16, 32, 48, 64, 80, 96, 192])
        ax2.set_xticklabels(['', '4', '8', '12', '16', '20', '24', ''])
        plt.tight_layout()
        plt.show()

    def lighten_color(self, color, amount=0.5):
        """
        Lightens the given color by multiplying (1-luminosity) by the given amount.
        Input can be matplotlib color string, hex string, or RGB tuple.

        Examples:
        >> lighten_color('g', 0.3)
        >> lighten_color('#F034A3', 0.6)
        >> lighten_color((.3,.55,.1), 0.5)
        """
        import matplotlib.colors as mc
        import colorsys
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

    def hex_to_rgb(self, value):
        '''
        Converts hex to rgb colours
        value: string of 6 characters representing a hex colour.
        Returns: list length 3 of RGB values'''
        value = value.strip("#")  # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def rgb_to_dec(self, value):
        '''
        Converts rgb to decimal colours (i.e. divides each value by 256)
        value: list (length 3) of RGB values
        Returns: list (length 3) of decimal values'''
        return [v / 256 for v in value]

    def get_continuous_cmap(self, hex_list, float_list=None):
        ''' creates and returns a color map that can be used in heat map figures.
            If float_list is not provided, colour map graduates linearly between each color in hex_list.
            If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

            Parameters
            ----------
            hex_list: list of hex code strings
            float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

            Returns
            ----------
            colour map'''
        rgb_list = [self.rgb_to_dec(self.hex_to_rgb(i)) for i in hex_list]
        if float_list:
            pass
        else:
            float_list = list(np.linspace(0, 1, len(rgb_list)))

        cdict = dict()
        for num, col in enumerate(['red', 'green', 'blue']):
            col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
            cdict[col] = col_list
        cmp = LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
        return cmp

    def epochs_mse(self, file_path):
        df = pd.read_csv(file_path)
        plt.figure(tight_layout=True)
        plt.title("Graf över MSE för körda epoker")
        plt.plot(df["val_mse"], '#3CAEA3', label='Validerings MSE')
        plt.plot(df["mse"], '#F6D55C', label='Träningsfel i MSE')
        plt.xlabel("Epoker")
        x_ticks = np.linspace(0, 30, 16)
        plt.xticks(ticks=x_ticks)
        plt.ylabel("Fel i MSE")
        plt.legend(fontsize=14)
        plt.yscale('log')
        plt.show()

    def batch_size_mse(self, file_path):
        df = pd.read_csv(file_path)
        plt.figure(tight_layout=True)
        plt.title("Graf över MSE för olika batchstorlekar")
        #plt.plot(df["val_mse"], '#3CAEA3', label='Validerings MSE')
        plt.scatter(df["batch_size"], df["mse"], s=100, c='#3CAEA3', label='Träningsfel i MSE')
        plt.xlabel("Stolek, batch")
        #x_ticks = np.linspace(0, 30, 16)
        plt.xscale("log", base=2)
        plt.xticks(ticks=[32, 64, 128, 256], labels=[32, 64, 128, 256])
        plt.ylabel("Valideringsfel i MSE")
        #plt.legend()
        plt.show()

    def persistance_first_hour(self):
        persistance = [np.mean(self.abs_hold_errors.iloc[:, i] / self.real.iloc[:, i]) * 100
                       for i in range(len(self.abs_hold_errors.iloc[0]))]
        print('Persistance model errors:' +
              '\n    15 min: %2.4f %% \n    30 min: %2.4f %% \n' % (persistance[1], persistance[2]) +
              '    45 min: %2.4f %% \n    60 min: %2.4f %%' % (persistance[3], persistance[4]))


if __name__ == "__main__":
    res = Results()
    res.sample_day()
    res.abs_confidenceInterval_days()


