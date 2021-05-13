import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from tidypython import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from Results import results

""" logplot.py for plotting logs from optimization runs. """
__author__ = "Ludvig Eriksson and Ellinor Lundblad"
__maintainer__ = "Ludvig Eriksson"
__email__ = "luderik@student.chalmers.se"

# ===== Font fix for match with report =====
rc('font', **{'family': 'serif', 'sans-serif': ['computer modern sans-serif']})
plt.rcParams.update({'font.size': 30})
rc('text', usetex=True)  # Use local TeX install to render figure text, comment out if TeX is not installed

# ===== Path for log file with values to plot =====
log_path_csv = '../RNN/logs/RNN_log_2021427_17_6_51.csv'

# Parameters to pick from the dataframe and use in a 2d plot agaisnt MAPE or MSE as box plot
plot_2d = []

# Parameters to plot against each other and MAPE/MSE as heatmap
plot_3d_x = ['nodes']  # Parameters for x-axis
plot_3d_y = ['layers']  # Parameters for y-axis

metric = 'mse'  # Metric to plot against

log_df = pd.read_csv(log_path_csv)  # Load data to pandas.DataFrame
values = log_df[metric]  # Copy column with metric values to new pandas.DataFrame

if len(plot_3d_x) != len(plot_3d_y):  # Raise error if plot_3d_x and plot_3d_y have different lengths
    raise Exception('plot_3d_x and plot_3d_y must be of the same length!')

if plot_2d:  # If plot_2d contains value
    for e in plot_2d:
        plt.figure(e)

        data = log_df[[e, metric]]  # Make pandas.DataFrame with data to plot.
        unique_values = data[e].unique()  # Take all unique values from parameter column.

        # Sort data into matrix, with one column per unique parameter value
        box_data = [np.array(values[data[data[e] == v].index]).reshape((-1,)) for v in unique_values]

        plt.xlabel(e)  # Set x label to parameter name
        plt.ylabel(metric)  # Set y label to metric name
        plt.boxplot(box_data)  # Make boxplot, one box per column and unique metric value
        plt.xticks(range(1, len(unique_values)+1), unique_values)  # Add x ticks to plot

if plot_3d_x:  # If plot_3d_x contains value
    for a in log_df['activation'].unique():  # Make one heatmap for each unique activation function
        act_data = log_df.loc[log_df[log_df['activation'] == a].index]  # Select data for activation function 'a'

        for i in range(len(plot_3d_x)):
            xlabel = plot_3d_x[i]
            ylabel = plot_3d_y[i]
            data = act_data[[xlabel, ylabel, metric]]  # Copy x-data, y-data and metric data columns from DataFrame
            data.groupby(xlabel)  # Sort data according to x-data values
            print('data:\n', data)

            xData = act_data[xlabel]  # Copy x-data to separate vector
            yData = act_data[ylabel]  # Copy y-data to separate vector

            # np.flip(_) is used to get correct order for plotting, can be removed if desired
            uniqueX = xData.unique()  # Get unique values from x-data
            uniqueY = np.flip(yData.unique())  # Get unique values from y-data in reversed order

            # Make matrix for color-value assignment
            map = np.zeros((len(uniqueY), len(uniqueX)))

            # Fill color map matrix with correct values
            for j, x in enumerate(uniqueX):
                filtX = data.loc[data[data[xlabel] == x].index, :]  # Filter out all entries with x-value 'x'
                print('filtX:\n', filtX)
                for k, y in enumerate(uniqueY):
                    value = values.iloc[filtX[filtX[ylabel] == y].index]  # Filter out all entries with y-value 'y'
                    if len(value) > 1:
                        map[k, j] = np.mean(value)  # Fill matrix position with mean
                    else:
                        map[k, j] = value  # If only one value, fill matrix position with value

            fig, ax = plt.subplots()  # Create figure window
            fig.canvas.set_window_title(a+': '+xlabel+' vs '+ylabel)  # Set window title
            ax.set_title('Aktiveringsfunktion: %s' % a)  # Set plot title
            fig.tight_layout = True  # Use tight layout in figure
            im = ax.imshow(map, cmap=results.Results.get_continuous_cmap(results.Results(), ['#FFFFFF', '#3CAEA3', '#000000'], [0, 0.05, 1]), vmin=np.min(log_df['mse']), vmax=0.009)  # Heatmap plot using custom color scale
            ax.set_xlabel("Noder")  # Set x-label
            ax.set_ylabel("Lager")  # Set y-label
            cbar = ax.figure.colorbar(im, ax=ax)  # Create color bar for heatmap
            cbar.ax.set_ylabel("MSE")  # Set color bar y-label
            # Create x- and y-ticks ...
            ax.set_xticks(np.arange(len(uniqueX)))
            ax.set_yticks(np.arange(len(uniqueY)))
            # ... and label them with the respective list entries
            ax.set_xticklabels(uniqueX)
            ax.set_yticklabels(uniqueY)
plt.show()

