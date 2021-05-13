import pandas as pd
import numpy as np


""" linear_regression_input_comparison.py for all training and testing of linear regression model. """
__author__ = "Johan Lamm"
__maintainer__ = "Johan Lamm"
__email__ = "lammj@student.chalmers.se"

class DataPreprocessing():
    def mean_valid_obs(self,x):
        min_obs = 12
        valid_obs = x.notnull().sum()
        if valid_obs < min_obs:
            return np.nan
        return (x.sum()/valid_obs)


    def create_powerinput(self,datapaths):
        df = pd.DataFrame()
        for datapath in datapaths:
            df = pd.concat([df,pd.read_csv(datapath,parse_dates=['Date'],index_col='Date')])
        df = df.sort_index()
        df = df[['Consumption (kW)']]
        df = df.resample('15Min').apply(self.mean_valid_obs)
        df = df['2018-03-14':'2019-03-15']
        df = df.interpolate(limit=2)
        df = df.loc[df.groupby([df.index.year, df.index.month, df.index.day])['Consumption (kW)'].filter(
            lambda x: len(x[pd.isnull(x)]) < 1).index]
        return df

    def create_smhiinput(self,datapaths, feature_name):
        df1 = pd.DataFrame()
        for datapath in datapaths:
            df = pd.read_csv(datapath,sep=',')
            df["Date"] = df['Datum'] + ' ' + df['Tid (UTC)']
            df.index = pd.to_datetime(df["Date"])
            df = df.sort_index()
            df = df[df.index.year > 2017]
            df = df.resample("1min").bfill()
            df1 = pd.concat([df,df1],axis=1)

        df = df1[[feature_name]]
        df = df.resample('15min').first()
        return df

    def create_features(self,datapaths):
        if len(datapaths) == 2:
            df1 = pd.read_csv(datapaths[0],parse_dates=['Date'],index_col='Date')
            df2 = pd.read_csv(datapaths[1],parse_dates=['Date'],index_col='Date')
            df = df1.join(df2)
        else:
            df = pd.read_csv(datapaths[0],parse_dates=['Date'],index_col='Date')
        #historical and future quarter hours
        for i in range(-96, 97):
            if i > 0:
                df["p"+str(i*15)+'m'] = df['Consumption (kW)'].shift(i)
            if i < 0:
                df["f"+str(-i*15)+'m'] = df['Consumption (kW)'].shift(i)
        #onehot encoding for quaters
        for i in range(0, 96):
            df["Q" + str(i)] = (df.index.hour * 60 + df.index.minute) // 15 == i
        #Weekdays
        for i in range(0, 7):
            df["Day" + str(i)] = df.index.hour == i
        df = df.dropna()
        return df
