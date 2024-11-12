import sys
import os

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import sidetable as stb
import mplfinance as mpf
from scipy.stats import zscore

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
from tensorflow.keras.metrics import Accuracy, Precision, F1Score, Recall
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import pmdarima as pm
import ta



class ModelUtil:

    
    def train_test_split(self, data, test_size, model_name):
        '''
        This funciont is used to split the data to training and testing based on number of dates

        Parameter:
        ----------
            data(pd.DataFrame)
            test_size(int): number of days to consider for testing

        Return:
        ------
            train_data, test_data
        '''

        if model_name == 'SARIMAX':
            series = data[['Close', 'ATR', 'Bhband', 'Bhband_indicator', 'Bma', 'Dhband', 'Dlband',
                'Mband', 'ketler']]
            
        else:
            series = pd.DataFrame(data['Close'])

        train = pd.DataFrame(series[:-test_size])
        test = pd.DataFrame(series[-test_size:])

        train_dates = data.index[:-test_size]
        test_dates = data.index[-test_size:]   

        return train, test, train_dates, test_dates, series
    
    
    def train_Arima_model(self,data, test_size=60, model_name = "ARIMA" ,exogenous=False):

        train, test, train_dates, test_dates, series = self.train_test_split(data, test_size, model_name)
        
        
        if model_name == 'ARIMA':
            model = pm.auto_arima(train['Close'], seasonal=False, trace=True, error_action='ignore',
                                    suppress_warnings=True, stepwise=True)
        else:
            model = pm.auto_arima(train['Close'].values, X= train[['ATR', 'Bhband', 'Bhband_indicator', 'Bma', 'Dhband', 'Dlband',
                        'Mband', 'ketler']].values if model_name =="SARIMAX" else None,
                                    start_P=0,
                                    start_q = 0,
                                    max_p=1, max_q=1, max_d=1, start_p=0, 
                                    seasonal=False if model_name=="ARIMA" else True, m=12, trace=True,
                                    error_action='ignore', suppress_warnings=True
                                )
        
        return model    
    

    def plot_model_forecast(self, model,data,model_name, test_size):
        '''
        This function is used to plot the model forecast with intervals

        Parameter:
        ---------
            model: the trained model
            data(pd.DataFrame): the data
            model_name: the name of the model - ARIMA, SARIMA, SARIMAX
            test_size: the size of days for testing


        Return:
        -------
            matplotlib.pyplot 
        '''
        train, test, train_dates, test_dates, series = self.train_test_split(data, test_size, model_name)
        forecast = model.predict(n_periods=test_size, X=test[['ATR', 'Bhband', 'Bhband_indicator', 'Bma', 'Dhband', 'Dlband', 'Mband', 'ketler']].values if model_name == 'SARIMAX' else None)

        # # Get the confidence intervals as well
        forecast_conf, conf_int = model.predict(n_periods=test_size, X=test[['ATR', 'Bhband', 'Bhband_indicator', 'Bma', 'Dhband', 'Dlband', 'Mband', 'ketler']].values if model_name == 'SARIMAX' else None, return_conf_int=True)

        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15,12))

        ax1.plot(train_dates[-90:], series[-test_size -90: -test_size]['Close'], label = 'Training data', color='blue')
        ax1.plot(test_dates, test['Close'],label='Actual', color = 'green')
        ax1.plot(test_dates, forecast, label='Forecast', color='red')

        ax1.fill_between(test_dates,
                            conf_int[:, 0],
                            conf_int[:, 1],
                            color='red',
                            alpha=0.1
                            )

        ax1.set_title('Price frocast with confidence interval')
        ax1.legend()
        ax1.grid(True)


        errors = test['Close'].values - forecast.values
        ax2.plot(test_dates, errors, color='red', label='Forecast Errors')
        ax2.axhline(y=0, color='black', linestyle='--')
        ax2.fill_between(test_dates, errors, np.zeros_like(errors),
                            alpha=0.5, color='red' if np.mean(errors) < 0 else 'green')

        ax2.set_title("Forecast errors actual - forecast")
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        plt.show()
