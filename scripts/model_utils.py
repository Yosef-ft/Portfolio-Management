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

    @staticmethod
    def train_test_split(data, test_size, model_name):
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