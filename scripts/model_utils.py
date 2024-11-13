import sys
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import sidetable as stb
import mplfinance as mpf
from scipy.stats import zscore
import joblib
import pickle

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

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

from logger import LOGGER
logger = LOGGER


class ModelUtil:

    def __init__(self):
        self.logger = logger
    
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

        if model_name == 'SARIMAX' or 'LSTM':
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
        
        self.logger.info(f"Training {model_name} started ....")
        start_time = time.time()
        
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
        
        end_time = time.time()
        self.logger.info(f"Training {model_name} took {round(end_time - start_time, 2)} seconds")

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

        if model_name == 'ARIMA':
            errors = test['Close'].values - forecast.values
        else:
            errors = test['Close'] - forecast
        ax2.plot(test_dates, errors, color='red', label='Forecast Errors')
        ax2.axhline(y=0, color='black', linestyle='--')
        ax2.fill_between(test_dates, errors, np.zeros_like(errors),
                            alpha=0.5, color='red' if np.mean(errors) < 0 else 'green')

        ax2.set_title("Forecast errors (actual - forecast)")
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        plt.show()


    def model_metrics(self, model, model_name,data, test_size):
        
        train, test, train_dates, test_dates, series = self.train_test_split(data, test_size, model_name)

        forecast = model.predict(n_periods=test_size, X=test[['ATR', 'Bhband', 'Bhband_indicator', 'Bma', 'Dhband', 'Dlband', 'Mband', 'ketler']].values if model_name=='SARIMAX' else None)

        # # Get the confidence intervals as well
        forecast_conf, conf_int = model.predict(n_periods=test_size, X=test[['ATR', 'Bhband', 'Bhband_indicator', 'Bma', 'Dhband', 'Dlband', 'Mband', 'ketler']].values if model_name == 'SARIMAX' else None, return_conf_int=True)
        actual_values = test['Close'].values

        # Mean Absolute Error (MAE)
        mae = mean_absolute_error(actual_values, forecast)

        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mean_squared_error(actual_values, forecast))

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((actual_values - forecast) / actual_values)) * 100

        # Print the results
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}%")


    def create_sequence(self, data, time_steps =63):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps)])
            y.append(data[i + time_steps])
        return np.array(X), np.array(y)

    def train_lstm(self, data, seq_length, model_name, ticker, epochs=10, batch_size=32):

        start_time = time.time()
        logger.info(f"Training LSTM model for {ticker}...")

        train, test, train_dates, test_dates, series = self.train_test_split(data=data, test_size=seq_length, model_name=model_name)

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train['Close'].values.reshape(-1, 1))
        test_scaled = scaler.transform(test['Close'].values.reshape(-1, 1))

        joblib.dump(scaler, f'../Models/{ticker}-scaler.joblib')

        X_train, y_train = self.create_sequence(train_scaled, seq_length)
        X_val, y_val = self.create_sequence(test_scaled, seq_length)

        X_train = X_train.reshape(X_train.shape[0], seq_length, train_scaled.shape[1])
        X_val = X_val.reshape(X_val.shape[0], seq_length, train_scaled.shape[1])

        model = Sequential()
        model.add(Input(shape=(seq_length, train_scaled.shape[1])))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(Dropout(0.1))
        model.add(LSTM(50, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')
        model.summary()

        callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[callback],
            verbose=0
        )

        folder_path = '../Models/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-00")
        filename = f'{folder_path}{ticker}-{timestamp}.pkl'
        
        with open(filename, 'wb') as file:
            pickle.dump(model, file)

        print(f"Model saved as {filename}")

        end_time = time.time()
        logger.info(f"Model training took {round(end_time - start_time,2)} seconds")

        return model, history

    def lstm_predict_future(self, data, ticker, model_path, scaler_path, predict_days):

        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        scaler = joblib.load(scaler_path)

        time_step = 252 * 4  # Last 4 years 252 trading days 
        last_data = data[-time_step:]  
        last_data_scaled = scaler.transform(last_data['Close'].values.reshape(-1, 1))  
        
        input_seq = last_data_scaled.reshape(1, time_step, last_data_scaled.shape[1])
        predictions = []
        current_date = pd.to_datetime(data.index[-1]) + timedelta(days=1)  

        for _ in range(predict_days):
            predicted_price_scaled = model.predict(input_seq, verbose=0)
            
            predicted_price = scaler.inverse_transform(predicted_price_scaled[0][0].reshape(-1,1))
            
            predictions.append((current_date, predicted_price))
            
            input_seq = np.append(input_seq[:, 1:, :], [[predicted_price_scaled[0]]], axis=1)
            
            current_date += timedelta(days=1)

        prediction_df = pd.DataFrame(predictions, columns=['Date', f'{ticker}'])
        prediction_df.set_index('Date', inplace=True)
        
        return prediction_df


    def plot_forecast(self, ticker, historical_data, forecasted_data, confidence_interval):
        conf_int_pct = (1 - confidence_interval) * 100
        
        lower_bound = forecasted_data[ticker] * (1 - confidence_interval)
        upper_bound = forecasted_data[ticker] * (1 + confidence_interval)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data[ticker], mode='lines', name=f'{ticker} Historical Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecasted_data.index, y=forecasted_data[ticker], mode='lines', name=f'{ticker} Forecasted Price', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=forecasted_data.index, y=upper_bound, mode='lines', fill=None, line=dict(color='gray'), showlegend=False))
        fig.add_trace(go.Scatter(x=forecasted_data.index, y=lower_bound, fill='tonexty',  line=dict(color='gray'), name='Confidence Interval'))
        
        fig.update_layout(title=f"{ticker} Stock Price Forecast with {conf_int_pct:.2f}% Confidence Interval",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=50, r=50, t=50, b=50),
                        template='plotly_white')
        
        fig.show()

    def model_metrics_Lstm(self, model, model_name, data, test_size):
        train, test, train_dates, test_dates, series = self.train_test_split(data, test_size, model_name)

        forecast = model.predict(test[['Close','ATR', 'Bhband', 'Bhband_indicator', 'Bma', 'Dhband', 'Dlband', 'Mband', 'ketler']].values)
        actual_values = test['Close'].values
        
        # Mean Absolute Error (MAE)
        mae = mean_absolute_error(actual_values, forecast)

        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mean_squared_error(actual_values, forecast))

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((actual_values - forecast) / actual_values)) * 100

        # Print the results
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}%")   

    def plot_history(self, history):
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('LSTM Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()         

    def plot_actual_Vs_forecasted(self, data, test_size, model_name, model):
        plt.figure(figsize=(15, 8))
        train, test, train_dates, test_dates, series = self.train_test_split(data, test_size, model_name)
        plt.plot(test.index, test['Close'], label='Actual', linewidth=2)
        forecast = model.predict(test[['Close','ATR', 'Bhband', 'Bhband_indicator', 'Bma', 'Dhband', 'Dlband', 'Mband', 'ketler']].values)

        plt.plot(test.index, forecast, label=f' Prediction', linestyle='--')

        plt.title('Model Predictions Comparison')
        plt.xlabel('Date')
        plt.ylabel("Price")
        plt.legend()
        plt.show()        