import yfinance as yf
import pandas as pd
import time
import ta

from logger import LOGGER
logger = LOGGER

class Utils:

    @staticmethod
    def get_price_data(start="2015-01-01", end="2024-10-31"):
        '''
        This funtion is used to get historical data for different symbols
            - symbols: TSLA - Stock
                       BND - Bond
                       SPY - Index

        Parameter:
        ----------
            start(str): start time
            end(str): end time
        '''
        start_time = time.time()
        # Download tesla historical data
        tsla = yf.Ticker("TSLA")
        tsla_hist = tsla.history(period="1d", start=start, end=end)
        tsla_hist.drop(['Dividends', 'Stock Splits'], axis=1,inplace=True)

        # Download Vanguard Total Bond Market ETF historical data
        bnd = yf.Ticker("BND")
        bnd_hist = bnd.history(period="1d", start=start, end=end)
        bnd_hist.drop(['Dividends', 'Stock Splits'], axis=1,inplace=True)

        # Download S&P500 hisotical Data
        sp500 = yf.Ticker("SPY")
        sp500_hist = sp500.history(period="1d", start=start, end=end)  
        sp500_hist.drop(['Dividends', 'Stock Splits', 'Capital Gains'], axis=1,inplace=True)

        end_time = time.time()
        logger.info(f"Fetching data took {round(end_time - start_time, 2)}s")    

        return tsla_hist, bnd_hist, sp500_hist  
    
    @staticmethod
    def resample_timeframe( data: pd.DataFrame, tf: str)-> pd.DataFrame:
        '''
        This funtion is used to resample the data from dataframe 1 day to different timeframes
        '''
        return data.resample(tf).agg(
                {'Open' : 'first', 'High' : 'max', 'Low' : 'min', 'Close':'last', 'Volume': 'sum'}
            )
    
    @staticmethod
    def add_features( data):
        '''
        Funtion used to add features

        Parameter:
        ---------
            data(pd.DataFrmae)
        '''
        data['SMA_week'] = data['Close'].rolling(window=7).mean()  
        data['SMA_month'] = data['Close'].rolling(window=30).mean()  
        data['SMA_quarter'] = data['Close'].rolling(window=90).mean()  
        data['SMA_semi_yearly'] = data['Close'].rolling(window=180).mean()  
        data['SMA_yearly'] = data['Close'].rolling(window=360).mean()  

        data['std_week'] = data['Close'].rolling(window=7).std()  
        data['std_month'] = data['Close'].rolling(window=30).std()  
        data['std_quarter'] = data['Close'].rolling(window=90).std()  
        data['std_semi_yearly'] = data['Close'].rolling(window=180).std() 
        data['std_yearly'] =data['Close'].rolling(window=360).std()  

        return data       
    

    @staticmethod
    def volatility_indicators(data):
        '''
        Funcion to create new features for training model

        Parameter:
        ---------
            data(pd.DataFrame): data conatining open, high, low, close and volume

        Return:
        ------
            data(pd.DataFrame): dataFrame contining volatility indicators
        '''
        high = data['High']
        low = data['Low']
        close = data['Close']
        open = data['Open']

        data['ATR'] = ta.volatility.average_true_range(high, low, close, window=14, fillna=True)
        data['Bhband'] = ta.volatility.bollinger_hband(close, window=20, window_dev=2, fillna=True)
        data['Bhband_indicator'] = ta.volatility.bollinger_hband_indicator(close, window=20, window_dev=2, fillna=True)
        data['Blband'] = ta.volatility.bollinger_lband(close, window=20, window_dev=2, fillna=True)
        data['Blband_indicator'] = ta.volatility.bollinger_lband_indicator(close, window=20, window_dev=2, fillna=True)
        data['Bma'] = ta.volatility.bollinger_mavg(close, window=20, fillna=True)
        data['Dhband'] = ta.volatility.donchian_channel_hband(high, low, close, window=20, offset=0, fillna=True)
        data['Dlband'] = ta.volatility.donchian_channel_lband(high, low, close, window=20, offset=0, fillna=True)
        data['Mband'] = ta.volatility.donchian_channel_mband(high, low, close, window=10, offset=0, fillna=True)
        data['ketler'] = ta.volatility.keltner_channel_hband(high, low, close, window=20, window_atr=10, fillna=True, original_version=True)    

        return data    