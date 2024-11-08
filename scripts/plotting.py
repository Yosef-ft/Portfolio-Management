import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ruptures as rpt
from pandas.plotting import lag_plot

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from utils import Utils


class Plots:

    def plot_seasonal_decompose(self, data_limiter: int, timeframe: int, data: pd.DataFrame):
        '''
        This function is used to plot seasonal decompose for price data

        Parameter:
        ---------
            data_limiter(int): a numerical value to limit the amount of data to be displayed
            timeframe(int): the time period for the plot
            data(pd.DataFrame)

        Return:
        ------
            matplotlip.pyplot object
        '''

        series = data['Close'][data_limiter:]
        result = seasonal_decompose(series, model='additive', period=timeframe)
        result.plot()


    def plot_data(self, data: pd.DataFrame, title: str, col: str, unusual_values = False):
        '''
        This funcion is used to plot line plot of the data

        Parameter:
        ----------
            data(pd.Dataframe)
            title(str): title of the symbol
            col(str): y axis of the plot

        Return:
        ------
            matplotlip.pyplot object            
        '''

        sns.set_theme("notebook")  

        plt.figure(figsize=(12, 6)) 
        sns.lineplot(data=data, x=data.index, y=col, color='b') 

        if unusual_values:
            unusual_returns = data[(data['Z-score Returns'] > 2.5) | (data['Z-score Returns'] < -2.5)]
            plt.scatter(unusual_returns.index, unusual_returns['Daily pct Change'], color='red', 
            label=f"Unusual Returns (±{2.5}std)", s=50, marker='o')
            plt.title(f'{title} Daily return Trend Over Time')

        else:     
            plt.title(f'{title} Price Trend Over Time')  
        plt.xlabel('')  
        plt.ylabel('Price') 

        plt.xticks(rotation=45)  
        plt.tight_layout()  

        plt.show()         


    def plot_changePoint(self, price, model:str, col: str):
        '''
        The model is used to plot changePoint

        Parameter:
        ---------
            price(pd.DataFrame)
            model(str): rbf, l1, l2
            col(str): y axis of the plot
        '''
        price_array = price['Close'].values
        model = model
        algo = rpt.Pelt(model=model).fit(price_array)
        change_points = algo.predict(pen=20)

        plt.figure(figsize=(14, 7))
        plt.plot(price.index, price[col], label='Price data')
        for cp in change_points[:-1]:
            plt.axvline(x=price.index[cp], color='red', linestyle='--', label='Change Point' if cp == change_points[0] else "")
        plt.xlabel('Date')
        plt.ylabel(f'{col} Price (USD)')
        plt.title(' Prices with Detected Change Points')
        plt.grid()
        plt.legend()
        plt.show()            


    def plot_distribution(self, price):
        '''
        Function to plot the distrution of price

        Parameter:
        ---------
            Price(pd.Dataframe): price data
        '''
        sns.set_style("whitegrid")  
        sns.set_palette("pastel")    

        plt.figure(figsize=(15, 5))

        sns.histplot(data=price, x='Close', kde=True, bins=30, color='skyblue', alpha=0.7)

        plt.title('Distribution of Price', fontsize=18, fontweight='bold')
        plt.xlabel('Price', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)


        plt.xlim(price['Close'].min(), price['Close'].max())  
        plt.ylim(0, 1800)  

        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()        


    def lag_plot(self, price_data: pd.DataFrame, period: int):
        '''
        Function to plot lag

        Paramter:
        --------
            price_data(pd.DataFrmae)
            period(int): lag period
        '''
        # Plot lag plot
        plt.figure(figsize=(8, 8))
        lag_plot(price_data['Close'], lag=period)
        plt.title(f'Lag Plot (Lag={period})')
        plt.xlabel('Price(t)')
        plt.ylabel('Price(t-1)')
        plt.grid()
        plt.show()  


    def plot_volatility(self, price_data: pd.DataFrame, sma: bool, symbol: str):
        '''
        Function to plot moving average and the std

        Parameter:
        ---------
            price(pd.DataFrame)
            sma(bool): if sma is true it plot the sma and if false it plots the STD 
            symbol(str): name of the stock you are plotting

        '''
        # Calculate SMA and STD
        price = price_data
        price = Utils.add_features(price)

        plt.figure(figsize=(14, 7))
        if sma:
            sns.lineplot(x=price.index, y='SMA_week', data=price, label='SMA Week')
            sns.lineplot(x=price.index, y='SMA_month', data=price, label='SMA Month')
            sns.lineplot(x=price.index, y='SMA_yearly', data=price, label='SMA Yearly')

            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.title(f'{symbol} Prices and SMAs Over Time')
            plt.legend(title='Simple Moving Averages')

        else:
            sns.lineplot(x=price.index, y='std_week', data=price, label='STD Week')
            sns.lineplot(x=price.index, y='std_month', data=price, label='STD Month')
            sns.lineplot(x=price.index, y='std_yearly', data=price, label='STD Yearly')

            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.title(f'{symbol} Prices and STD Over Time')
            plt.legend(title='Standard deviation Moving Averages')
                    
        plt.show()          
