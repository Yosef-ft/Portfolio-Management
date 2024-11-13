import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import pypfopt
from pypfopt import risk_models, expected_returns, plotting
from pypfopt import objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return, ema_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import plotting
import cvxpy


class PortOptimizer:

    def test_risk_models(self, historical_data: pd.DataFrame):
        '''
        This funcion is used to see which risk model is appropriate for our data

        Parameters:
            historical_data (pandas.Datafrme): dataframe containing adjusted close for all tickers
        '''
        past_df, future_df = historical_data.iloc[:-252], historical_data.iloc[-252:]
        future_cov = risk_models.sample_cov(future_df)

        sample_cov = risk_models.sample_cov(past_df)
        future_variance = np.diag(future_cov)
        mean_abs_errors = []

        risk_methods = [
            "sample_cov",
            "semicovariance",
            "exp_cov",
            "ledoit_wolf",
            "ledoit_wolf_constant_variance",
            "ledoit_wolf_single_factor",
            "ledoit_wolf_constant_correlation",
            "oracle_approximating",
        ]

        for method in risk_methods:
            S = risk_models.risk_matrix(historical_data, method=method)
            variance = np.diag(S)
            mean_abs_errors.append(np.sum(np.abs(variance - future_variance)) / len(variance))
            
        xrange = range(len(mean_abs_errors))
        plt.barh(xrange, mean_abs_errors)
        plt.yticks(xrange, risk_methods)
        plt.show()



    def test_return_models(self, historical_data: pd.DataFrame):
        '''
        This funcion is used to see which return model is appropriate for our data

        Parameters:
            historical_data (pandas.Datafrme): dataframe containing adjusted close for all tickers
        '''       
        past_df, future_df = historical_data.iloc[:-252], historical_data.iloc[-252:]
        future_cov = risk_models.sample_cov(future_df) 
        future_rets = expected_returns.mean_historical_return(future_df)
        mean_abs_errors = []
        return_methods = [
            "mean_historical_return",
            "ema_historical_return",
            "capm_return",
            ]

        for method in return_methods:
            mu = expected_returns.return_model(past_df, method=method)
            mean_abs_errors.append(np.sum(np.abs(mu - future_rets)) / len(mu))
            
        xrange = range(len(mean_abs_errors))
        plt.barh(xrange, mean_abs_errors)
        plt.yticks(xrange, return_methods)
        plt.show()        

    def calculate_eReturn_covariance(self, adj_close: pd.DataFrame):
        '''
        This function calculates the covariance matrix and expected return for a given adjusted price.

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers
        
        Returns:
            covariance_matrix(pandas.DatFrame), Expected_return(pandas.Series)
        '''
        
        expected_return = ema_historical_return(adj_close, frequency=252, span=252)
            
        covariance_matrix = CovarianceShrinkage(adj_close).ledoit_wolf()

        return covariance_matrix, expected_return
    

    def calculate_EfficientFrontier(self, adj_close: pd.DataFrame):
        '''
        This fuction calculates the efficient frontier

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers

        Returns:
            efficient_frontier (pypfopt.efficient_frontier.efficient_frontier.EfficientFrontier)
        '''

        covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
        ef = EfficientFrontier(expected_return, covariance_matrix)

        return ef

        
    def clean_weights(self, adj_close: pd.DataFrame,max_return: bool = True, allow_shorts = False):
        '''
        This function calculates the weights and returns the clean weight for your portfolio

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers
            max_return(bool): If true it calculated the max sharpe but false it calculates the min volatility
        
        Returns:
            clean_weight(OrderedDict)
        '''

        if allow_shorts:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
            ef = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(-1,1)) 
            
        else:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
            ef = EfficientFrontier(expected_return, covariance_matrix)          

        if max_return:
            weights = ef.max_sharpe()
        else: 
            weights = ef.min_volatility()
        weights = pd.DataFrame(list(weights.items()), columns=['Symbols', 'Ratio'])
        clean_weight = weights.loc[weights['Ratio'] != 0]

        return clean_weight
    

    def plot_weights(self, adj_close: pd.DataFrame,max_return: bool= True ,allow_shorts=False):
        '''
        This function takes Efficient Frontier and plots the weights

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers
            allow_shorts(bool): this will plot if we allow short selling
            max_return(bool): If true it calculated the max sharpe but false it calculates the min volatility

        Returns:
            matplotlib plot
        '''

        if allow_shorts:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
            ef = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(-1,1)) 
            
        else:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
            ef = EfficientFrontier(expected_return, covariance_matrix)   
             
        if max_return:
            plotting.plot_weights(ef.max_sharpe())
        else:
            plotting.plot_weights(ef.min_volatility())


    def plot_efficient_frontier(self, adj_close: pd.DataFrame,max_return: bool= True, allow_shorts=False):
        '''
        This function takes Efficient Frontier and plots the Efficient frontier

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers
            allow_shorts(bool): this will plot if we allow short selling
            max_return(bool): If true it calculated the max sharpe but false it calculates the min volatility

        Returns:
            matplotlib plot
        ''' 

        if allow_shorts:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
            ef = EfficientFrontier(expected_return, covariance_matrix)   
            ef_plot = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(-1,1))

            if max_return:
                weights_plot = ef_plot.max_sharpe()
            else: 
                weights_plot = ef_plot.min_volatility()

            ef_plot.portfolio_performance(verbose=True)

            ef_constraints = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(-1,1))

            ef_constraints.add_constraint(lambda x: cvxpy.sum(x) == 1)

            fig, ax = plt.subplots()
            ax.scatter(ef_plot.portfolio_performance()[1], ef_plot.portfolio_performance()[0], marker='*', color='r', s=200,
                    label='Tangency portfolio')

            ax.legend()
            plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
            plt.show()

        else:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
            ef = EfficientFrontier(expected_return, covariance_matrix)   
            ef_plot = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(0,1))


            if max_return:
                weights_plot = ef_plot.max_sharpe()
            else: 
                weights_plot = ef_plot.min_volatility()

            ef_plot.portfolio_performance(verbose=True)

            ef_constraints = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(0,1))

            ef_constraints.add_constraint(lambda x: cvxpy.sum(x) == 1)

            fig, ax = plt.subplots()
            ax.scatter(ef_plot.portfolio_performance()[1], ef_plot.portfolio_performance()[0], marker='*', color='r', s=200,
                    label='Tangency portfolio')

            ax.legend()
            plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
            plt.show()        