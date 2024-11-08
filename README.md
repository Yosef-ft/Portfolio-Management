# Portfolio management

## Overview

The goal of this project is to create a robust portfolio optimization model that leverages future predicted data rather than solely relying on historical data. Traditional portfolio management often depends on analyzing past performance, but this approach aims to enhance predictive accuracy by integrating time series forecasting to anticipate market conditions. By focusing on future data predictions, we aim to optimize asset allocations, mitigate risk, and maximize returns proactively.

## Key Objectives
1. Predict Future Market Trends: Develop and implement predictive models (such as ARIMA, SARIMA, and LSTM) to forecast asset prices and market indices. This step will provide forward-looking insights to guide investment decisions.

2. Risk Management Through Predictive Analytics: Utilize predictive models to forecast volatility and other risk factors. This allows for preemptive adjustments to the portfolio to maintain an optimal balance between risk and return, enhancing the portfolio's resilience to market fluctuations.

3. Enhance Decision-Making with Quantitative Metrics: Calculate key performance indicators like Sharpe Ratio, Value at Risk (VaR), and expected returns based on forecasted data rather than historical averages. This approach improves decision-making by using forward-looking risk and return expectations.

4. Validate and Compare Predictions with Real Outcomes: Regularly evaluate the forecasted results against actual market data to refine and adjust the predictive models. This feedback loop enhances the model's accuracy and its ability to respond to changing market conditions, ensuring that the portfolio remains optimized.



## Getting Started
### Prerequisites
Make sure you have the following installed:
  * Python 3.x
  * Pip (Python package manager)

### Installation
Clone the repository:
```
git clone https://github.com/Yosef-ft/Portfolio-Management.git
cd Portfolio-Management
```
Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install the required packages:
```
pip install -r requirements.txt
```
