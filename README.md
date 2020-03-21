# Time Series - Exploratory Analysis and Forecasting
This assignment is about the different approaches to time series analysis and forecasting. We will look at exploring data with pandas and matplotlib, understand stationarity of our data and also demonstrate forecasting for both stationary and non stationary time series with Smoothing Methods and ARIMA. We will test out Multivariate time series aproaches too using Linear Regression, Random Forests and LSTMs.

---
## Preparing the data
The dataset is split in 9 files, spans 6 stores and contains the following fields with accordingly mentioned unique values - 

SKU Code        4289
Brand Code       187
Category          11
Bill Number    27191
Store Code         6
Sale Date        424
Sale/Return        2
MRP              540
Sales Price      734
Sales Qty         12

The time duration we have data for from each store differs but we find that the data falls in the 5/2017 to 7/2018 range. 
The dataset has no missing values, has a few anomalies due to Sales Price being greater than MRP and the Sales Qty of a purchase being zero. There are also a few duplicates in the dataset.  

The time data is in a string format and needs to be converted to datetime. For exploratory purposes we add a few more columns -  Year --> Which year
Month --> Which Month of the year
Week --> Which Week of the year
Sales --> Sales Price * Sales Qty
Discount --> (MRP - Sales Price) * Sales Qty

The discount values are later processed more since our exploration needs percentages. 

The 9 files are merged, preprocessed and then split into 6 files, one per store. This is pickled as one dictionary in the directory ```preproc_files```.

For each store, several new dataframes are created and pickled for ease in exploration. Key fields to be analysed are - 
Sales 
Discount
Sales Price
SKU Code

Each of these have 2 pickled files associated with them. Both of them are dictionaries. Each dictionary has keys as the store names and values as the dataframes associated with those stores.  

First one is the dataframes after applying ```df.groupby(['Year', 'Month'])['YOUR FIELD'].yourOp()```. The discount metric needs another step of arithmetic to convert to percentages.  
The second one is the dataframes after applying ```df.groupby(['Category', 'Brand Code', 'Year', 'Month'])['YOUR FIELD'].yourOp()```. 

The code for all the above steps is in the ```preprocessing.ipynb``` notebook. 

---
## Exploratory Analysis
This will mostly deal with analysed our pickled data to draw insights from it. 
The first pickled file for each feature will provide us with 1 plot per store that will plot the metric against time (month/year). 
The second pickled file is meant for a deep dive by the fields ```Category``` and ```Brand Code```. Here each plot will include subplots for as many months a store has been active. Each subplot will detail the concerned metric for different categories and brands. 

All the plots can be found in the ```plots/exploratorary``` directory.

---
## Forecasting
The Sales Time series is extracted and is analysed for stationarity using the Dickey Fuller Tests. I've used the 5% Critical Value as the deciding metric. Our time series is not stationary. It is made stationary by detrending the series using a rolling mean and then differencing it. 

The plots for Dickey Fuller tests are not available in the ```plots``` directory but can be viewed in the ```forecasting.ipynb``` notebook. The notebook will contain the code for all the following sections. 

### Stationary forecasting
Since our time series is non-stationary, stationarity has been coerced onto the series by taking a log and calculating a first order differenced series. There are other methods that haven't been applied in this project to coerce stationarity like seasonal decomposition. 

This series is used to get AutoCorrelation and Partial AutoCorrelation plots. 

#### AR, MA and ARIMA models
The following (article)[https://towardsdatascience.com/time-series-in-python-exponential-smoothing-and-arima-processes-2c67f2a52788] states these rules to chose our differencing coefficient - 
```
“ — Rule 1 : If the series has positive autocorrelations out to a high number of lags, then it probably needs a higher order of differencing.
— Rule 2 : If the lag-1 autocorrelation is zero or negative, or the autocorrelations are all small and patternless, then the series does not need a higher order of differencing. If the lag-1 autocorrelation is -0.5 or more negative, the series may be overdifferenced.”
(Robert Nau, Statistical Forecasting)

“ If the lag-1 autocorrelation of the differenced series ACF is negative, and/or there is a sharp cutoff, then choose a MA order of 1”.

“ If the lag-1 autocorrelation of the differenced series PACF is negative, and/or there is a sharp cutoff, then choose a AR order of 1”.
```
Our autocorrelation plots fall in Rule 2 - lag-1 autocorrelation is small and patternless but it is not negative or zero. Hence we use a differencing coefficient of 1. For AR and MA coefficients as well, due to our lag-1 correlation value, we take both to be 1. 

We still experiment with all three models - AR, MA and ARIMA and find that the RMSE is the least for AR regression. The plots for the experiemnts and forecasting can be found in ```plots/forecasting/ARIMA```. 

### Non-stationary forecasting
Smoothing methods arerecommended for non-stationary time-series. We compare three different smoothing methods
1. Simple Exponential Smoothing
2. Holts Linear Smoothing
3. Holts Exponential Smoothing (helps capture seasonal trends as well)

Several different hyperparamters for each of these value are tested, their RMSE values calulated and compared to find the best hyperparameters. The plots for hyperparamter tuning experiments as well as forecasting can be found in the ```plots/forecasting/smoothing``` directory. 

### Multivariate forecasting
The data is first turned into a multivariate dataset by inducing a time-lag in the series. We have used a lag-value of 6 which was decided randomly. 

#### Regression methods
Two regressors are compared
1. Random Forest
2. Linear Regression

I did not get the time to tune the hyperparamters for either but I did experiment with some of them manually. The plots can be found in the ```plots/forecasting/regression``` directory. 

#### LSTMs
Two LSTM based architectures are considered. 
1. Bidirectional LSTM based
2. Time Distributed Convolutional LSTM based

I did not get the time to tune the hyperparamters, time-lag value for either or experiment with depth and variety of architectures. The plots can be found in the ```plots/forecasting/BiLSTM``` and ```plots/forecasting/ConvLSTM``` directories. 


## Improvements
1. feature reduction and dropping irrelevant or less relevant data points depending on category, brand code, sales price, etc. 
2. feature selection using decision trees or statistical tests
3. using better regressors and ensemble methods like gradient boosting
4. testing out different ways of stationarising a series like seasonal decomposition. also trying other tests for stationarising like KPSS to make sure our algorithms are on the safer side. 
5. hyperparamter tuning for regressors for a large variety using optimization tools like hyperopt. 
6. for deep learning methods, archiecture selection, activation function selections, depth and no of nodes, all should be experimented with. 