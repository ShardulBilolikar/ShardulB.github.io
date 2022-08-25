
import pandas as pd
from datetime import date, timedelta
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import numpy as np
from pmdarima import auto_arima 


def arima_model(df_t, pf_list,X_test):

    polt=df_t.keys()[0]
    df_close = df_t[polt]

    # def test_stationarity(timeseries):
    #     rolmean = timeseries.rolling(12).mean()
    #     rolstd = timeseries.rolling(12).std()

        
    #     print("Results of dickey fuller test")
    #     adft = adfuller(timeseries,autolag='AIC')

    #     output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    #     for key,values in adft[4].items():
    #         output['critical value (%s)'%key] =  values
    #     print(output)
        
    # test_stationarity(df_close)

    # result = seasonal_decompose(df_close, model='multiplicative', freq = 30)

    df_log = np.log(df_close)
    moving_avg = df_log.rolling(12).mean()
    std_dev = df_log.rolling(12).std()

    train_data = df_t[polt]

    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                        test='adf',       # use adftest to find             optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=True,    # Yes Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)

    model = ARIMA(train_data, order=(3, 1, 2))  
    fitted = model.fit(disp=-1) 

    fc, se, conf = fitted.forecast(7, alpha=0.05)  # 95% confidence
    arima_predictions = pd.DataFrame(pf_list,columns=['date'])
    arima_predictions['pred'] = list(fc)
    arima_predictions.set_index('date', inplace=True)

    # lower_series = pd.Series(conf[:, 0], index=X_test.index)
    # upper_series = pd.Series(conf[:, 1], index=X_test.index)

    return arima_predictions