import pandas as pd
from datetime import date, timedelta
from fbprophet import Prophet

def fb_prophet_model(df_t):

  polt=df_t.keys()[0]
  #Training Dataset Creation
  td = pd.DataFrame()
  td['ds'] = df_t.index
  td['y'] = df_t[polt].values

  #Model Preparation
  m = Prophet(daily_seasonality=True)
  m.add_country_holidays(country_name='IN')
  m.fit(td)

  #Predicting
  fb_pred = m.make_future_dataframe(periods=7)
  y_pred_fb = m.predict(fb_pred)

  #Preparing Prediction DF
  fb_pred['pred'] = y_pred_fb['yhat']
  fb_pred.set_index('ds', inplace=True)
  fb_pred.index.names = ['date']
  fb_predictions = fb_pred[-7:]

  return fb_predictions