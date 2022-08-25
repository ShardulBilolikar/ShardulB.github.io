import pandas as pd
from datetime import date, timedelta
import xgboost as xgb

def xg_boost_model(X_train , y_train , X_test , pf_list):

  # Creating Model Object
  xg_reg = xgb.XGBRegressor(n_estimators=1000)
  # Fitting the Model
  xg_reg.fit(X_train, y_train,
          eval_set=[(X_train, y_train)],
          early_stopping_rounds=50,
        verbose=False) # Change verbose to True if you want to see it train
        
  #Predicting from Model
  y_pred_xg = xg_reg.predict(X_test)

  #Creating prediction result Dataframe
  xg_predictions = pd.DataFrame(pf_list,columns=['date'])
  xg_predictions['pred'] = list(y_pred_xg)
  xg_predictions.set_index('date', inplace=True)

  return xg_predictions