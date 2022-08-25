
import pandas as pd
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor

def random_forest_model(X_train , y_train , X_test , pf_list):
  
  #Creating Model Object
  rf_reg=RandomForestRegressor()
  #Fitting Model
  rf_model=rf_reg.fit(X_train,y_train)
  #Predicting from Model
  y_pred_rf=rf_model.predict(X_test)    

  #Creating prediction result Dataframe
  rf_predictions = pd.DataFrame(pf_list,columns=['date'])
  rf_predictions['pred'] = list(y_pred_rf)
  rf_predictions.set_index('date', inplace=True)
  
  return rf_predictions