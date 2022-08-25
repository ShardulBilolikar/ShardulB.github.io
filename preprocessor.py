from datetime import date, timedelta, datetime
import csv
import pandas as pd
import numpy as np
import os


os.chdir("/content/Drive/MyDrive/AQAP/DATA/")


def pre_process(filename):
    # Reading File and renaming Columns
    df_t = pd.read_csv(f'/content/Drive/MyDrive/AQAP/DATA/{filename}')
    df_t = df_t.rename(columns={' pm10': 'pm10', ' pm25': 'pm25',
                                ' o3': 'o3', ' no2': 'no2', ' so2': 'so2', ' co': 'co'})

    # Converting Date column to Date Time Format
    df_t['date'] = pd.to_datetime(df_t['date'], format='%d-%m-%y')
    df_t.sort_values(by='date', inplace=True)
    start_date = df_t.date.min()
    end_date = date.today() - timedelta(days=1)
    r = pd.date_range(start=start_date, end=end_date)
    df_t = df_t.set_index('date').reindex(r).rename_axis('date').reset_index()

    raw_preprocessed = df_t.set_index('date')
    # EXPORT PREPROCESSED RAW
    reindex_name = filename[:-8]+'_rawprocessed.csv'
    raw_preprocessed.to_csv(reindex_name, index=True)
    # used for plotting historical data

    # Adding Columns for Date, Month and Year
    dd, mm, yy = [], [], []
    for i in range(len(df_t)):
        d = int(df_t["date"][i].day)
        dd.append(d)
        m = int(df_t["date"][i].month)
        mm.append(m)
        y = int(df_t["date"][i].year)
        yy.append(y)
    df_t["dd"] = dd
    df_t["mm"] = mm
    df_t["yy"] = yy
    df_t.set_index('date', inplace=True)

    # Removing 2020 Data
    df_t = df_t[df_t.yy != 2020]
    df_t = df_t.apply(pd.to_numeric,  errors='coerce')

    # Removing Date Keys from Key List
    date_keys = ['dd', 'mm', 'yy']
    pollutants = list(df_t.keys())
    for date_key in date_keys:
        pollutants.remove(date_key)

    # Fillin NaN Data with Monthly Medians
    for k in pollutants:
        df_t[k] = df_t.groupby(["mm", "yy"])[k].transform(
            lambda x: x.fillna(np.nanmedian(x)))


    for k in pollutants:
        df_t[k] = df_t.groupby(["yy"])[k].transform(
            lambda x: x.fillna(np.nanmedian(x)))

    #Dropping Keys with NaN left after PreProcessing
    drop_l=[]
    for p in pollutants:
        count_nan = len(df_t[p]) - df_t[p].count()
        if(count_nan>0):
            drop_l.append(p)
    for i in drop_l:
        df_t=df_t.drop(i, axis = 1)

    #EXPORT PREPROCESSED RAW
    pred_file = filename[:-8]+'_preprocessed.csv'
    df_t.to_csv(pred_file, index=True)
    #used for plotting historical data

    return df_t
