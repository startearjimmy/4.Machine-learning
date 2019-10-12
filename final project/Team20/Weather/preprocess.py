import pandas as pd
import numpy as np
df = pd.read_csv('taipei_weather.csv')
df['temperature'] = df['temperature'].str.replace('℃','')
df['temperature'] = df['temperature'].str.replace(' ','')
tmp = list()
col = pd.DataFrame()
for i in range (0,df.shape[0]):
    slash = df['temperature'][i].find('/')
    tmp.append(df['temperature'][i][slash+1:])
    col =  df['temperature'][i][:slash]
    df['temperature'][i] = col
df['night_temp'] = pd.Series(tmp)
df.to_csv('taipei_temperature_preprocess.csv')


