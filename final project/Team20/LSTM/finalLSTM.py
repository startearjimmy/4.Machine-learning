# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 08:49:21 2018

@author: Asus
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#read data
readers=pd.read_csv('data/TSMC.csv',encoding='Big5', index_col=0 )

#change data type from list to matrix
def change_datatype(readers):
    row,column=readers.shape
    data=np.zeros((row,column))
    for i in range(row):
        for j in range(column):
            if type(readers.at[readers.index[i],readers.columns[j]])==str:
                string=readers.at[readers.index[i],readers.columns[j]]
                if j==0:
                    data[i][j]=np.int64((string.replace(',','')))
                elif j==1:
                    data[i][j]=np.int64((string.replace(',','')))
                elif j==6:
                    if string[0]=='-':
                        data[i][j]=float(string.replace('-',''))*-1
                    elif string[0]=='+':
                        data[i][j]=float(string.replace('+',''))
                    elif string[0]=='X':
                        data[i][j]=float(string.replace('X',''))
                    else:
                        data[i][j]=float(string)
                else:
                    data[i][j]=int(string.replace(',',''))
            else:
                data[i][j]=readers.at[readers.index[i],readers.columns[j]]
    return data

#normalize
def normalize(df):
    newdf= df.copy()
    meanvalue=np.zeros(8)
    SDvalue=np.zeros(8)
    for i in range(8):
        meanvalue[i]=np.mean(df[:,i])
        SDvalue[i]=np.std(df[:,i])

    for i in range(8):
        newdf[:,i]=(df[:,i]-meanvalue[i])/SDvalue[i]
    
    return newdf, meanvalue, SDvalue

#denormalize
def denormalize(df, meanvalue, SDvalue,columnnumber):
    denorm_value=(df*SDvalue[columnnumber])+meanvalue[columnnumber]
    return denorm_value

#seperate data into train part & test part
def data_helper(data, time_frame,column):
    

    datavalue = data

    result = []
    # 若想要觀察的 time_frame 為20天, 需要多加一天做為驗證答案
    for index in range( len(datavalue) - (time_frame+1) ): # 從 datavalue 的第0個跑到倒數第 time_frame+1 個
        result.append(datavalue[index: index + (time_frame+1) ]) # 逐筆取出 time_frame+1 個K棒數值做為一筆 instance
    
    result = np.array(result)
    #print(result.shape)
    number_train = round(0.9 * result.shape[0]) # 取 result 的前90% instance做為訓練資料
    
    # 訓練資料
    x_train = result[:int(number_train), :-1] # 訓練資料中, 只取每一個 time_frame 中除了最後一筆的所有資料做為feature
    y_train = result[:int(number_train), -1,column] # 訓練資料中, 取每一個 time_frame 中最後一筆資料的最後一個數值(收盤價)做為答案

    # 測試資料
    x_test = result[int(number_train):, :-1]
    y_test = result[int(number_train):, -1,column]

    return [x_train, y_train, x_test, y_test]

#build the LSTM learnimg model
def build_model(input_length, input_dim):
    d = 0.3
    model = Sequential()

    model.add(LSTM(256, input_shape=(input_length, input_dim), return_sequences=True))
    model.add(Dropout(d))

    model.add(LSTM(256, input_shape=(input_length, input_dim), return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(16,kernel_initializer="uniform",activation='relu'))
    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))

    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])

    return model

#data variable
#成交股數 成交金額 開盤價 最高價 最低價 收盤價 漲跌價差 成交筆數 >> 0-7
data_attribution=8
predict_data_attribution=6

# learning model variable
time_frame=80
epoch=30


data=change_datatype(readers)
data=data[2406:3206,:]
print(data.shape)
#print(data.shape)
data_norm, meanvalue, SDvalue=normalize(data)
#print(data_norm[:,predict_data_attribution])
#print(meanvalue)

x_train, y_train, x_test, y_test = data_helper(data_norm, time_frame,predict_data_attribution)
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#print(y_test)


# input 20天、8維 建立LSTM 模型
# 用訓練好的 LSTM 模型對測試資料集進行預測 
model = build_model( time_frame, data_attribution )
model.fit( x_train, y_train, batch_size=128, epochs=epoch, validation_split=0.1, verbose=1)
pred = model.predict(x_test)
pred2=pred[:,0]


# 將預測值與正確答案還原回原來的區間值
denorm_pred = denormalize(pred2, meanvalue, SDvalue,predict_data_attribution)
denorm_ytest = denormalize(y_test, meanvalue, SDvalue,predict_data_attribution)

#mean_squared_error of real & predict
MSE=mean_squared_error(denorm_pred, denorm_ytest)
SD=MSE**0.5
print(MSE,SD/meanvalue[predict_data_attribution])

#plot
plt.plot(denorm_pred,color='red', label='Prediction')
plt.plot(denorm_ytest,color='blue', label='Answer')
plt.legend(loc='best')
plt.savefig("TSMC_adagrad.png")
plt.show()

"""
#export txt
out=np.zeros((denorm_pred.shape[0],2))
out[:,0]=denorm_ytest
out[:,1]=denorm_pred
file=open('result/TSMC_成交筆數.txt','w')
for i in range(out.shape[0]):
    for j in range(out.shape[1]):
        file.write(str(out[i][j]))
        file.write('\t')
    file.write('\n')
file.close()
"""
#dead code
"""           
#normalize
def normalize(df):
    newdf= df.copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    
    for i in range(8):
        temp=df[:,i]
        temp=temp.reshape(-1,1)
        temp=min_max_scaler.fit_transform(temp)
        newdf[:,i]=temp[:,0]
    return newdf
#denormalize
def denormalize(df, norm_value):
    original_value = df[:,7]
    original_value = original_value.reshape(-1,1)
    norm_value=norm_value.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    temp = min_max_scaler.inverse_transform(norm_value)
    denorm_value=temp[:,0]
    
    return denorm_value
"""
    
