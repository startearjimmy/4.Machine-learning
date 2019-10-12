import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
#preprocess

stock_name = '太子'
# stock = pd.read_csv(stock_name+'_new.csv', engine='python')
# T = pd.read_csv('taipei_temperature_preprocess.csv' )
# T['date'] = T['date'].str.replace('月','/')
# T['date'] = T['date'].str.replace('年','/')
# T['date'] = T['date'].str.replace('日','')
# print(stock.values[1302][1])
# print(T['date'][0] == stock.values[1303][1])
# a = 1303
# b = 0
# night_temperature = list()
# temperature = list()
# open_price = list()
# price_dif = list()
# while 1:
#     if a >= stock.shape[0] or b >= T.shape[0]:
#         break
#     if T['date'][b] == stock.values[a][1]:
#         print('a=',a,' b=',b, stock.values[a][1], '=', T['date'][b])
#         temperature.append(T['temperature'][b])
#         night_temperature.append(T['night_temp'][b])
#         open_price.append(stock.values[a][4])
#         price_dif.append(stock.values[a][8])
#         b+=1
#         a+=1
#     elif T['date'][b] == stock.values[a+1][1]:
#         a+=1
#     else:
#         print('a=',a,' b=',b, stock.values[a][1], '!=', T['date'][b])
#         b+=1
# res =list()
# for i in range(0,len(price_dif)):
#     res.append( (temperature[i],night_temperature[i],open_price[i],price_dif[i] ) )
# df = pd.DataFrame.from_records(res,columns = ['temperature','night_temperature','open_price','price_dif'])
# df.to_csv(stock_name+'_tree.csv',encoding='utf_8')

df = pd.read_csv(stock_name+'_tree.csv', engine ='python')
clf = tree.DecisionTreeRegressor()
train = df[0:1500].reset_index(drop=True)
test = df[1500:].reset_index(drop=True)
# train = np.nan_to_num(train)
# test = np.nan_to_num(test)

#decision tree

# clf.fit(train[['temperature','night_temperature','open_price']].values,train[['price_dif']].values)
# print(clf.score(test[['temperature','night_temperature','open_price']].values,test[['price_dif']].values))
# y = clf.predict(test[['temperature','night_temperature','open_price']].values)
# plt.scatter(y, test[['price_dif']].values)
# plt.plot([-20, -20], [20, 20], 'k-')
# plt.show()

#raindom forest

# regr = RandomForestRegressor()
# regr.fit(train[['temperature','night_temperature','open_price']].values,train[['price_dif']].values)
# print(regr.score(test[['temperature','night_temperature','open_price']].values,test[['price_dif']].values))
# y = regr.predict(test[['temperature','night_temperature','open_price']].values)
# plt.scatter(y, test[['price_dif']].values)
# plt.plot([-20, -20], [20, 20], 'k-')
# plt.show()

