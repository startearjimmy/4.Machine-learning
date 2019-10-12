import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
# stock_name = '太子'
# df = pd.read_csv(stock_name+'_tree.csv', engine ='python')
df = pd.read_csv('tree.csv', engine ='python')
train = df[0:1500].reset_index(drop=True)
test = df[1500:].reset_index(drop=True)
cat_features = [0,1,2]
model=CatBoostRegressor(learning_rate=0.1, loss_function='RMSE')
# model.fit(train[['temperature','night_temperature','open_price']].values,train[['price_dif']].values,eval_set=(test[['temperature','night_temperature','open_price']].values,test[['price_dif']].values),plot=True)
# model.fit(train[['temperature','night_temperature','open_price']],train[['price_dif']])
# s = model.score(test[['temperature','night_temperature','open_price']],test[['price_dif']])
# y = model.predict(test[['temperature','night_temperature','open_price']])
# plt.xlim(-10, 5)
# plt.ylim(-10, 5)
# plt.scatter(y, test[['price_dif']].values)
# plt.show()
# print(s)
# 3.7596557227546348

# model.fit(train[['open_price']],train[['price_dif']])
# s = model.score(test[['open_price']],test[['price_dif']])
# y = model.predict(test[['open_price']])
# plt.scatter(y, test[['price_dif']].values)
# plt.plot([-20, -20], [20, 20], 'k-')
# plt.show()
# print(s)
# 3.7110662522985582

# model.fit(train[['temperature','night_temperature']],train[['price_dif']])
# s = model.score(test[['temperature','night_temperature']],test[['price_dif']])
# y = model.predict(test[['temperature','night_temperature',]])
# plt.scatter(y, test[['price_dif']].values)
# plt.plot([-20, -20], [20, 20], 'k-')
# plt.show()
# print(s)
# 3.656174466270046

model.fit(train[['temperature']],train[['price_dif']])
s = model.score(test[['temperature']],test[['price_dif']])
y = model.predict(test[['temperature']])
plt.scatter(y, test[['price_dif']].values)
plt.plot([-20, -20], [20, 20], 'k-')
plt.show()
print(s)
3.497712262220728