import pandas as pd 

name = '太子'

df = pd.read_csv( name +'.csv', engine ="python")

for i in range(0,df.shape[0]):
    slash = df['日期'][i].find('/')
    tmp = int(df['日期'][i][:slash])
    tmp += 1911
    tmp2 = str(tmp)
    df['日期'][i] = tmp2 + df['日期'][i][slash:]
df['成交股數'] = df['成交股數'].str.replace(',', '')
df['成交金額'] = df['成交金額'].str.replace(',', '')
df['成交筆數'] = df['成交筆數'].str.replace(',', '')
#print(df['成交股數'])
df.to_csv(name +'_new.csv',encoding='utf-8')