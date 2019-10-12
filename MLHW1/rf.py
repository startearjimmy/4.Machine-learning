import pandas as pd
import math
from sklearn import tree
from sklearn.model_selection import train_test_split

data = pd.read_csv('ver3.csv')
row_nan = list()
# x =data['Rating'][7]
# print(math.isnan(data['Rating'][7]))
for x in range(0,len(data)):
    if(math.isnan(data['Rating'][x])):
        row_nan.append(x)
data = data.drop(data.index[row_nan])
data = data.reset_index()
# datatmp = data.groupby('Category')['Name'].nunique()
# ART_AND_DESIGN          41
# AUTO_AND_VEHICLES       46
# BEAUTY                  28
# BOOKS_AND_REFERENCE     72
# BUSINESS               123
# COMICS                  30
# COMMUNICATION           77
# DATING                  68
# EDUCATION               39
# ENTERTAINMENT           35
# EVENTS                  23
# FAMILY                 934
# FINANCE                124
# FOOD_AND_DRINK          35
# GAME                   494
# HEALTH_AND_FITNESS     104
# HOUSE_AND_HOME          19
# LIBRARIES_AND_DEMO      13
# LIFESTYLE              135
# MAPS_AND_NAVIGATION     46
# MEDICAL                144
# NEWS_AND_MAGAZINES      59
# PARENTING               28
# PERSONALIZATION        142
# PHOTOGRAPHY            112
# PRODUCTIVITY           128
# SHOPPING                59
# SOCIAL                  94
# SPORTS                 112
# TOOLS                  306
# TRAVEL_AND_LOCAL        59
# VIDEO_PLAYERS           60
# WEATHER                 18
# print(datatmp)

# k = 8
# trees = 2
# score = 0
max_acc = 0.0
kt = (0,0)
index = int(len(data)/10)
for k in range(2,15):
    for trees in range(2,50):
        score = 0
        for x in range(0,k):
            data = data.sample(frac=1).reset_index(drop=True)
            test = data[0:index]
            train = data[index:len(data)]
            for t in range(0,trees):
                trf,tef,trr,ter = train_test_split(train[['Rating','Price','Last Updated','Android Ver']],train[['Installs']], test_size = 0.1)
                sample_tree = tree.DecisionTreeClassifier()
                sample_tree.fit(trf,trr)
                x = sample_tree.score(test[['Rating','Price','Last Updated','Android Ver']],test[['Installs']])
                score += x
        print((k,trees),"==>",score/( k*trees ))
        if (score/( k*trees ) ) > max_acc:
            max_acc = (score/( k*trees ) )
            kt = (k,trees)
print(kt,max_acc)
