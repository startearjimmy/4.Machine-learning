import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split



#Read data
raw_data = pd.read_csv('bezdekIris.data',names=['sepal_length','sepal_width','petal_length','petal_width','class'])
#preprocess class name to integers
#"Iris-setosa": 1
#"Iris Versicolour": 2
#"Iris-virginica": 3
cleanup_nums = {'class': {"Iris-virginica": 3,"Iris-versicolor": 2, "Iris-setosa": 1}}
raw_data.replace(cleanup_nums, inplace=True)
raw_data[['class']].astype('int32')
#shuffle data doing kfold
data = raw_data.sample(frac=1).reset_index(drop=True)
# k-fold
k = 10
tree_num = 5
fracc = 0
maxx = 0
kk = 0
for x in range(2,15):
    k=x
    ki=int(150/x)
    for j in range(10,50,1):
        y = j/100
        accuracy = 0

        for i in range(0,k):
            # spilt data into training and testing data
            # train: 120, test:30
            test_data = data.iloc[0:ki]
            train_data = data.iloc[ki:150].reset_index(drop=True)
            # k-fold
            #raindom forest
            #parameters
            tree_weight = list()
            trees = list()
            for t in range(0,tree_num):
                #randomly select data to create trees
                feature_train_data,feature_test_data,class_train_data,class_test_data = train_test_split(train_data[['sepal_length','sepal_width','petal_length','petal_width']],train_data[['class']],test_size=y)
                #create a tree
                sample_tree = tree.DecisionTreeClassifier()
                #print("num: ",t,"tree\n",feature_train_data.values)
                sample_tree.fit(feature_train_data.values,class_train_data.values)
                #count weight by score
                class_predict = sample_tree.predict(feature_test_data.values)
                score = 0
                for ii in range(0,len(class_predict)):
                    if class_predict[ii] == class_test_data.values[ii]:
                        score += 1
                weight = score/len(class_test_data)
                tree_weight.append(weight)
                trees.append(sample_tree)
            #testing
            #vote count votes
            #result records testing results
            vote = list()
            result = list()
            for n in range(0,ki):
                vote.append([0,0,0])
                result.append(0)
            for t in range(0,tree_num):
                predict_tmp = trees[t].predict(test_data[['sepal_length','sepal_width','petal_length','petal_width']].values)
                for ii in range(0,ki):
                    vote[ii][predict_tmp[ii]-1] = vote[ii][predict_tmp[ii]-1] +1
            #count vote
            for n in range(0,ki):
                if(vote[n][0]>vote[n][1] and vote[n][0]>vote[n][2]):
                    result[n] = 1
                elif(vote[n][1]>vote[n][0] and vote[n][1]>vote[n][2]):
                    result[n] = 2
                elif(vote[n][2]>vote[n][0] and vote[n][2]>vote[n][1]):
                    result[n] = 3
            #count accuracy
            score_tmp = 0
            for ii in range(0,ki):
                if result[ii] == test_data[['class']].values[ii]:
                    score_tmp += 1
            #print("==\n", score_tmp, "\n==")
            accuracy += score_tmp/ki
            #folding
            data_tmp = data.iloc[ki:150].reset_index(drop=True)
            data_tmp = data_tmp.append(test_data,ignore_index=True)
            data = data_tmp.reset_index(drop=True)
        print(accuracy/k, k, y)
        if(accuracy/k > maxx):
            maxx = accuracy/k
            fracc = y
            kk = k
#result
print(maxx,kk,y)


