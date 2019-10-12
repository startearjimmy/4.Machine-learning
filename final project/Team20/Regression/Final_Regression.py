# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:05:19 2018

@author: 羅宇呈
"""

import numpy as np
from sklearn import datasets, linear_model, preprocessing
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression,HuberRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from pathlib import Path
from itertools import combinations
import random
from mpl_toolkits.mplot3d import axes3d, Axes3D
import os
from pathlib import Path




def  cal_cost(theta,X,y):

    m = len(y)
    predictions = X.dot(theta)
    cost = np.sum(np.square(predictions-y))/m
    return cost


def gradient_descent(X,y,theta,learning_rate,iterations):

    #print(y)
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,6))

    for idx in range(iterations):

        #print(X.T)# 1,X
        
        prediction = np.dot(X,theta)

        theta = theta -(1/m)*learning_rate*(X.T.dot((prediction - y)))

        #print(theta)

        cost_history[idx]  = cal_cost(theta,X,y)
        
    return theta, cost_history, theta_history


def Huber_Loss_gradient_descent(X,y,theta,learning_rate,iterations):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate 
    iterations = no of iterations
    
    Returns the final theta vector and array of cost history over no of iterations
    '''

    """
    def huber_loss(m, b, x, y, dy, c=2):
    y_fit = m * x + b
    t = abs((y - y_fit) / dy)
    flag = t > c
    return np.sum((~flag) * (0.5 * t ** 2) - (flag) * c * (0.5 * c - t), -1)
    """
    #print(y)

    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,3))
    bias=1
    

    for idx in range(iterations):

        #print(X.T)# 1,X
        
        prediction = np.dot(X,theta)

        theta = theta -(1/m)*learning_rate*(X.T.dot((prediction - y)))

        #print(theta)

        cost_history[idx]  = cal_cost(theta,X,y)
        
    return theta, cost_history, theta_history

"""
def adagrad(model, X_train, y_train, minibatch_size):
    cache = {k: np.zeros_like(v) for k, v in model.items()}

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for k in grad:
            cache[k] += grad[k]**2
            model[k] += alpha * grad[k] / (np.sqrt(cache[k]) + eps)

    return model
"""


def Stochastic_gradient_descent(X,y,theta,learning_rate,iterations):
                                    
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate 
    iterations = no of iterations
    
    Returns the final theta vector and array of cost history over no of iterations
    '''

    #print(y)

    m = len(y)

    velocity = np.zeros((3,1))
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,3))
    gamma=0.85

    for idx in range(iterations):

        #print(X.T)# 1,X
        
        update_idx=random.randint(0,m-1)

        prediction=np.c_[X[update_idx]]

        prediction = np.multiply(prediction,theta) 

        #print(prediction.shape)
        
        grad=(1/m)*learning_rate*(X[update_idx].T.dot((prediction - y[update_idx])))

        # * SGD + Momemtum 

        velocity=np.multiply(gamma,velocity)+grad

        theta = theta - velocity

        #print(theta.shape)

        #print(theta)
            
        #gamma = gamma * 0.99
        
        cost_history[idx]  = cal_cost(theta,X,y)
        
    return theta, cost_history, theta_history



# * Read dataset by Pandas
# * Then transfer it into numpy array


mypath = "F:\EECS_CS\Dataset\\"


Result_path="F:\EECS_CS\ML_RESULT\\"

dataset = os.listdir(mypath)

for file in dataset:
   
    data=pd.read_csv(mypath+file)

    
    NS = np.array(data[u'成交股數'],dtype=str)
    TO = np.array(data[u'成交金額'],dtype=str)
    OP=np.array(data[u'開盤價'],dtype=str)
    DH=np.array(data[u'最高價'],dtype=str)
    DL=np.array(data[u'最低價'],dtype=str)
    CP=np.array(data[u'收盤價'],dtype=str)
    DP=np.array(data[u'漲跌價差'],dtype=str)
    NT=np.array(data[u'成交筆數'],dtype=str)
    
    dictionary={'NS':NS,'TO':TO,'OP':OP,'DH':DH,'DL':DL,'CP':CP,'DP':DP,'NT':NT}
    
    for attr in dictionary:
        if attr == 'DP':
            for i in range(0,len(dictionary[attr])):
                if dictionary[attr][i][0] == '+':
                    dictionary[attr][i]=dictionary[attr][i].replace("+","")
                    dictionary[attr][i]=float(dictionary[attr][i])
                elif dictionary[attr][i][0] == '-':
                    dictionary[attr][i]=dictionary[attr][i].replace("-","")
                    if dictionary[attr][i]=="":
                        dictionary[attr][i]=0
                    else:
                        dictionary[attr][i]=float(dictionary[attr][i])*-1
                elif dictionary[attr][i][0] == 'X':
                    dictionary[attr][i]=dictionary[attr][i].replace("X","")
                    dictionary[attr][i]=float(dictionary[attr][i])
                else:
                    dictionary[attr][i]=float(dictionary[attr][i])
        else:
            for i in range(0,len(dictionary[attr])):
                tmp=dictionary[attr][i].replace("-","")
                if tmp=="":
                    dictionary[attr][i]=0
                else:
                    dictionary[attr][i]=dictionary[attr][i].replace(",","")
                    dictionary[attr][i]=float(dictionary[attr][i])
               #print(dictionary[attr][i])

    
    
    dictionary['NS'] = dictionary['NS'].astype(float)
    dictionary['TO'] = dictionary['TO'].astype(float)
    dictionary['OP']=dictionary['OP'].astype(float)
    dictionary['DH']=dictionary['DH'].astype(float)
    dictionary['DL']=dictionary['DL'].astype(float)
    dictionary['CP']=dictionary['CP'].astype(float)
    dictionary['DP']=dictionary['DP'].astype(float)
    dictionary['NT']=dictionary['NT'].astype(float)

    
    
    #print(NS.shape())
    
    # * Transfer data shape to -1,1
    # * By applying concatnate on each attribute
    
    dictionary['NS']=np.c_[dictionary['NS']]
    dictionary['TO']=np.c_[dictionary['TO']]
    dictionary['OP']=np.c_[dictionary['OP']]
    dictionary['DH']=np.c_[dictionary['DH']]
    dictionary['DL']=np.c_[dictionary['DL']]
    dictionary['CP']=np.c_[dictionary['CP']]
    dictionary['DP']=np.c_[dictionary['DP']]
    dictionary['NT']=np.c_[dictionary['NT']]
    
    (rol,col)=dictionary['NS'].shape
    
    
    
    AV=np.zeros((rol,1))
    
    for index in range(0,rol):   
        if dictionary['NT'][index]!=0:      
            AV[index]=dictionary['NS'][index]/dictionary['NT'][index]
        else:
            AV[index]=0
    
    
        
    # * preprocessing
    # * standardlization    
   

    AV=preprocessing.scale(AV)

    
    dictionary['NS']=preprocessing.scale(dictionary['NS'],axis=0)

    
    dictionary['TO']=preprocessing.scale(dictionary['TO'],axis=0)



    
    dictionary['OP']=preprocessing.scale(dictionary['OP'],axis=0)
    dictionary['DH']=preprocessing.scale(dictionary['DH'],axis=0)
    dictionary['DL']=preprocessing.scale(dictionary['DL'],axis=0)
    dictionary['CP']=preprocessing.scale(dictionary['CP'],axis=0)
    dictionary['NT']=preprocessing.scale(dictionary['NT'],axis=0)
    dictionary['DP']=preprocessing.scale(dictionary['DP'],axis=0)   
    

    #print(AV)
    
    
    # * Part 2-1
    # * Linear regression with single variable by built-in function
    """        
    Result_path="D:\EECS\ML_RESULT\\Bulit_in_Linear_Regression\\"+file+"\\"
    
    if not os.path.exists(Result_path):
        os.makedirs(Result_path)
    
    data = pd.read_csv(mypath+file)
    (row,column)=data.shape
    
    
    f = open(str(Result_path)+'Data.txt', 'w')
    
    max_r2=0
    min_mse=99999
    max_r2_attr=''
    min_mse_attr=''
    max_coeff=0
    max_coeff_attr=''
    
    for attribute,value in dictionary.items():
        
        
        #print(attribute,value)
        
        f.write("X:"+attribute+"\n"+"Y:Average_Volumn"+'\n')
    
        LR=LinearRegression()
        
        LR.fit(np.reshape(value, (len(value), 1)), np.reshape(AV, (len(AV), 1)))
    
        f.write('Weight:'+str(LR.coef_[0][0])+'\n')
    
        f.write('Bias:'+str(LR.intercept_[0])+'\n')
    
        if(max_coeff<LR.coef_[0][0]): 
           max_coeff=LR.coef_[0][0]
           max_coeff_attr=attribute
           
        print("Coefficient:",LR.coef_[0][0])
        print("Intersection:",LR.intercept_[0] )
        
        predicted_y=LR.predict(np.reshape(value, (len(value), 1)))
    
        if(max_r2<r2_score(AV, predicted_y)): 
            max_r2=r2_score(AV, predicted_y)
            max_r2_attr=attribute
    
        if(min_mse>mean_squared_error(AV, predicted_y)): 
            min_mse=mean_squared_error(AV, predicted_y)
            min_mse_attr=attribute
    
        f.write('r2_score:'+str(r2_score(AV, predicted_y))+'\n')
        #f.write('mean_squared_error:'+str(mean_squared_error(Concrete_compressive_strength, predicted_y))+"\n\n")   
        #print("r2_score:",r2_score(Concrete_compressive_strength, predicted_y))
        #print("mean_squared_error:",mean_squared_error(Concrete_compressive_strength, predicted_y))
        #print("\n")
    
        plt.scatter(value, AV, color='black',s=10)
        plt.plot(value, predicted_y, color='blue', linewidth=3)
        plt.xticks()
        plt.yticks()
        plt.xlabel(attribute)
        plt.ylabel('Average_Volumn')
        plt.savefig(str(Result_path)+attribute+' '+'Average_Volumn'+'.png')
        plt.clf()
    f.write("Max coefficient: "+max_coeff_attr+"  "+str(max_coeff)+'\n')
    f.write("Max r2_score: "+max_r2_attr+"  "+str(max_r2)+'\n')
    f.write("Min mean_squared_error: "+min_mse_attr+"  "+str(min_mse)+'\n')
    f.close()
    """
    
    
    # * Part 2-2
    # * Linear regression with single variable by your own gradient descent
    
    """
    Result_path="D:\EECS\ML_RESULT\\Batch_Gradient_Descent\\"+file+"\\"
    
    if not os.path.exists(Result_path):
        os.makedirs(Result_path)    
    

    
    
    f = open(str(Result_path)+'hw_3-2.txt', 'w')
    
    for attribute,value in dictionary.items():
        
        f.write(attribute+" vs "+"Average_Volumn\n")
        theta = np.random.randn(2,1)
        learning_rate = 0.0001         
        precision = 0.000001        #this tells us when to stop the algorithm
        previous_0_step_size = 1    #
        previous_1_step_size = 1
        max_iters = 100000           # maximum number of iterations
        iters = 0    
        x_b = np.c_[np.ones((len(value),1)),value]
        theta,cost_history,theta_history = gradient_descent(x_b,AV,theta,learning_rate,max_iters)
        print('final cost/mse:  {:0.3f}'.format(cost_history[-1]))
        predict_y=np.dot(x_b,theta)
        print(cost_history)
        f.write("weight:"+str(theta[1][0])+" bias:"+str(theta[0][0])+"\n")
        f.write("R2_score: "+str(r2_score(AV,predict_y))+"\n")
        plt.scatter(value, AV, color='black',s=10)
        plt.plot(value, predict_y, color='red',linewidth=0.5,marker='o')
        plt.xticks()
        plt.yticks()
        plt.xlabel(attribute)
        plt.ylabel('Average_Volumn')
        plt.savefig(Result_path+attribute+' '+'Average_Volumn'+'.png')
        plt.clf()
        x_axis = np.linspace(0, max_iters, max_iters, endpoint=False)
        plt.scatter(x_axis,cost_history, color='blue',s=0.1)
        plt.xticks()
        plt.yticks()
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.savefig(Result_path+attribute+' '+'Average_Volumn'+'_cost'+'.png')
        plt.clf()
    f.close()
    """
    
    # * Part 3-3
    # * Linear regression with multi-variable by your own gradient descent
    """
    f = open(str(mypath)+'\\3-3\\'+'HW_3-3.txt', 'w')
    
    for combinations in list(combinations(dict, 2)):
        
        x1,x2=combinations
    
        X=np.concatenate((dict[x1], dict[x2]), axis=1)
       
        X_train,X_test,y_train,y_test=train_test_split(X,Concrete_compressive_strength ,test_size=0.2)
    
        print("now testing: \n",x1," and ",x2," vs ","concrete_compressive_strength\n")
    
        f.write("now testing: \n"+x1+" and "+x2+" vs "+"concrete_compressive_strength\n")
    
        theta = np.random.randn(3,1)
    
        s_theta=theta
    
        learning_rate = 0.0001         
        precision = 0.000001        
        max_iters = 1000000          
        iters = 0     
    
        x_b = np.c_[np.ones((len(X_train),1)),X_train]  
        
        # * Gradient Descent
    
        theta,cost_history,theta_history = gradient_descent(x_b,y_train,theta,learning_rate,max_iters)
    
        #print('final cost/mse:  {:0.3f}'.format(cost_history[-1]))
    
        predict_y=np.dot(x_b,theta)
    
        # * Write training data
    
        f.write("training dataset:"+"\n")
    
        f.write("Batch Gradient descent:"+"\n")
    
        f.write("theta_1:"+str(theta[2][0])+" theta_2:"+str(theta[1][0])+"theta_3:"+str(theta[0][0])+"\n")
    
        f.write("MSE Score: "+str(cost_history[-1])+"\n")
    
        f.write("R2_score: "+str(r2_score(y_train,predict_y))+"\n\n")
    
        # * Stochastic_gradient_descent
        
        f.write("Stochatstic Gradient descent"+"\n")
    
        s_theta,s_cost_history,s_theta_history = Stochastic_gradient_descent(x_b,y_train,s_theta,learning_rate*3,max_iters)
    
        s_predict_y=np.dot(x_b,s_theta)
    
        f.write("theta_1:"+str(s_theta[2][0])+" theta_2:"+str(s_theta[1][0])+"theta_3:"+str(s_theta[0][0])+"\n")
    
        f.write("training dataset:"+"\n")
    
        f.write("MSE Score: "+str(s_cost_history[-1])+"\n")
    
        f.write("R2_score: "+str(r2_score(y_train,s_predict_y))+"\n\n")
    
        # * Plot training data's 3-D pics
        # * Batch Gradient Descent
    
        ones,attr1,attr2=np.hsplit(x_b, 3)
    
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    
        ax.scatter(attr1,attr2,y_train,s=10,c='blue')
       
        ax.scatter(attr1,attr2,predict_y,c='red')
    
        ax.set_xlabel(x1)
        ax.set_ylabel(x2)
        ax.set_zlabel('Concrete_compressive_strength')
    
        plt.savefig(str(mypath)+'\\3-3\\'+'\\BGD_Scatter_Training\\'+x1+" "+x2+' '+'concrete_compressive_strength'+'.png')
        plt.clf()
        plt.close()
    
        # * Stochastic_gradient_descent
    
        ones,attr1,attr2=np.hsplit(x_b, 3)
    
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    
        ax.scatter(attr1,attr2,y_train,s=10,c='blue')
       
        ax.scatter(attr1,attr2,s_predict_y,c='red')
    
        ax.set_xlabel(x1)
        ax.set_ylabel(x2)
        ax.set_zlabel('Concrete_compressive_strength')
    
        plt.savefig(str(mypath)+'\\3-3\\'+'\\SGD_Scatter_Training\\'+x1+" "+x2+' '+'concrete_compressive_strength'+'.png')
        plt.clf()
        plt.close()  
    
        # * Plot MSE loss graph
        # * Batch Gradient Descent
    
        x_axis = np.linspace(0, max_iters, max_iters, endpoint=False)
        
    
        fig = plt.figure()
    
        plt.scatter(x_axis,cost_history, color='blue',s=0.1)
        plt.xticks()
        plt.yticks()
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.savefig(str(mypath)+'\\3-3\\'+'\\BGD_MSE\\'+x1+" "+x2+' '+'concrete_compressive_strength'+'_MSE'+'.png')
        plt.clf()
        plt.close()
    
        # * Stochastic_gradient_descent
    
        s_x_axis = np.linspace(0, max_iters, max_iters, endpoint=False)
    
        fig = plt.figure()
        plt.scatter(s_x_axis,s_cost_history, color='blue',s=0.1)
        plt.xticks()
        plt.yticks()
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.savefig(str(mypath)+'\\3-3\\'+'\\SGD_MSE\\'+x1+" "+x2+' '+'concrete_compressive_strength'+'_Stochatstic Gradient descent_MSE'+'.png')
        plt.clf()
        plt.close()
    
        # * Now test with testing dataset
        # * Record r2_score and MSE value both
    
        x_test=np.c_[np.ones((len(X_test),1)),X_test]
    
        test_predicted_y = np.dot(x_test,theta)
    
        f.write("testing dataset:"+"\n")
    
        f.write("Batch Gradient descent:"+"\n")
    
        f.write("MSE Score: "+str(mean_squared_error(y_test,test_predicted_y))+"\n")
    
        f.write("R2_score: "+str(r2_score(y_test,test_predicted_y))+"\n\n")
    
    
        # * Stochastic_gradient_descent
        
        f.write("Stochatstic Gradient descent"+"\n")
    
        #print(s_theta)
    
        s_test_predicted_y = np.dot(x_test,s_theta)
    
        f.write("MSE Score: "+str(mean_squared_error(y_test,s_test_predicted_y))+"\n")
    
        f.write("R2_score: "+str(r2_score(y_test,s_test_predicted_y))+"\n\n")
    
        # * Plot testing data's 3-D pics
        # * Batch Gradient Descent
    
        ones,attr1,attr2=np.hsplit(x_test, 3)
    
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    
        ax.scatter(attr1,attr2,y_test,s=10,c='blue')
       
        ax.scatter(attr1,attr2,test_predicted_y,c='red')
    
        ax.set_xlabel(x1)
        ax.set_ylabel(x2)
        ax.set_zlabel('Concrete_compressive_strength')
    
        plt.savefig(str(mypath)+'\\3-3\\'+'\\BGD_Scatter_Testing\\'+x1+" "+x2+' '+'concrete_compressive_strength'+' testing '+'.png')
        plt.clf()
        plt.close()
    
        # * Stochastic_gradient_descent
    
        ones,attr1,attr2=np.hsplit(x_test, 3)
    
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    
        ax.scatter(attr1,attr2,y_test,s=10,c='blue')
       
        ax.scatter(attr1,attr2,s_test_predicted_y,c='red')
    
        ax.set_xlabel(x1)
        ax.set_ylabel(x2)
        ax.set_zlabel('Concrete_compressive_strength')
    
        plt.savefig(str(mypath)+'\\3-3\\'+'\\SGD_Scatter_Testing\\'+x1+" "+x2+' '+'concrete_compressive_strength'+' testing '+'.png')
        plt.clf()
        plt.close()
        
    
        f.write("----------------------------------------------------------------------------------------\n")
    
    f.close()
    
    """
    
    # * Part 3-4
    # * Polynomial regression by your own gradient descent
    
    """
    for j in range(0,100):
        max_r2=-1
        f = open(str(mypath)+'\\3-5\\'+str(j)+'_HW_3-5.txt', 'w')
        count=0
    
        for i in range(1,9):
            for combination in list(combinations(dict, i)):
                count=count+1
                a=0
    
                for element in combination:
                    if a==0:
                        X=dict[combination[0]]
                        a=1
                    else:
                        X=np.concatenate((X,dict[element]), axis=1)
            
            
            
            
            #print(X.shape)
                poly = PolynomialFeatures(2)
                X=poly.fit_transform(X)
            #print(X.shape)
                X_train,X_test,y_train,y_test=train_test_split(X,Concrete_compressive_strength ,test_size=0.2)
    
                (row,col)=X.shape
                f.write("Now Testing:\n")
                comstr=""
                for element in range(len(combination)):
                    comstr=comstr+combination[element]
                    comstr=comstr+" "
                f.write(comstr+"\n")
        
                f.write("Training dataset:\n")
    #huber = HuberRegressor().fit(X, Concrete_compressive_strength)    
                theta = np.random.randn(col,1)    
                learning_rate = 0.05         
                precision = 0.000001        
                max_iters = 10000          
                iters = 0     
        
                theta,cost_history,theta_history = gradient_descent(X_train,y_train,theta,learning_rate,max_iters)
                predict_y=np.dot(X_train,theta)
    #f.write("theta_5:"+str(theta[5][0])+" theta_4:"+str(theta[4][0])+"theta_3:"+str(theta[3][0])+" theta_2:"+str(theta[2][0])+"theta_1:"+str(theta[1][0])+" theta_0:"+str(theta[0][0])+"\n")
                f.write("MSE Score: "+str(mean_squared_error(y_train,predict_y))+"\n")
                f.write("R2_score: "+str(r2_score(y_train,predict_y))+"\n\n")
                f.write("Testing dataset:\n")
                test_predict_y=np.dot(X_test,theta)
                f.write("MSE Score: "+str(mean_squared_error(y_test,test_predict_y))+"\n")
                print("r2_score:",r2_score(y_train,predict_y))
                if(max_r2<r2_score(y_train,predict_y)):
                    max_r2=r2_score(y_train,predict_y)
                f.write("R2_score: "+str(r2_score(y_test,test_predict_y))+"\n\n")
                x_axis = np.linspace(0, max_iters, max_iters, endpoint=False)
        
                plt.scatter(x_axis,cost_history, color='blue',s=0.1)
                plt.xticks()
                plt.yticks()
                plt.xlabel('iteration')
                plt.ylabel('cost')
                plt.savefig(str(mypath)+'\\3-5\\'+comstr+' '+'concrete_compressive_strength'+'_cost'+'.png')
                plt.clf()
                plt.close()
        f.write(str(max_r2)+"   "+str(count))
        f.close()
    """


    count=0
    if not os.path.exists(Result_path+file):
        os.makedirs(Result_path+file)


    f = open(Result_path+file+'\\_HW_3-6.txt', 'w')
    sum = 99999
    for i in range(1,7):
        for combination in list(combinations(dictionary, i)):
            count=count+1
            a=0
    
            for element in combination:
                    if a==0:
                        X=dictionary[combination[0]]
                        a=1
                    else:
                        X=np.concatenate((X,dictionary[element]), axis=1)       

            a=0
    
            for element in dictionary.items():

                if a==0:        
                    X=element[1]
                    a=1
                else:
                    X=np.concatenate((X,element[1]), axis=1)
            

            print("Now testing:")

            print(combination)
            
            poly = PolynomialFeatures(2)

            X=poly.fit_transform(X)

            #print(X.shape)

            X_train,X_test,y_train,y_test=train_test_split(X,AV ,test_size=0.2)
    
            (row,col)=X.shape
            f.write("Now Testing:\n")

            comstr=""
            for element in range(len(combination)):
                comstr=comstr+combination[element]
                comstr=comstr+" "
            f.write(comstr+"\n") 
            
            

            f.write("Training dataset:\n")
            #huber = HuberRegressor().fit(X, AV)    
            theta = np.random.randn(col,1)    
            learning_rate = 0.003        
            precision = 0.000001        
            max_iters = 11000          
            iters = 0     
        
            theta,cost_history,theta_history = gradient_descent(X_train,y_train,theta,learning_rate,max_iters)

            for i in range(0,len(theta)):
                f.write(str(theta[i])+" ")
            f.write("\n")

            predict_y=np.dot(X_train,theta)
            
            f.write("MSE Score: "+str(mean_squared_error(y_train,predict_y))+"\n")
            f.write("R2_score: "+str(r2_score(y_train,predict_y))+"\n\n")

            if sum > mean_squared_error(y_train,predict_y)+r2_score(y_train,predict_y):
                sum=mean_squared_error(y_train,predict_y)+r2_score(y_train,predict_y)
                final_com=comstr


            f.write("Testing dataset:\n")
            test_predict_y=np.dot(X_test,theta)
            f.write("MSE Score: "+str(mean_squared_error(y_test,test_predict_y))+"\n")
            print("r2_score:",r2_score(y_train,predict_y))
            f.write("R2_score: "+str(r2_score(y_test,test_predict_y))+"\n\n")
            x_axis = np.linspace(0, max_iters, max_iters, endpoint=False)

            plt.scatter(x_axis,cost_history, color='blue',s=0.1)
            plt.xticks()
            plt.yticks()
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.savefig(Result_path+file+'\\'+comstr+' '+'Average_Volumn'+'_cost'+'.png')
            plt.clf()
            plt.close()
    f.write(final_com+'\n')    
    f.close()
    
    
