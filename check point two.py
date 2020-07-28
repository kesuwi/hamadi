# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:58:46 2020

@author: DARDOURI
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



data = pd.read_csv("titanic-passengers.csv",sep=";")
print("data\n",data,"\n"*5)
print(data.info(),"\n"*5)
print(data.describe())
na_counts = data.isnull().sum().sum()
print(data.isnull().sum())
print("\nmissing values count\n",na_counts)
data0 = data.drop("Cabin",axis=1)
data1 = data0.fillna(data["Age"].mean())
data1.drop("Name" , axis = 1 ,inplace =True)
print(data1.info())
data1["Survived"] = LabelEncoder().fit_transform(data1["Survived"])
data1["Sex"] = LabelEncoder().fit_transform(data1["Sex"])
print(data1 ["Embarked"])
one_hot = pd.get_dummies(data1["Embarked"], dtype = int)
data1 = data1.drop("Embarked", axis = 1 )
data1 = data1.drop("Ticket" , axis = 1)
one_hot.columns = ['a','C', 'Q', 'S']
one_hot.drop("a", axis = 1 , inplace =True )
data_final = data1.join(one_hot)
data_final.drop("PassengerId" , axis = 1 , inplace = True)
print(data_final)
print(data_final.info())
print(data_final.describe())
print(plt.style.available)
plt.style.use("fivethirtyeight")
index = np.array((),dtype =float)
for i in data_final["Sex"]:   
    if i==0:
        i+=0.4
        index = np.append(index,i)
    else:
        i-=0.4
        index = np.append(index,i)
#t he left one if for female where the right one is for male"
fig1 , ax1 = plt.subplots(nrows = 1 , ncols = 1 , figsize=(6.5,6.5))
ax1.scatter(index,data_final["Age"], c = data_final["Survived"],
            cmap = "plasma_r" ,alpha = 0.75 , s = 50 ,edgecolor ="k",
            label = {"blue":"dead","yellow":"alive"})
ax1.set_title("survivals" , fontsize = 20)
ax1.set_xlim(0,1)
ax1.set_yscale("log")
ax1.set_xlabel("female   male",x = 0.5)
ax1.set_ylabel("Age")
plt.colorbar(cm.ScalarMappable(), ax =ax1, label = ["death","alive"] )
plt.legend(fontsize = 13 , loc=(0,0))
# c,q,s = 0,0,0
# for i in data["Embarked"]:
#     if i=="C":
#         c+=1
#     elif i=="Q":
#         q+=1
#     else:
#         s+=1
# cs,qs,ss = 0,0,0
# for i in range(len(data_final["Survived"])):
#                 if data_final.iloc[i][0]==1:
#                     if data_final["C"][i] == 1:
#                         cs+=1
#                     if data_final["Q"][i]==1:                      
#                         qs+=1
#                     if data_final["S"][i] ==1:
#                         ss+=1
first_class,second_class,third_class = 0,0,0
for i in range(len(data_final["Survived"])):
    if data_final["Survived"][i]==1:
        if data_final["Pclass"][i]==1:
            first_class+=1
        elif data_final["Pclass"][i]==2:
            second_class +=1
        else:
            third_class+=1
pclass = np.array((first_class, second_class,third_class)) 
pclass_label = np.array(("first class","second class","third class"))       
fig11 ,ax11 = plt.subplots(nrows = 1 ,ncols = 1 , figsize=(8,8))
ax11.bar(pclass_label,pclass,color = np.array(('y','r','b'),dtype = str),
         edgecolor="k")
ax11.set_xlabel("Pclasses")
ax11.set_ylabel("Survivals")           
ax11.set_title("f(Pclass)= Survival")
#labels = np.array(("1st","2nd","3rd"))
# handles = [plt.Rectangle((0,0), 1, 1, color = np.array(('y','r','b'),dtype = str))]
plt.legend([ 'F' ,'S', 'T'])

groupby_data = data_final[['Survived','Pclass']].groupby(['Survived'],as_index = True).mean()
print(groupby_data)
fig2 ,(ax20,ax21) = plt.subplots(nrows=2 , ncols =1  ,figsize = (6,6))
# ax20.scatter(data_final["SibSp"], data_final["Survived"], marker ='o')
ms,fs = 0,0
for i in range(len(data_final["Sex"])):
    if data_final["Survived"][i]==1:
        if data_final["Sex"][i]==1:
            ms+=1
        elif data_final["Sex"][i] ==0:
            fs+=1
b0,b1,b2,b3,b4,b5,b6 = 0,0,0,0,0,0,0
for i in range(len(data_final["Sex"])):
    if data_final["Survived"][i]==1:
        if data_final["Parch"][i]==0:
            b0+=1
        if data_final["Parch"][i]==1:
            b1+=1
        if data_final["Parch"][i]==2:
            b2+=1
        if data_final["Parch"][i]==3:
            b3+=1
        if data_final["Parch"][i]==4:
            b4+=1
        if data_final["Parch"][i]==5:
            b5+=1
        if data_final["Parch"][i]==6:
            b6+=1
arr = np.array((b0,b1,b2,b3,b4,b5,b6))
par_count = pd.value_counts(data_final["Parch"]) 
ax20.pie(par_count , explode = [0,0,0,1,1,1,1])
ax21.stackplot([0,1,2,3,4,5,6],arr)
ax21.set_xlim(0,6)
print(par_count)
#create new columns
data_final["Title"] = data["Name"]  
print(data_final.info())         
# da = np.array((c,s,q))
# da0 = np.array(("Cherbourg","Queenstown","Southampton"))
# colord = np.array(("#03cafc","#fc5603","#03fc0f"))
fig3,ax3 = plt.subplots(nrows = 1 , ncols = 1 , figsize = (6,5))
# ax3.scatter(data_final["Title"],data_final["Sex"])
# ax3.scatter(data_final["Title"],data_final["Age"])
# ax3.scatter(data_final["Title"],data_final["Fare"])
print(data_final["Title"])
for i in range(len(data_final["Title"])):
    if "Capt"  or "Col" or "Major" or "Dr" or "Rev" in i:
        data_final["Title"].replace(to_replace=data_final["Title"][i],value ="Officer" ,inplace = True)
    elif "Jonkheer" or "Don" or "Sir" or "Lady" or "the Countess" or "Dona" in i:
        data_final["Title"].replace(to_replace=data_final["Title"][i],value = "Royalty",inplace = True)
    elif  "Mme" or "Mlle" or "Miss" or "Ms" or "Mr" or "Mrs" in i:
        data_final["Title"].replace(to_replace=data_final["Title"][i],value ="Mrs",inplace = True)
    elif  "Master" in i:
        data_final["Title"].replace(to_replace=data_final["Title"][i],value = "Master",inplace = True)

# print(data_final["Title"])
    
# ax30.set_title("distribution of Embarkers")
# plt.show()
# print(data_final.cov())
# def plot_correlation_map( df ):

#     corr = df.corr()

#     s , ax = plt.subplots( figsize =( 12 , 10 ) )

#     cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

#     s = sns.heatmap(

#         corr, 

#         cmap = cmap,

#         square=True, 

#         cbar_kws={ 'shrink' : .9 }, 

#         ax=ax, 

#         annot = True, 

#         annot_kws = { 'fontsize' : 12}

#         )
# print( plot_correlation_map(data_final))
# print(data_final)

                       
"""
this function produces a square matrix that describe the correlation between each two pairs
(x,y) of our features . we not that the matrix is symetric along the diagonal so we can
restrict our study to the first half.
first of all the all diagonal values is 1 wich is obvious because cor(x,x) = 1.
we see that the majority of correlations are negative  and between -0 and -0,2 ;so we could 
affirm that the they vary in  opposite direction, but as we push on the upper left  and bottom right
edges the correlation decrease even more to reach values such that {-0,54,-0.34,-0.5,-0,78}
wich indicate that those paires approach  more of beeing linearly related.
finally some pairs at the corner and the center are positivel  correlated with values
not exceeding 0.5.
thus, we can infere that the "Fare" feauture and the "Southampton Embarker" feautures are
strongly negatively correlated where as the age is relatively weakly correlated with evey 
other feauture except the "Pclass"  in wich the correlation value is "-0.33" 
"""