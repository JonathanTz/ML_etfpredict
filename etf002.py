# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 19:49:38 2018
@author: user
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split #新的import方法，上面的好像要被除掉?
from sklearn.learning_curve import learning_curve #使用learning_curve模組來可視化機器學習過程
from sklearn.svm import SVR #導入支持向量機模型
from sklearn import preprocessing #要進行標準化則要先導入這個套件
import matplotlib.pyplot as plt #導入畫圖模組
from datetime import datetime


#資料處理
df = pd.read_csv('tw00500427.csv', encoding='big5')
df.columns = ['tick','date','name','open','high','low','close','volume']
df.drop('tick', axis=1, inplace=True) #刪掉不需要的columns，使用inplace參數確實刪除
df.drop('name', axis=1, inplace=True) #刪掉不需要的columns，使用inplace參數確實刪除
df['DateTime'] = df['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
df.index = df['DateTime']
df_close = df['close'] #先將收盤價提出來
df.drop('date', axis=1, inplace=True) #刪掉不需要的columns，使用inplace參數確實刪除
df.drop('DateTime', axis=1, inplace=True) #刪掉不需要的columns，使用inplace參數確實刪除
df = df.resample('W-SUN').mean() #轉成週資料 用MON會吃到下禮拜的
df = df.dropna(axis=0, how='any') #去除na值
df.drop('close', axis=1, inplace=True) #刪掉不需要的columns，使用inplace參數確實刪除
df.drop(df.index[0], inplace=True) #把第一列資料刪除
df_data = df.values #將df直接換成dataset的矩陣

df_close_MON = df_close.resample('W-MON').ffill()
df_close_TUE = df_close.resample('W-TUE').ffill()
df_close_WED = df_close.resample('W-WED').ffill()
df_close_THU = df_close.resample('W-THU').ffill()
df_close_FRI = df_close.resample('W-FRI').ffill()

                        
df_target = pd.DataFrame({
                    'Monday':df_close_MON,
                    'Tuesday':df_close_TUE,
                    'Wednesday':df_close_WED,
                    'Thursday':df_close_THU ,
                    'Friday':df_close_FRI
                    })

#print(df_target)
##creat dataframe with all date in sample period
raw_date = pd.DataFrame(index=pd.date_range('1/7/2013','5/3/2018'))
merge_date = pd.concat([raw_date,df_target],axis=1)
##add a column record dayofweek
merge_date['day_week']=merge_date.index.dayofweek
##refresh date from dayofweek=0 to dayofweek=4
start = merge_date[merge_date.day_week==0].index[0].date()
end = merge_date[merge_date.day_week==4].index[len(merge_date[merge_date.day_week==4])-1].date()
merge_date = merge_date[start:end]

all_data = pd.concat([merge_date['Monday'].reset_index(drop=True),
                    merge_date['Tuesday'].shift(-1).reset_index(drop=True),
                    merge_date['Wednesday'].shift(-2).reset_index(drop=True),
                    merge_date['Thursday'].shift(-3).reset_index(drop=True),
                    merge_date['Friday'].shift(-4).reset_index(drop=True),
                    ],axis=1)
all_data.index = merge_date['Monday'].index
all_data = all_data.dropna(axis=0,how='all')

all_data.drop(all_data.index[5], inplace=True) #刪除過年資料 2013
all_data.drop(all_data.index[110], inplace=True) #刪除過年資料 2015
all_data.drop(all_data.index[159], inplace=True) #刪除過年資料 2016

all_data = all_data.values #矩陣化

#特徵資料製作



#模型建構
X = df_data
y = all_data
#特徵處理
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3	) #記得要放test_size
#y_train = np.argmax(y_train, axis=1)
model = SVR() #機器學習的模型是使用SVC
model.fit(X_train, y_train) #放入訓練的data，用fit訓練
model.predict(X_test) #考試囉
#print(y_test) #對答案
print(model.score(X_test, y_test)) #測分數囉















