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
df = df.resample('W-MON').mean() #轉成週資料
df = df.dropna(axis=0, how='any') #去除na值
df.drop('close', axis=1, inplace=True) #刪掉不需要的columns，使用inplace參數確實刪除
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


'''
#模型建構

X = df_data
y = df_close

#特徵處理
X = preprocessing.scale(X)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1	) #記得要放test_size
model = SVR() #機器學習的模型是使用SVC
model.fit(X_train, y_train) #放入訓練的data，用fit訓練
model.predict(X_test) #考試囉
#print(y_test) #對答案
print(model.score(X_test, y_test)) #測分數囉
'''











