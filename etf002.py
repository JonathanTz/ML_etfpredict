# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 19:49:38 2018

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve #使用learning_curve模組來可視化機器學習過程
from sklearn.svm import SVR #導入支持向量機模型
from sklearn import preprocessing #要進行標準化則要先導入這個套件
import matplotlib.pyplot as plt #導入畫圖模組


#資料處理
df = pd.read_csv('tw0050.csv', encoding='big5')
df.columns = ['tick','date','name','open','high','low','close','volume']
df.index = df['date']
df = df.set_index('date') #轉成DatetimeIndex
df_close = df['close'] #先將收盤價提出來
df.drop('tick', axis=1, inplace=True) #刪掉不需要的columns，使用inplace參數確實刪除
df.drop('name', axis=1, inplace=True) #刪掉不需要的columns，使用inplace參數確實刪除
df.drop('close', axis=1, inplace=True) #刪掉不需要的columns，使用inplace參數確實刪除
df_data = df.values #將df直接換成dataset的矩陣

#模型建構

X = df_data
y = df_close


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3) #記得要放test_size
model = SVR() #機器學習的模型是使用SVC
model.fit(X_train, y_train) #放入訓練的data，用fit訓練
model.predict(X_test) #考試囉
#print(y_test) #對答案
print(model.score(X_test, y_test)) #測分數囉












