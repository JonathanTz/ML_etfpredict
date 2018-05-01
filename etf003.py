# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 21:35:31 2018

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

#特徵處理
X = preprocessing.scale(X)

train_sizes, train_loss, test_loss = learning_curve(
        SVR(), X, y, cv=10, scoring = 'mean_squared_error',
        train_sizes = [0.1, 0.25, 0.5, 0.75, 1])
    
        #learning輸入為(1.model, 2.X(學習樣本), 3.y(答案),
        #4.cv(cross_validation)交叉驗證，也就是分成幾層去測試，每層都是不同的切割樣本方式，使結果較為準確
        #5.scoring, 如果是迴歸(regression)要計算誤差值則用mean_squared_error(中文叫均方差)，
        #如果是要分類(classifier)則使用accurarcy，
        #6.最後train_sizes則是在進度10%時做紀錄、25%做紀錄...到100%)
        #train_sizes別忘記加s會debug de半天
        #調整SVC,odel的gamma值來改變學習曲線效果，進而觀察overfitting狀況
        
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

#在這邊因為是分成10層做交叉驗證(cross_validation)所以必須在最後做平均，
#axis為設定矩陣運算時所需要的參數，
#記得若是運算誤差值時，要加負號，才會變正的。
#train_loss算出來是一個5*5的矩陣
#print(train_loss)
#print(train_loss_mean)

#畫圖啦
                                  
plt.plot(train_sizes, train_loss_mean, '-o', color="r", label="Traning")
plt.plot(train_sizes, test_loss_mean, '-o', color="g", label="CrossValidation")
plt.xlabel("Traning Example")
plt.ylabel("Loss")
plt.legend() #讓圖例生效，前面有註明限的label名稱這個才會有效→↑
plt.show

#調整SVC,odel的gamma值來改變學習曲線效果，進而觀察overfitting狀況
#http://blog.csdn.net/szlcw1/article/details/52336824 SVC參數介紹