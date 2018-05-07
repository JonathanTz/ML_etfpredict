from Get_Data import get_data
from sklearn import preprocessing
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.recurrent import LSTM
import keras


raw_data= get_data(1)
raw_data=raw_data[['open','high','low','volume','close']]
raw_data.dropna(how='any',inplace=True)

##data scaler
def normalize(df):
    new_data=df.copy()  
    min_max_scaler=preprocessing.MinMaxScaler()
    new_data['open']=min_max_scaler.fit_transform(raw_data.open.values.reshape(-1,1))
    new_data['high']=min_max_scaler.fit_transform(raw_data.high.values.reshape(-1,1))
    new_data['low']=min_max_scaler.fit_transform(raw_data.low.values.reshape(-1,1))
    new_data['volume']=min_max_scaler.fit_transform(raw_data.volume.values.reshape(-1,1))
    new_data['close']=min_max_scaler.fit_transform(raw_data.close.values.reshape(-1,1))

    return new_data


##classfy multiple set
new_data=normalize(raw_data)
No_feature=len(new_data.columns)
new_matrix=new_data.as_matrix()
result=[]

for i in range(len(new_matrix)-(20+1)):
    result.append(new_matrix[i:(i+20+1)])
result=np.array(result)
num_train=round(0.9*result.shape[0])

x_train=result[:int(num_train),:-1]
y_train=result[:int(num_train),-1][:,-1]
x_test=result[int(num_train):,:-1]
y_test=result[int(num_train):,-1][:,-1]

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],x_train.shape[2]))
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2]))




###model building
def Model_building(input_len,input_dim):
    d=0.3
    model=Sequential()
    model.add(LSTM(256,input_shape=(input_len,input_dim),return_sequences=True))
    model.add(Dropout(d))## giveup d % neurons

    model.add(LSTM(256,input_shape=(input_len,input_dim),return_sequences=False))
    model.add(Dropout(d))## giveup d % neurons

    model.add(Dense(16,kernel_initializer="uniform",activation='relu'))
    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))

    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    return model

model=Model_building(20,5)
model.fit(x_train,y_train,batch_size=128,epochs=50,validation_split=0.1,verbose=1)



def denormalize(df,nor_value):
    origin=df['close'].values.reshape(-1,1)
    nor_value=nor_value.reshape(-1,1)
    min_max_scaler=preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(origin)
    denor_value=min_max_scaler.inverse_transform(nor_value)

    return denor_value

pred=model.predict(x_test)
denorm_pre=denormalize(raw_data,pred)
denorm_ytest=denormalize(raw_data,y_test)

import matplotlib.pyplot as plt
plt.plot(denorm_pre,color='red',label='predict')
plt.plot(denorm_ytest,color='blue',label='y_test')
plt.legend(loc='best')
plt.show()
