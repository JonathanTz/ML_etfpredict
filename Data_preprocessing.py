import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from Get_Data import get_data



##load data and calculate label
all_data=get_data(1)
all_data['close_1']=all_data.close.shift(-1)
all_data['open_1']=all_data.open.shift(-1)
all_data['stdev']=(all_data.close-all_data.open)/all_data.open
all_data=all_data[:-1]
labeltest=[]
for i in range(len(all_data)):
    ##two class
    if all_data.close.shift(-1)[i]/all_data.open.shift(-1)[i]>(all_data.stdev.quantile(.55)+1.000):
       labeltest.append([1,0,0])
    elif all_data.close.shift(-1)[i]/all_data.open.shift(-1)[i]<(all_data.stdev.quantile(.45)+1.000):
        labeltest.append([0,1,0])
    else:
        labeltest.append([0,0,1])

    #five class
##    f1=(all_data['close_1'][i]/all_data['open_1'][i]>1)*1
##    f2=(all_data['close_1'][i+1]/all_data['open_1'][i+1]>1)*1
##    f3=(all_data['close_1'][i+2]/all_data['open_1'][i+2]>1)*1
##    f4=(all_data['close_1'][i+3]/all_data['open_1'][i+3]>1)*1
##    f5=(all_data['close_1'][i+4]/all_data['open_1'][i+4]>1)*1
##    labeltest.append([f1,f2,f3,f4,f5])

all_data['label']=pd.Series(labeltest,index=all_data.index)
##all_data['label']=pd.Series(labeltest,index=all_data.index[:-5])
all_data['volatility']=np.std([1,2,3])

#random sample
def sample_suffling(sample_x,sample_y,length=20):
    fset=[]
    sample_len=len(sample_x)-length
    
    for i in range(sample_len):
        sample_x_chunk=np.array(sample_x[i:i+length])
        min_max_scaler=preprocessing.MinMaxScaler()
        sample_x_chunk=min_max_scaler.fit_transform(sample_x_chunk)
        fset.append([sample_x_chunk,sample_y[i+length]])

    random.shuffle(fset)

    return fset
def sample_visualizing(sample_x):
    pic_array=[]
    for i in range(len(sample_x)):
        pic=[]
        sample_x[i]=np.transpose(sample_x[i])
        for j in range(len(sample_x[i])):
            picline=[]
            for k in range(len(sample_x[i][j])):
                append_pt=[sample_x[i][j][k],sample_x[i][j][k],sample_x[i][j][k]]
                picline.append(append_pt)
            pic.append(picline)
        pic_array.append(pic)
        #plt.imshow(pic)
        #plt.show()
        
    return (pic_array)



##calculate indicator and preprocessing
def data_processing(test_size=0.05):
    all_data['ma5']=all_data.close.rolling(window=5).mean()
    all_data['ma20']=all_data.close.rolling(window=20).mean()
    all_data['ma60']=all_data.close.rolling(window=60).mean()
    all_data.dropna(inplace=True)

    df_X=all_data[['open','high','low','close','volume','ma5','ma20','ma60']]
    df_y=all_data['label']
    #random sample
    datasets=sample_suffling(df_X,df_y)
    datasets=np.array(datasets)
    testing_size=int(test_size*len(datasets))
    #split sample to two sample
    train_x=list(datasets[:,0][:-testing_size])
    train_x=sample_visualizing(train_x)
    train_y=list(datasets[:,1][:-testing_size])
    
    test_x=list(datasets[:,0][-testing_size:])
    test_x=sample_visualizing(test_x)
    test_y=list(datasets[:,1][-testing_size:])
    return train_x,train_y,test_x,test_y


