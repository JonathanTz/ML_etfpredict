import pandas as pd
import os

fileNo={1:'taetfp',2:'tasharep',3:'tetfp',4:'tsharep'}
def get_data(idx):
      ##check whether pickle exist
      if not os.path.isfile(fileNo[idx]+'.pickle'):
            ###there is a error if encoding=big5
            data=pd.read_csv("Data/"+fileNo[idx]+".csv",encoding='cp950')
            data.columns=['tick','date','name','open','high','low','close','volume']
            data.name=pd.DataFrame([(lambda x:data.name[x].strip())(x) for x in range(len(data.name))])
            data.to_pickle(fileNo[idx]+'.pickle')
      data=pd.read_pickle(fileNo[idx]+'.pickle')
      dt1=data.groupby(['name'])['tick'].mean()
      
      ###Show all ticks, and you should input the ticks u want,
      print(dt1)
      ticks=input('enter the tick you want')
      ###

      price=data[data.tick==int(ticks)]
      return price
print(get_data(2))
