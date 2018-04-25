import pandas as pd
data=pd.read_csv("Data/taetfp.csv",encoding='big5')
data.columns=['tick','date','name','open','high','low','close','volume']
data.name=pd.DataFrame([(lambda x:data.name[x].strip())(x) for x in range(len(data.name))])
def GetTick(dt):
    dt1=dt.groupby(['name'])['tick'].mean()
    print(dt1)

GetTick(data)
def etf_p(*args):
    ticks=input('enter the tick you want')
    price=data[data.tick==int(ticks)]
    return price
print(etf_p())
a=1
