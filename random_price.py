import numpy as np
from Get_Data import get_data
tick=get_data(1)
tick['return']=abs(tick.close-tick.open)/tick.open
vol=np.random.normal(tick[-90:]['return'].mean(),tick[-90:]['return'].std(),5)

ud=[1,-1,1,1,-1]

price=25.2
vollist=(abs(vol)*ud+1)
print (vollist)
print(price*np.cumprod(vollist))
