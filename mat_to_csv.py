import pandas as pd
import h5py
import time
from datetime import datetime
import numpy as np

filepath = 'DailyPP.mat'
file1=h5py.File(filepath,'r') #read-only
#for k,v in file1.items():  #字典格式
 #   print(k)

#先清空上次运行程序时产生的csv文件dailyp，这么做的问题是开头有个空格
k={}
kdata=pd.DataFrame(k)
kdata.to_csv('dailyp.csv')

#按列写入file1中的data
for key in file1.keys():
    k=str(key)
    f=file1[key][:][0]
    data={k:f}
    dfdata=pd.DataFrame(data)
  #  df=open('dailyp.csv','w')
   # df.write('\n')        #
    dfdata.to_csv('dailyp.csv',mode='a',index=False)  #mode='a'，可以追加写入,index=False, 包含名称k


#把时间数字序列转换成常规时间
DP = pd.read_csv('dailyprice.csv')
time=[]
#把时间数 转化为年月日时间
for day in DP['Days']:
    d = datetime.fromordinal(day)
    ymd = d.strftime('%Y-%m-%d')
    time.append(ymd)
Time={'days':time}
dfTime=pd.DataFrame(Time)
dfTime.to_csv('dailyprice.csv',mode='a',index=False)


#question is 追加写入时，如何换成新的列