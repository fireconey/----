# 数据的标准化，是由于有的数据由于量纲不同导致的
# 数据量级有非常大的差异
# 导致数据大的或小的对结果影响大，其他的数据不起作用
import pandas as pd
import numpy as np

# 大小标准化
from sklearn.preprocessing  import MinMaxScaler as mima  
data=pd.read_csv("data16.csv")

model=mima()
data[["累计票房","豆瓣评分"]]=model.fit_transform(data[["累计票房","豆瓣评分"]])
print(data) 

# Z标准化比较特别，直接传入参数
from sklearn.preprocessing  import scale as z  
data[["累计票房","豆瓣评分"]]=z(data[["累计票房","豆瓣评分"]])
print(data) 

# 归一化的标准化
from sklearn.preprocessing import Normalizer  as nor
nor=nor()
data[["累计票房","豆瓣评分"]]=nor.fit_transform(data[["累计票房","豆瓣评分"]])
print(data) 
