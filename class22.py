# 季节性时间分解是在一定时间段上
# 即有时间周期的规律又有总体的趋势
# 还有误差的
import pandas as pd
import numpy as np
import statsmodels.api as sm
from  matplotlib.pylab import *
data=pd.read_csv("data22.csv")
# 处理数据中的时间字符，变成时间的格式
# 由于strptime只能处理字符型的所以要强转
time=lambda date:pd.datetime.strptime(
	str(date),"%Y%m%d")

# 
data["时间"]=data["时间"].apply(time)
print(data) 
print(data["总销量"].values) 
# 取出后是一维的向量不在是一维的表
# print(type(data["总销量"]))
# 注意sessional_decompose必须是数组不能对一维
# 向量,freq是周期
result=sm.tsa.seasonal_decompose(
   data["总销量"].values,
   freq=10
	)
print(result) 
result.trend
result.seasonal
result.resid
ty=result.plot()
# 要使用matplot的show自带的有close()

show()
