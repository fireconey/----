# 时间序列就是随机数有一定的规律
# 只能是平稳的序列
# 不平稳的序列（如总体上一致上升）
# 需要使用差分法进行平稳处理
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib.pylab import *
import statsmodels.tsa.stattools as ts;
data=pd.read_csv("date23.csv")
print(data) 
time=lambda x:pd.datetime.strptime(str(x),"%Y%m%d")
# 使用apply会每次传入一个数据，使用time（数据）是把
# 数据打包给他了，所以time得到很长的字符串，导致不能解析。
# apply()括号中只要求函数的名称不要括号。
data["date"]=data["date"].apply(time)
data.index=data["date"]
print(data) 

figure(figsize=(10,6))
plot(data["date"],data["value"],label="raw")
legend(["ko"])
# show()


# 使用检测法看看是否平稳
# 预测函数要求传入的index是时间
adf_Data = ts.adfuller(data.iloc[:,1])
# 可以查看第二个参数知道有多大的概率说不平稳
print(adf_Data) 


# 数据的平稳操作
# 是前面一个数据减去后面一个数据
# 基础是最后一个
diff=data["value"].diff(1).dropna()
print(diff) 
# 得到p，q值
ic=sm.tsa.arma_order_select_ic(
	diff,
    max_ar=10,
    max_ma=10,
    ic="aic"
	)


model=sm.tsa.ARMA(diff,(15,9)).fit()
# 由于diff是一维向量,不能使用列名来取值
delta=model.fittedvalues-diff
scor=1-delta.var()/diff.var()
plot(diff.index,diff,"r")
plot(diff.index,model.fittedvalues,"g")
legend(["diff","model"])
show()
# 开始的时间必须在样本中
p=model.predict(start="2016-03-09",end="2020-03-09")
print(p) 

print(ic) 