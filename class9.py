# 多重线性回归
# 画多个图要使用pandas的画图工具
import pandas as pd
from matplotlib.pylab import plot as plt
# 下面的输入错误，包已经改了
# from pandas.tools.plotting import scatter_matrix as sc
from pandas.plotting import  scatter_matrix as sc
data=pd.read_csv("data1.csv")
from sklearn.linear_model import LinearRegression as line

#四方图，diagobal表示在对角线的位置画什么
#kde表示正态分布图
#sc(data,figsize=(10,10),diagonal="kde")

model=line()
# 由于要求填入的数据是表，不是一维向量所以使用双【】
x=data[["店铺的面积","距离最近的车站"]]
y=data[["月营业额"]]
model.fit(x, y)

score=model.score(x,y)
print(score)

pridect=model.predict([[200,400]])
print(pridect)




