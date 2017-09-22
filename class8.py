# 线性回归
# 是两个变量的平均差的积除以
# 自变量的方差
# 相关性是两个平均差的积除以
# 两变量的标准差的积再除以样本个数
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as line
# pylab 中的scatter画直方图，plot画曲线。
from matplotlib.pylab import*
data=pd.read_csv("data.csv")
print(data)
cor=data.corr()
print(cor)
#scatter(data.广告投入,data.销售额)
#show()
model=line()
# 一个中括号表示一维向量
# 两个中括号表示表
# sklearn要求是表的格式
x=data[["广告投入"]]
y=data[["销售额"]]
# print(x)

result=model.fit(x, y)

# 线性回归好坏使用
# 残差的方差除以回归线的残差
score=model.score(x,y)
print(score)
# 截距
intercept=model.intercept_
print(intercept)

# 相关系数
z=model.coef_
print(z)

pridect=model.predict(100)
print(pridect[0][0])
plot(y,x)
show()
