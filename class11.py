
#一元非线性y=ax^n+a1x^n-1.....
#变成y=ax1+a1x1.......
import numpy as np
import pandas as mp
from sklearn.linear_model import LinearRegression as ln
import matplotlib as plt
data=mp.read_csv("D:\\bigdata\\4.3\\data.csv",encoding="utf-8")
x=data[["等级"]]
y=data[["资源"]]
font={"family":"SimHei"}
plt.rc("font",**font)

plt.rcParams['axes.unicode_minus'] = False

from pandas.plotting import scatter_matrix as mtr
#mtr(data[["等级","资源"]],alpha=0.8,figsize=(10,10),diagonal="kde")

resul=[]
ip=[]
sco=[]
from sklearn.preprocessing import PolynomialFeatures as pl
#找到最佳的n值
for i in range(225):
     pf=pl(degree=i)
     ip.append(i)
     x1=pf.fit_transform(x)
     lr=ln()
     lr.fit(x1,y)
     score=lr.score(x1,y)
     sco.append(score)
data=mp.DataFrame({
          "i":ip,
          "s":sco
          })

yu=data.sort_values(by=["s"],ascending=False).reset_index(drop=True)["i"][0]#删除原来的index

pf=pl(degree=10)

x1=pf.fit_transform(x)
lr=ln()
lr.fit(x1,y)
score=lr.score(x1,y)



#由于多元非线性的输入数变成了一元的数来计数的，所以预测的也要转换
for i in range(0,20):
     tr=pf.fit_transform([[i]])
     result=lr.predict(tr)
     resul.append(result[0][0])

from matplotlib.pylab import plot,show,draw
plot(x,y)
plot(x,resul)
show()




