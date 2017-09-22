# 分类就是指定了类别名称
# 聚类是没有指定类的叫啥，只想分类，是自己分为不固定的类别
import pandas as pd
import numpy as np
from sklearn.decomposition  import PCA   as p
from pandas.plotting import scatter_matrix as kl
from matplotlib.pylab import *
data=pd.read_csv("data19.csv")
col=[
    '工作日上班时电话时长', '工作日下半时电话时长',
    '周末电话时长',
    '国际电话时长', '总电话时长', '平均每次通话时长'
]


# 设置字体
font={"family":"SimHei"}
matplotlib.rc("font",**font)

kl(data[col],figsize=(10,10),diagonal="kde")
# show()

pca=p(n_components=3)
data1=pca.fit_transform(data[col])
sheet=pd.DataFrame(
					data1
					)

print(sheet)

# 使用pylab来点图是单个的
import matplotlib.pylab as plt
plt.rc("font",**font)
plt.scatter(sheet[0],sheet[1])
# show()
# pandas的点图是两两的
# kl(sheet[[0,1]])
# show()
#


# 使用kmeans算法
from sklearn.cluster  import KMeans as km
model=km(n_clusters=3)
reslut=model.fit(sheet)
pr=model.predict(sheet)
print(pr)
# c表示颜色使用预测的分类来做
# 训练后得到的是模型不能使用模型作颜色的标准
plt.scatter(sheet[0], sheet[1],c=pr)
# show()
dm=pd.DataFrame(columns=col+["分类"])
data["分类"]=pr
sorda=data.sort_values(by=["分类"],ascending=False)
for i in range(3):
	kh=data[data["分类"]==i]
	tongji=kh.mean()
	# print(tongji) 
	fu=dict(tongji)
	dm=dm.append(fu,ignore_index=True)
	# print(type(fu))





# #以下为统计各个类的差异
# #定义一个空的列表，列明为fcolumns再增加一个“分类”
# dm=pd.DataFrame(columns=col+["分类"])
# #上面通关过kmeans算的了一个分类的数列(依据)ptarget这里就是分类
# data_gb=data[col].groupby(pr)
# i=0

# plt.figure(figsize=(30,30))
# #data_gb.groups是得到分组的标记
# for g in data_gb.groups:
# #      得到某一分类的平均值  data_gb.get_group(g) 得到分组的所有标记下的所有值。
#       rm=data_gb.get_group(g).mean()
# #      标记分类为g
#       rm["分类"]=g;
# #      数据田间appened是一列数据添加到一行中取
#       dm=dm.append(rm,ignore_index=True)
#       subd=data_gb.get_group(g)
#       for column in col:
#             i=i+1;
# #            i是标记那第几个图。3行5列
#             p=plt.subplot(3,5,i)
#             p.set_title(column)
#             p.set_ylabel(str(g)+"分类",fontsize=12)
#             plt.hist(subd[column],bins=20)
