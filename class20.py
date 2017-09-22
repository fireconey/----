# 1、根据对购买商品的评分得到
# 评分向量
# 2、使用评分向量来计算客户的相似度
# 3、找到距离一定在内的所有客户或
# 规定客户的数量，找到他们买的商品
# 排序他们的商品。
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as ou
data=pd.read_csv("data20.csv")
print(data)

#pivot如果没有指定填充的值则使用剩下的列填充
#一般指定填充值通过value=""
#aggfunc表示条件一模一样的时候怎样处理，如一个用户购买了同意一个商品那么
#他的评分是多个评分相加等等。把分数放在那种情况中
user=data.pivot_table(index="UserID",columns="ItemID",aggfunc=np.sum,fill_value=0)
print(user) 

# 由于列明的一行前有名称，表米有名称所以要处理
user.columns=[101,102,103,104,105,106,107]

print(user) 
# 计算欧氏距离，变成了距离的矩阵
dist=pd.DataFrame(ou(user))
dist.index=user.index
dist.columns=user.index
print(dist) 


# 计算相似度
sim=1/(1+dist)
print(sim) 



# 要找用户为3的所有相似用户
#邻居数为3
k=3
# 用户为3的人
userid=3

# 得到的只是排序的标签
simuser=sim.sort_values(userid,ascending=False)[userid].index[1:1+k]
print(simuser) 


# 找出最相近的3个
usersim=sim.\
ix[simuser,userid]
print("*************找到相似的3个**********") 
print(usersim) 

# 相似用户的评分,
# 是相似的3个用户对同一个商品的评价的高低。
# usersim是一维化的向量
# 一维向量和矩阵乘，是向量的乘法。就是把
# 矩阵看成多个向量组成的。
# 每个属性与向量相乘得到一个结果。
# 总的结果集还是一个一维向量,
# 一维向量放在前面是纵向乘，放在后面是横向乘，
# 有时位数不一样就会出错。
score=pd.DataFrame(
    np.dot(usersim,user.ix[simuser])
	)
# print("***********") 
# print(usersim,type(usersim))
# print(user.ix[simuser] ,type(user.ix[simuser]) )  
# print(score) 
reslut=user.columns[score.sort_values(0,ascending=False).index.values]
# print(reslut) 