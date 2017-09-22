# KNN(K邻近算法)
# 距离质心最小。
# 交叉验证就是分为
# K份，第m份为验证集
# 其余为测试集
# 得到k个分数，算平均数
import numpy as np
import pandas as pd
from sklearn  import datasets
ir=datasets.load_iris()
r=ir.data
shee=pd.DataFrame(r)

# 自己的数据可以不要。
from sklearn.model_selection  import train_test_split as split
data_train,data_test,target_train,target_test=split(
   ir.data,
   ir.target,
   test_size=0.3
	)  


from sklearn.neighbors import KNeighborsClassifier as knn

# 参数是分类的个数，可以从数据集中知道
k=knn(n_neighbors=5)
k.fit(data_train, target_train)
score=k.score(data_test,target_test)
print(score) 

# 交叉验证。是独立的评分验证所有的算法都可以
from sklearn.model_selection  import  cross_val_score as sco  
scor=sco(k, ir.data,ir.target,cv=5)
print(scor)
# 预测 # 

result=k.predict([[79,63,8,9]])
print(result)


