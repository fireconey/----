#压缩变量法综合为一个新的变量，名称变化了。
#对聚类很好，对分类等不太好。
# 特征选择是截取了一部分。


# 这里压缩的是要训练的x数据，不能有y数据
import pandas as pd
import numpy as np
from sklearn.decomposition  import PCA as p  
data=pd.read_csv("data17.csv")
print(data) 
pca=p(n_components=2)
data2=pca.fit_transform(data)
print(pd.DataFrame(data2)) 
