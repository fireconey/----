# 使用sklearn包
import pandas as pd
import numpy  as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
contents=["我 是 中国 人",
		  "你 是 美国 人",
		  "她 叫 什么 名字",
		  "她 是 谁?",
]
# min_df=0表示字符小于0的不省略
# token_pattern表示正则
vertor=CountVectorizer(min_df=0,token_pattern=r"\b\w+\b")
# 得到有位置的描述的单位矩阵的说明（次品统计）
txtv=vertor.fit_transform(contents)
print(txtv)

print(txtv.todense())
print(vertor.vocabulary_)

# 变成卡尔基模型表使用坐标标记的描述
tf=TfidfTransformer()
tfid=tf.fit_transform(txtv)
print(tfid)
# 变成了矩阵
matx=tfid.toarray()
# print(matx)

# argsort是排序,axies=1按照行排列，结果是一个给定下表的矩阵
matrx=np.argsort(matx,axis=1)
# 得到特征名称（行名）
names=vertor.get_feature_names();
print(names)

# names告诉有哪些名字，matrx是根据其中的定位来取值。
# index用到的是map技术，使用0,1,2.....来对应names
# 指定names是说明获取名称
keywords=pd.Index(names)[matrx].values
print(keywords)



tagDF =pd.DataFrame({
     
    'tag1':keywords[:, 0], 
    'tag2':keywords[:, 1], 
    'tag3':keywords[:, 2], 
    'tag4':keywords[:, 3], 
    'tag5':keywords[:, 4]
})

