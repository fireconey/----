# 数据透视表 pivot_table() 卡尔基模型

import pandas as pd
import numpy as np
sheet=pd.DataFrame({
    "A":[0,1,3],
    "B":[1,2,4],
    "C":[7,7,7]    })
# groupby得到有组别标签的分组对象变成list可以查看,
# 分组后其原来的index也有
#agg意思是属于类别（分组的名称）的情况下其他列有值得有多少个
#所以后面选取的列不论C，还是B都是一样的值,默认为在
#index情况下的,agg后会，他会创建新的表
data=sheet.groupby(by=["A","B"])["B"].agg({"计数":np.size}).reset_index().sort_values(
    by=["计数"],
    ascending=False
);
data2=data

# print(sheet)
#使用各个列的值当名称了
#pivot_table 使统计的矩阵法
tf=data.pivot_table(values="计数",columns="A",index="B",fill_value=0)
# print(tf)

def hanlder(x): 
    return (np.log2(len(sheet)/(np.sum(x>0)+1)))


IDF = tf.apply(hanlder)
print(IDF)
TF_IDF =pd.DataFrame(tf*IDF)#进行的是矩阵乘法不是书上的乘
print(TF_IDF)

