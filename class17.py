# 数据的清洗：特征的选择
# 如果数据基本上没有变动不选（方差基本为0-不发散）
# 优选与目标相关性大的
# 可以使用特征消除发，自动帮我们选择特征
# 还可以使用训练过的模型进行选取
import pandas as pd
import numpy as np


data=pd.read_csv("data16.csv")

# 使用方差抽取的方式
from sklearn.feature_selection  import VarianceThreshold  as var
#给个方差的阈值，要大于1
var=var(threshold=3)
data1=var.fit_transform(data[["累计票房","豆瓣评分"]])
# 获取成功当选的列名
name=var.get_support() 
# 由于第二个的方差小于3所以不选
# print(name) 


# 使用相关性（线性回归）选择
data=pd.read_csv("data17.csv")
from sklearn.feature_selection  import SelectKBest  as se 
from sklearn.feature_selection  import f_regression as f  
# k为指定你要多少个特征
select=se(f,k=2)
feature=data[['月份', '季度', '广告费用', '客流量']]

data2=select.fit_transform(feature,data["销售额"])
name=select.get_support()
print(name) 


# 递归特征消除法
from sklearn.feature_selection  import RFE  
from sklearn.linear_model  import LinearRegression  
rf=RFE(estimator=LinearRegression(),
	n_features_to_select=2)
data3=rf.fit_transform(feature,data["销售额"])

name=rf.get_support()
print(name) 


# 模型训练法
from sklearn.feature_selection  import SelectFromModel as fr
# 先初始化训练的模型
model=LinearRegression()
# 初始化特征选取的模型
selemode=fr(model)
selemode.fit_transform(feature,data["销售额"])
print(selemode.get_support()) 


