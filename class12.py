# 对表的列相加得到的是以
# 列名为行名的，以0为列名的
# 一维向量(原来的行名没有了)
import pandas as pd
import numpy as np


sheet=pd.DataFrame({  
					"A":[1,2,3,4],
					"B":[6,7,8,9],

					},index=["w","e","u","p"])
print(sheet) 

data=sheet.apply(np.sum,axis=0)
print(data) 