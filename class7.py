# 两个事物相关性有多大
import pandas as pd
import numpy  as np

sheet=pd.DataFrame({
	"x":[1,1,0,3,4],
	"y":[2,3,4,0,1]
	})
co=sheet.corr()
print(co)

