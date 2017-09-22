# 移动平均
# 简单的移动平均，就是每次移动的时候
# 取k个值，然后算平均数
# 权重平均，就是在简单平均上提高近期数据的权重

import pandas as pd
import numpy as np
from matplotlib.pylab  import *
data=pd.read_csv("data21.csv")
# 由于时间做x但是没有时间，所以模拟时间
x=pd.Series(range(1,len(data)+1))
y=data["公司A"]

result=pd.rolling_mean(y, 5)
plot(x,y,x,result)





# wma
#定义窗口的大小(计算几个值的东西)
wl=5
# 计算窗口的权重
ww=np.arange(1,wl+1)
ww=ww/sum(ww)

reult2=y.rolling(wl).aggregate(lambda x:sum(x*ww))
plot(x,y)
plot(x,reult2)
legend(["old","new1","new2"])
show()