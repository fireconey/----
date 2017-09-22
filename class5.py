# 扩散法：对应列的对应数据相运算

import pandas as pd
import numpy as np
sheet=pd.DataFrame({
"A":[1,2,3],
"B":[4,5,6],
"C":[7,8,9]

	})

data = sheet.groupby(
    by=["A", "B"]
)["B"].agg({
    "计数":np.size
}).reset_index().sort_values(
    by=["计数"],
    ascending=False
);

data=data.pivot_table(index="B",columns="A" ,fill_value=0,values="计数")
# print(data)

sheet1=pd.Series([1,2,4])
# 一维向量和表相乘时，是表的一行乘向量的
# 一列，且要求一维向量的行名称和表的列名称对应否则
# 出错如下面的sheet列名为a，b，c，向量行名为0,1,2：
# print(sheet*sheet1)

print((sheet.T)*sheet1)



# print(sheet1)
# 下面修改一维向量的行名，可以和pivothou的表乘
# sheet1.index=sheet["A"]
# print(data*sheet1)
# # 不是向量就不要紧
# print(sheet*[1,2,3])


