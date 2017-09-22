# 决策树
import pandas as pd
import numpy as np

data=pd.read_csv("data5.3.csv")


column=["Gender", "ParentEncouragement"]
fixdata=pd.get_dummies(data,columns=column,prefix=column,prefix_sep="=",drop_first=True)


inputdata=fixdata[['ParentIncome', 'IQ', 'Gender=Male',
                   'ParentEncouragement=Not Encouraged'
			]]

outputdata=fixdata[["CollegePlans"]]


from sklearn.tree  import DecisionTreeClassifier as dec
model=dec(max_leaf_nodes=8)

model.fit(inputdata, outputdata)

score=model.score(inputdata,outputdata)
print(score)


# 一下是输出决策树
from sklearn.tree  import  export_graphviz as ex
# with是单独的语法与上面的不一样。
# with open("data.cot","w")  as f
f=open("data.dot","w")
f=ex(model,out_file=f)

import pydot
from sklearn.externals.six  import StringIO

filedot=StringIO()
ex(model,
	out_file=filedot,
	# 要和表中的第一个相同
	class_names=["计划","不计划"],
	# 以下从表哑变量的表中得到
    feature_names=["父母的收入","智商","性别=男的","父母的鼓励=不鼓励"],
    # 一下设置图片样式special——表示使用有中文
     filled=True,rounded=True,special_characters=True
    )

gr=pydot.graph_from_dot_data(filedot.getvalue())
gr.get_node("node")[0].set_fontname("Microsoft YaHei")
gr.write_png("tree.png")

print(fixdata["CollegePlans"]) 


