# -*- coding: utf-8 -*-
import os
import os.path
import numpy as np
import pandas as pd
from scipy.misc import imread


filespath=[]
for root,dir,files in os.walk("sample"):
    #root会找出所有的路径，包括上面已知的路径
	# print(root)
	#如果文件下没有任何文件夹就打印空的list
	# print(dir)
	# print(file)
	# 由于root下有多个文件所以一次root会导入多个集，
	# 要取出单个文件下面这个会出错
	# filespath.append(root.join(file))
	for file in files:
		# print(file)

		# 数据.join(数据)表示使用前面的当分隔符，链接括号中的
		# os.path.join()不同
		filespath.append(os.path.join(root,file))
		# print(filespath)

	# print("\n")
contents=[]
for root in  filespath:
	op=open(root,"r",encoding="utf8")
	# 有的有空格或回车不能读取报Decode error - output not utf-8
	# 使用.strip()去除
	contents.append((op.read()).replace("\n","").replace("&nbsp","").replace("\t",""))
# print(contents)



data=pd.DataFrame({
	"filespath":filespath,
	"contents":contents
	})
# print(data)
# agg表示聚合运算的，括号中要指定运算
# 聚合后的数index变成了index，不能排列，要恢复
# 下面只是截取contents的数据，所以不能排列filepath
# data1=data.groupby(by=["contents"])["contents"].agg({"计数":np.size}).reset_index().sort_values(by=["contents"])
# print(data1)


# 分词
import jieba as c
segments=[]
filespat=[]
# 表是个整体不可拆开，所以直接in data1不行
# 要拆开数据，下面是迭代每一行。
for  index,row in data.iterrows():
	# print(row)
	filepa=row["filespath"]
	conten=row["contents"]
	seg=c.cut(conten)
	for sg in seg:
		# 需要判断大小否则有空格，有的空格使用replace不能删除。
         if len(sg)>1:
              segments.append(sg)
              filespat.append(filepa)
#         else:print(sg)


# print(segments)

clearndata=pd.DataFrame({
	"seg":segments,
	"filespath":filespat
	})


path="StopwordsCN.txt"
# 文件中有非法的字符文件不能读取出现Error tokenizing data. C error: EOF inside string starting at line 13
# 在文件的13行（就是引号有问题）
stop=pd.read_csv("StopwordsCN.txt",encoding="utf-8",index_col=False)
# isin()是pandas的函数，用于表，下面是错的。表只能使用isin 不能使用 in
# data1=data[~data.contents.isin(object(open(path,"r",encoding="utf8").read()))]
data2=clearndata[~clearndata.seg.isin(stop.stopword)]

#print(data2)

plotdata=data2.groupby(by=["seg"])["seg"].agg({"计数":np.size}).reset_index().sort_values(by=["计数"],ascending=False)
# print(plotdata)

from wordcloud  import WordCloud as w,ImageColorGenerator as ge
import matplotlib.pylab as plt
# work=w(font_path="simhei.ttf",
# 	background_color="white")
# # 变成字典是，索引是字典名称的。所以要成索引
# words=plotdata.set_index("seg").to_dict()

# wc=work.fit_words(words["计数"])
# plt.imshow(wc)
# plt.show()
imag=imread("D:\\bigdata\\2.5\\贾宝玉.png")
wc=w(font_path="simhei.ttf",background_color="white",
	mask=imag ) #mask是用于指定轮廓的。
plt.figure(figsize=(10,40))
# 二列表变成字典会变成多层结构的字典{“计数”：{..}}
words=plotdata.set_index("seg").to_dict()
# 数据训练的是字典型的要变成字典
wold=wc.fit_words(words["计数"])
col=ge(imag)
plt.imshow(wold.recolor(color_func=col))
print(words)
plt.show()

