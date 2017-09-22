import os 
import os.path
import jieba.analyse as analy
import pandas as pd

files=[]
filepath=[]
tag1=[]
tag2=[]
tag3=[]
tag4=[]
path="C:\\Users\\TH\Desktop\\练习\\sample"
for root ,dir ,files1 in os.walk(path):
	for file in files1:
		files.append(file)
		fpath=os.path.join(root,file)
		filepath.append(fpath)
		ope=open(fpath,"r+",encoding="utf-8")
		content=ope.read()
		tags=analy.extract_tags(content,topK=4)
		tag1.append(tags[0])
		tag2.append(tags[1])
		tag3.append(tags[2])
		tag4.append(tags[3])

sheet=pd.DataFrame({
	"filepath":filepath,
	"files":files,
	"tag1":tag1,
	"tag2":tag2,
	"tag3":tag3,
	"tag4":tag4
	})
print (sheet)

