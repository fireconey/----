# 逻辑回归(分类)
# 有大小意义的直接影射为数值大小
# 没有大小意义的变成哑变量
import pandas as pd
import numpy as np
from sklearn.linear_model  import LogisticRegression as logc
data=pd.read_csv("data2.csv")
data=data.dropna()
# print(data.shape) 
# print(data) 

dum=[
    'Gender', 'Home Ownership', 
    'Internet Connection', 'Marital Status',
    'Movie Selector', 'Prerec Format', 'TV Signal'
]

data=pd.get_dummies(data,columns=dum,prefix=dum,prefix_sep=" ",drop_first=True)

edu={
    'Post-Doc': 9,
    'Doctorate': 8,
    'Master\'s Degree': 7,
    'Bachelor\'s Degree': 6,
    'Associate\'s Degree': 5,
    'Some College': 4,
    'Trade School': 3,
    'High School': 2,
    'Grade School': 1
     }

data["Education Level Map"]=data['Education Level'].map(edu)

fre={
    'Never': 0,
    'Rarely': 1,
    'Monthly': 2,
    'Weekly': 3,
    'Daily': 4
         }

data['PPV Freq Map'] = data['PPV Freq'].map(fre)
data['Theater Freq Map'] = data['Theater Freq'].map(fre)
data['TV Movie Freq Map'] = data['TV Movie Freq'].map(fre)
data['Prerec Buying Freq Map'] = data['Prerec Buying Freq'].map(fre)
data['Prerec Renting Freq Map'] = data['Prerec Renting Freq'].map(fre)
data['Prerec Viewing Freq Map'] = data['Prerec Viewing Freq'].map(fre)

col = [
    'Age', 'Num Bathrooms', 'Num Bedrooms', 'Num Cars', 'Num Children', 'Num TVs', 
    'Education Level Map', 'PPV Freq Map', 'Theater Freq Map', 'TV Movie Freq Map', 
    'Prerec Buying Freq Map', 'Prerec Renting Freq Map', 'Prerec Viewing Freq Map', 
    'Gender Male',
    'Internet Connection DSL', 'Internet Connection Dial-Up', 
    'Internet Connection IDSN', 'Internet Connection No Internet Connection',
    'Internet Connection Other', 
    'Marital Status Married', 'Marital Status Never Married', 
    'Marital Status Other', 'Marital Status Separated', 
    'Movie Selector Me', 'Movie Selector Other', 'Movie Selector Spouse/Partner', 
    'Prerec Format DVD', 'Prerec Format Laserdisk', 'Prerec Format Other', 
    'Prerec Format VHS', 'Prerec Format Video CD', 
    'TV Signal Analog antennae', 'TV Signal Cable', 
    'TV Signal Digital Satellite', 'TV Signal Don\'t watch TV'
]

# 由于col已经是list了所以已经是两个【】
inputdata=data[col]
# 一个【】表示的是一位向量，两个是表结构
outputdata=data[['Home Ownership Rent']]


model=logc()
model.fit(inputdata, outputdata)
score=model.score(inputdata,outputdata)
print(score)

result=model.predict(inputdata)
print(result) 