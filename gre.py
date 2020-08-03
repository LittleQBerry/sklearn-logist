import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
#导入逻辑回归进行建模分类
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

datas =pd.read_csv('test2.csv')
col_1 =datas['label'].values
labels=np.array(col_1).reshape(1, -1)
col_2=datas['data'].values
data=np.array(col_2).reshape(1, -1)
print(data)
        #mask=i[0]
        #img=i[1]
        #label.append(mask)
        #data.append(img)
        #count=0
#对数据集进行分组，80%的图片作为训练集，20%的数据作为测试集
X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.1,random_state=1)
logreg = LogisticRegression(random_state=1).fit(X_train,y_train)
print(logreg)
#打印模型在训练数据集上的准确率
print("accuracy:",accuracy_score(logreg.predict(X_train),y_train))