import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle


    
data=pd.read_csv("C://Users//mrudu//Downloads//breastCancer.csv")
df=data.replace('?',np.nan) 
df=df.fillna(df.median())
df['bare_nucleoli']=df['bare_nucleoli'].astype('int64')
df.drop('id',axis=1,inplace=True)

X=df.drop('class',axis=1)
y=df['class']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30,random_state=1)


svc=SVC()
svc.fit(X_train,y_train)

pickle.dump(svc,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))




