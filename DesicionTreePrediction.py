import numpy, pandas as pd, sklearn.metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import time

df=pd.read_csv("dataset_TSMC2014_NYC.csv")
df["Time"]=df["utcTimestamp"]
df["Time"]=pd.DatetimeIndex(df["Time"])
df["Date"]=df["Time"].dt.date
df['Year']=df['Time'].dt.year
df['Month']=df['Time'].dt.month
df['Day-of-Week']=df['Time'].dt.day
df["Time"]=df['Time'].dt.time
df['TotalNumuserId'] = df.groupby('userId')['userId'].transform('count')
#Create column = gps
df["GPS"] = df["longitude"]*df["latitude"]
df2=df[["userId","latitude","longitude","GPS","Year","Month","Day-of-Week","TotalNumuserId"]]


dt=DecisionTreeClassifier(min_samples_split=30)
features=list(df2.columns[:])
classes=pd.Series(df["userId"].unique(), name="userId").reset_index()
df3=pd.merge(df2,classes)
y=df3["index"]
X=df3[features]
dt.fit(X,y)
sklearn.metrics.confusion_matrix(y,dt.predict(X),classes["userId"])
sklearn.metrics.confusion_matrix(y,dt.predict(X))
sklearn.metrics.accuracy_score(y,dt.predict(X))

import sklearn
import sklearn.datasets
import sklearn.model_selection

hast=sklearn.datasets.make_hastie_10_2(n_samples=12000, random_state=None)
hdt=DecisionTreeClassifier(min_samples_split=30)
hdt.fit(hast[0],hast[1])
sklearn.metrics.accuracy_score(hast[1],hdt.predict(hast[0]))
sklearn.metrics.precision_score(hast[1],hdt.predict(hast[0]))
sklearn.metrics.confusion_matrix(hast[1],hdt.predict(hast[0]))
sklearn.metrics.recall_score(hast[1],hdt.predict(hast[0]))
sklearn.metrics.f1_score(hast[1],hdt.predict(hast[0]))
sklearn.metrics.roc_curve(hast[1],hdt.predict(hast[0]))
dtree=DecisionTreeClassifier(min_samples_split=30)
scores = sklearn.model_selection.cross_val_score(dtree, X, y, cv=5)
s=numpy.mean(scores)
print(s)
print(scores)
