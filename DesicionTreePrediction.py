import numpy as np, pandas as pd, sklearn.metrics
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
df2=df
df2["Time"]=df2["utcTimestamp"]
df2["Time"]=pd.DatetimeIndex(df2["Time"])
df2["Date"]=df2["Time"].dt.date
df2["Year"]=df2["Time"].dt.year
df2["Month"]=df2["Time"].dt.month
df2["Day"]=df2["Time"].dt.day
#Format datetime64[ns].
df2["Time Data"]=df2["Time"]

from sklearn.preprocessing import LabelEncoder
#Convert Binary ID's to integers.
var_mod = df2[["venueId","venueCategoryId","venueCategory"]]
le = LabelEncoder()
for i in var_mod:
    df2[i] = le.fit_transform(df2[i])
#Convert the coordinates from float to int.
df2["latitude"] = df2["latitude"].astype('int')
df2["longitude"] = df2["longitude"].astype('int')
df3=df2[["venueId","latitude","longitude","Year","Month","Day","venueCategoryId","venueCategory"]]

#Target Variable
target=pd.Series(df3["venueCategory"].unique(), name="venueCategory").reset_index()
target["venueCategoryId"]=df3["venueCategoryId"]
#Features
features=list(df3.columns[:6])
df3=pd.merge(df3,target)

dt=DecisionTreeClassifier()
y=df3["index"]
X=df3[features]

dt.fit(X,y)
sklearn.metrics.confusion_matrix(y,dt.predict(X),target["venueCategory"])
sklearn.metrics.confusion_matrix(y,dt.predict(X))
sklearn.metrics.accuracy_score(y,dt.predict(X))

import sklearn
import sklearn.datasets
import sklearn.model_selection

hast=sklearn.datasets.make_hastie_10_2(n_samples=12000, random_state=None)
hdt=DecisionTreeClassifier(min_samples_split=30)
hdt.fit(hast[0],hast[1])
a=sklearn.metrics.accuracy_score(hast[1],hdt.predict(hast[0]))
b=sklearn.metrics.precision_score(hast[1],hdt.predict(hast[0]))
c=sklearn.metrics.confusion_matrix(hast[1],hdt.predict(hast[0]))
d=sklearn.metrics.recall_score(hast[1],hdt.predict(hast[0]))
e=sklearn.metrics.f1_score(hast[1],hdt.predict(hast[0]))
f=sklearn.metrics.roc_curve(hast[1],hdt.predict(hast[0]))
dtree=DecisionTreeClassifier(min_samples_split=30)
scores = sklearn.model_selection.cross_val_score(dtree, X, y, cv=5)
s=np.mean(scores)
print("Accuracy is: ",a,"\n")
print("Precision is: ",b,"\n")
print("Confusion Matrix is: ",c,"\n")
print("Recall is: ",d,"\n")
print("FL is: ",e,"\n")
print("Roc Curve is: ",f,"\n")
print("Cross Validation Score is: ",scores,"\n")
print("Mean of Score is: ",s,"\n")
