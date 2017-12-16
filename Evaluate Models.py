#Ruslan Shakirov
#Foursquare - Modelling
#https://github.com/ruslanski/Data-Analysis-Foursquare

import numpy as np, pandas as pd, sklearn.metrics
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
import pygeohash as pgh
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import os
import time
#Interactive mode for automatic plotting in Python idle
from matplotlib import interactive
interactive(True)

#Load dataset
df=pd.read_csv("dataset_TSMC2014_NYC.csv")
df2=df
def data_munging():
      """
      Function splits timestamp in to multiple columns, geohashes coordinates,
      encodes non-numeric values to integers, creates target variables and features. 
      """
      #Convert string to datetime64[ns]
      df2["Times"]=pd.to_datetime(df2["utcTimestamp"])
      df2["Year"]=df2["Times"].dt.year
      df2["Month"]=df2["Times"].dt.month
      df2['Weekday'] = df2['Times'].dt.dayofweek
      df2["Time"]=df2["Times"].dt.time
      #Combine latitude and longitude into tuple(object)
      df2['Lat,Long'] = df2[['latitude', 'longitude']].apply(tuple, axis=1)
      #New column
      df3['geohash'] = ""
      #Geohash latitude and longitude coordinates with precision=5
      for index,row in df3.iterrows():
            value = pgh.encode(row['latitude'],row['longitude'], precision=5)
            #Set values as geohash(object)
            #at: get scalar values. Fast alternative of .loc
            df3.at[index,"geohash"]=value

      df4=df3[["venueCategory","venueId","Year","Month","Weekday","Hour","Minute","Geohash"]]
      #Encode string values to integers
      var_mod = df4[["venueCategory","venueId","Geohash"]]
      # LabelEncoder
      le = LabelEncoder()
      # apply "le.fit_transform"
      for i in var_mod:
          df4[i] = le.fit_transform(df4[i])
      #Target Variable
      target=pd.Series(df4["venueCategory"].unique(), name="venueCategory").reset_index()
      #Merge target values with dataframe
      df4=pd.merge(df4,target)
def evaluate_models():
      """
      Function evaluates four algorithms, creates validation dataset,
      test harness, estimates accuracy and shows results of each model.
      """
      #Make list of features ['venueId', 'Year', 'Month', 'Weekday', 'Hour', 'Minute', 'Geohash']
      features=list(df4.columns[1:8])
      Y=df4["venueCategory"]
      X=df4[features]
      #View total number of zones in city.(Geohashed zones)
      zones=pd.Series(df4["Geohash"].unique(),name="Geohash").reset_index()
      validation_size = 0.20
      seed = 3
      X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
      seed = 3
      #Test options and evaluation metric
      scoring = 'accuracy'
      #Spot Check Algorithms
      models = []
      models.append(('Logistic Regression', LogisticRegression()))
      models.append(('K-Nearest Neighbors', KNeighborsClassifier()))
      models.append(('Decision Tree Class', DecisionTreeClassifier()))
      models.append(('Gaussian Naive Byer', GaussianNB()))
      #Evaluate each model in turn
      results = []
      names = []
      for name, model in models:
          kfold = model_selection.KFold(n_splits=5, random_state=seed)
          cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
          results.append(cv_results)
          names.append(name)
          msg = "%s:\n Mean: %f (STD: %f)\n Execution Time: %r\n" % (name, cv_results.mean(), cv_results.std())
          print(msg)
      # Make predictions on validation dataset of Decision Tree
      dt = DecisionTreeClassifier()
      dt.fit(X_train, Y_train)
      predictions = dt.predict(X_validation)
      print(accuracy_score(Y_validation, predictions))
      print(confusion_matrix(Y_validation, predictions))
      print(classification_report(Y_validation, predictions))

     #Make prediction on validation dataset of K-nearest neightbour
      knn = KNeighborsClassifier()
      knn.fit(X_train, Y_train)
      predictions = knn.predict(X_validation)
      print(accuracy_score(Y_validation, predictions))
      print(confusion_matrix(Y_validation, predictions))
      print(classification_report(Y_validation, predictions))

data_munging()
evaluate_models()
