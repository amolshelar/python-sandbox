# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import csv
import math
from sklearn import tree

def get_data():
    df = pd.read_csv("train.csv")
    return df

df = get_data()
#print df

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):	#Transforms columns of X specified in self.columns using LabelEncoder().     
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

output1=MultiColumnLabelEncoder(columns = ['Name','Goal Name']).fit_transform(df)
#print output1

def encode_target(df, target_column):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)
    return (df_mod, targets)

df2, targets = encode_target(df, "Will the goal be achieved")
#print df2
#print targets

features = ['Total Saving', 'Total Income per month','Total Expenses per month']

y = df2["Target"]
X = df2[features]

f1 = pd.read_csv("output.csv")

output2=MultiColumnLabelEncoder(columns = ['Name','Goal Name']).fit_transform(f1)
f1=output2
X_test=pd.DataFrame(f1, columns = ['Total Saving', 'Total Income per month','Total Expenses per month'])

  
svc = LinearSVC(C=1.0)
svc.fit(X,y)

global y_pred
y_pred=svc.predict(X_test) 
    
print y_pred    



fruit_classifier = tree.DecisionTreeClassifier()
fruit_classifier.fit(X, y)
y_pred=fruit_classifier.predict(X_test) 
print y_pred    
columns = ['Total Saving', 'Total Income per month','Total Expenses per month']
with open("fruit_classifier.txt", "w") as f:
    f = tree.export_graphviz(fruit_classifier, out_file=f, class_names= columns)




y_test=[1]
def measure_performance(X,y,clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):

    if y_pred==1:
        print("Predicted class is : ",y_pred)
    else:
        print("Predicted class is : ",y_pred)
    if show_accuracy:
        print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n")

    if show_classification_report:
        print ("Classification report")
        print (metrics.classification_report(y,y_pred),"\n")
    
    if show_confusion_matrix:
        print ("Confusion matrix")
        print (metrics.confusion_matrix(y,y_pred),"\n")
      
 