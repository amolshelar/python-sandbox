# -*- coding: utf-8 -*-
#https://www.analyticsvidhya.com/blog/2015/11/improve-model-performance-cross-validation-in-python-r/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from collections import Counter
from sklearn import cross_validation
import numpy as np

train_data = pd.read_csv("train_full.csv")
print "Dataset Shape:: ", train_data.shape
#print train_data

features = ['Total Income per month','Total Expenses per month','Surplus per month','Time to Goal (in months)','Goal Value in today value terms','Amount you intend to invest per month for this goal','How much savings you have now? (Rs)','Future Value of Existing savings','Value of future investment','Future Value of Goal']
target = ['Will the goal be achieved']

X = train_data[features]
y = train_data[target]
#print X
#print y

test_data = pd.read_csv("test.csv")
X_test = test_data[features]
#print X_test

model = RandomForestClassifier(n_estimators=100)

#Simple K-Fold cross validation. 10 folds.
cv = cross_validation.KFold(len(X), n_folds=10)
print "-------------", len(X)
results = []
# "Error_function" can be replaced by the error function of your analysis
for train_index, test_index in cv:
	#print y
	#print X.ix[test_index]
        probas = model.fit(X.ix[train_index], y.ix[train_index]).predict_proba(X.ix[test_index])
        results.append( probas )
        
print results        
print "Results: " + str( np.array(results).mean() )
