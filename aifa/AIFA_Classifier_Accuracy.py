# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from collections import Counter

train_data = pd.read_csv("train-full.csv")
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

#**********************************************************************
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)
y_pred = clf.predict(X_test) 

#print clf
print "=====Prediction===== \n", y_pred

#with open("DecisionTree-Graph.txt", "w") as f:
#    f = export_graphviz(clf, out_file=f, feature_names=features, class_names=['No','Yes'], node_ids=True, filled=True, rounded=True)

#**********************************************************************
clf = RandomForestClassifier(criterion='entropy')
clf.fit(X, y)
y_pred = clf.predict(X_test) 

#print clf
print "=====Prediction===== \n", y_pred

#**********************************************************************
clf = LinearSVC(C=1.0)
clf.fit(X, y)
y_pred = clf.predict(X_test) 

#print clf
print "=====Prediction===== \n", y_pred  





def measure_metrics (clf_desc, clf, X_train, y_train, X_test):
	print "*************************************", clf_desc ,"*************************************"
	
	#print ("=====Predection=====")
	#print y_test
	#print y_pred

	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	print "=====Accuracy Score ", "{0:.2f}".format(metrics.accuracy_score(y_test, y_pred)*100), "%"

	print ("=====Classification Report")
	print metrics.classification_report(y_test, y_pred)

	print ("=====Confusion Matrix")
	print metrics.confusion_matrix(y_test, y_pred)


#split data into training and test set
#The parameter test_size is given value 0.3; it means test sets will be 30% of whole dataset  & training dataset’s size will be 70% of the entire dataset.
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 100)
print "Label count: ", Counter(y_train)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
measure_metrics ("DecisionTreeClassifier-gini", clf_gini, X_train, y_train, X_test)

clf_rfc = RandomForestClassifier(n_estimators=100)
measure_metrics ("RandomForestClassifier", clf_rfc, X_train, y_train, X_test)

clf_lr = LogisticRegression()
measure_metrics ("LogisticRegression", clf_lr, X_train, y_train, X_test)

clf_linear_svc = LinearSVC(C=1.0)
measure_metrics ("LinearSVC", clf_linear_svc, X_train, y_train, X_test)


