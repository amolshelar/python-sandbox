# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn import metrics 

train_data = pd.read_csv("train.csv")
print "Dataset Shape:: ", train_data.shape
#print train_data

features = ['Total Income per month','Total Expenses per month','Surplus per month','Goal Value in today value terms','Future Value of Goal']
target = ['Will the goal be achieved']

X = train_data[features]
y = train_data[target]
#print X
#print y

test_data = pd.read_csv("test.csv")
X_test = test_data[features]
#print X_test

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)
y_pred = clf.predict(X_test) 

#print clf
print "=====Prediction===== \n", y_pred


with open("DecisionTree-Graph.txt", "w") as f:
    f = export_graphviz(clf, out_file=f, feature_names=features, class_names=['No','Yes'], node_ids=True, filled=True, rounded=True)



#split data into training and test set
#The parameter test_size is given value 0.3; it means test sets will be 30% of whole dataset  & training dataset’s size will be 70% of the entire dataset.
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)


print ("=====Accuracy Score=====")
print metrics.accuracy_score(y_test, y_pred)*100
	
print ("=====Classification Report=====")
print metrics.classification_report(y_test, y_pred),"\n"
	    
print ("=====Confusion Matrix=====")
print metrics.confusion_matrix(y_test, y_pred),"\n"


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
y_pred_en = clf_entropy.predict(X_test)
#metrics_fa (y_test, y_pred)






