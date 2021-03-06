# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from collections import Counter
from sklearn.model_selection import GridSearchCV 


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


def encode_target(df, target_column):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)
    return (df_mod, targets)



train_data = pd.read_csv("train_full.csv")
print "Dataset Shape:: ", train_data.shape
#print train_data

features = ['Total Income per month','Total Expenses per month','Surplus per month','Goal Value in today value terms','Future Value of Goal']
target = ['Will the goal be achieved']

train_data_encoded, targets = encode_target(train_data, 'Will the goal be achieved')
#print train_data_encoded
#print targets

X = train_data_encoded[features]
y = train_data_encoded["Target"]
#print X
#print y
print Counter(y)

test_data = pd.read_csv("test.csv")
X_test = test_data[features]
#print X_test

clf = svm.SVC(C=1.0)
clf.fit(X, y)
y_pred = clf.predict(X_test) 

#print clf
print clf
print "=====Prediction===== \n", y_pred  


#Split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  

#param_grid = {'kernel':('linear', 'rbf'), 'C':[1], 'gamma': 
#              [0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5]}

#param_grid = [
#  {'C': [1], 'kernel': ['linear']},
#  {'C': [1], 'gamma': [0.001], 'kernel': ['rbf']},
# ]

param_grid = [
  {'C': [1], 'kernel': ['linear']}
  
 ]

svr = svm.SVC()
grid_search = GridSearchCV(svr, param_grid)
grid_search.fit(X, y)
predicted = grid_search.predict(X_test)
print '\n', svr
print '\nBest score: %0.3f' % grid_search.best_score_
print 'Best parameters set:', grid_search.best_params_
#cnf_matrix = confusion_matrix(y_test, predicted)




#split data into training and test set
#The parameter test_size is given value 0.3; it means test sets will be 30% of whole dataset  & training datasetís size will be 70% of the entire dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

clf = svm.SVC(C=1.0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print ("=====Accuracy Score=====")
print metrics.accuracy_score(y_test, y_pred)*100
	
print ("=====Classification Report=====")
print metrics.classification_report(y_test, y_pred),"\n"
	    
print ("=====Confusion Matrix=====")
print metrics.confusion_matrix(y_test, y_pred),"\n"
