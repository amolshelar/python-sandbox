# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from collections import Counter
import optunity
import optunity.metrics


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


def train_svm(data, labels, kernel, C, gamma, degree, coef0):
    """A generic SVM training function, with arguments based on the chosen kernel."""
    if kernel == 'linear':
        model = SVC(kernel=kernel, C=C)
    elif kernel == 'poly':
        model = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)
    elif kernel == 'rbf':
        model = SVC(kernel=kernel, C=C, gamma=gamma)
    else:
        raise ArgumentError("Unknown kernel function: %s" % kernel)
    model.fit(data, labels)
    return model

search = {'algorithm': {'k-nn': {'n_neighbors': [1, 5]},
                        'SVM': {'kernel': {'linear': {'C': [0, 2]},
                                           'rbf': {'gamma': [0, 1], 'C': [0, 10]},
                                           'poly': {'degree': [2, 5], 'C': [0, 50], 'coef0': [0, 1]}
                                           }
                                },
                        'naive-bayes': None,
                        'random-forest': {'n_estimators': [10, 30],
                                          'max_features': [5, 20]}
                        }
         }


@optunity.cross_validated(x=X, y=y, num_folds=5)
def performance(x_train, y_train, x_test, y_test,
                algorithm, n_neighbors=None, n_estimators=None, max_features=None,
                kernel=None, C=None, gamma=None, degree=None, coef0=None):
    # fit the model
    if algorithm == 'k-nn':
        model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
        model.fit(x_train, y_train)
    elif algorithm == 'SVM':
        model = train_svm(x_train, y_train, kernel, C, gamma, degree, coef0)
    elif algorithm == 'naive-bayes':
        model = GaussianNB()
        model.fit(x_train, y_train)
    elif algorithm == 'random-forest':
        model = RandomForestClassifier(n_estimators=int(n_estimators),
                                       max_features=int(max_features))
        model.fit(x_train, y_train)
    else:
        raise ArgumentError('Unknown algorithm: %s' % algorithm)

    # predict the test set
    if algorithm == 'SVM':
        predictions = model.decision_function(x_test)
    else:
        predictions = model.predict_proba(x_test)[:, 1]

    return optunity.metrics.roc_auc(y_test, predictions, positive=True)

performance(algorithm='SVM')

#optimal_configuration, info, _ = optunity.maximize_structured(performance,
#                                                              search_space=search,
#                                                              num_evals=300)
print(optimal_configuration)
print(info.optimum)

solution = dict([(k, v) for k, v in optimal_configuration.items() if v is not None])
print('Solution\n========')
print("\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items())))

