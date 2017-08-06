from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV 

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
#Split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  

# Linear Kernel
svc_linear = svm.SVC(kernel='linear', C=1)
svc_linear.fit(X_train, y_train)
predicted= svc_linear.predict(X_test)
cnf_matrix = confusion_matrix(y_test, predicted)
print svc_linear
print(cnf_matrix)

parameters = {'kernel':('linear', 'rbf'), 'C':[1,2,3,4,5,6,7,8,9,10], 'gamma': 
              [0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5]}
svr = svm.SVC()
grid_search = GridSearchCV(svr, parameters)
grid_search.fit(X_train, y_train)
predicted = grid_search.predict(X_test)
print svr
print '\nBest score: %0.3f' % grid_search.best_score_
print 'Best parameters set:', grid_search.best_params_
#cnf_matrix = confusion_matrix(y_test, predicted)

#svr = svm.SVC()
#grid_search = GridSearchCV(svr, param_grid)
#grid_search.fit(X, y)
#predicted = grid_search.predict(X_test)
#print svr
#print grid_search.best_params_