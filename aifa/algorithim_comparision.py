#http://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# prepare configuration for cross validation test harness
seed = 50

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


# prepare models
models = []
#models.append(('LogisticRegression', LogisticRegression()))
#models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
#models.append(('KNeighborsClassifier', KNeighborsClassifier()))
#models.append(('KNeighborsClassifier', DecisionTreeClassifier()))
#models.append(('RandomForestClassifier', RandomForestClassifier()))
#models.append(('GaussianNB', GaussianNB()))
models.append(('SVC', SVC()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	#print "Model", model 
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "========%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()