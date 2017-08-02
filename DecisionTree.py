from sklearn import tree
from IPython.display import Image  
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus 

X = [[0, 0], [1, 1], [2,2]]
y = [0, 1, 2]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

print clf.predict([[2., 2.]])

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())

