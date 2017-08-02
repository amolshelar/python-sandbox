#http://dataaspirant.com/2017/04/21/visualize-decision-tree-python-graphviz/
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import metrics

# creating dataset for modeling Apple / Orange classification
fruit_data_set = pd.DataFrame()
fruit_data_set["fruit"] = np.array([1, 2, 1, 1, 1,      # 1 for apple
                                    0, 0, 0, 0, 0])     # 0 for orange
fruit_data_set["weight"] = np.array([170, 175, 180, 178, 182,
                                     130, 120, 130, 138, 145])
fruit_data_set["smooth"] = np.array([9, 10, 8, 8, 7,
                                     3, 4, 2, 5, 6])
                                     
fruit_data_set["amol"] = np.array([9, 10, 8, 8, 7,
                                     3, 4, 2, 5, 6])

#Bilding Decision Tree ClassifierPython
X = fruit_data_set[["weight", "smooth", "amol"]]
y = fruit_data_set["fruit"]
fruit_classifier = tree.DecisionTreeClassifier()
fruit_classifier.fit(X, y)

print (fruit_classifier)



with open("fruit_classifier.txt", "w") as f:
    f = tree.export_graphviz(fruit_classifier, out_file=f)
    
# fruit data set 1st observation
test_features_1 = [[fruit_data_set["weight"][0], fruit_data_set["smooth"][0], fruit_data_set["amol"][0]]]
test_features_1_fruit = fruit_classifier.predict(test_features_1)
print "Actual fruit type: {act_fruit} , Fruit classifier predicted: {predicted_fruit}".format(
    act_fruit=fruit_data_set["fruit"][0], predicted_fruit=test_features_1_fruit)
 
y_pred = test_features_1_fruit 
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

#measure_performance(X_test,y_test,svc,show_classification_report=True, show_confusion_matrix=True)           
measure_performance(X,test_features_1_fruit,fruit_classifier,show_classification_report=True, show_confusion_matrix=True)     