from sklearn.preprocessing import LabelEncoder

X = ["pune", "delhi", "bombay", "banglore" ,"pune"] 	#Train data

print "Train Data: ", X
le = LabelEncoder()
le.fit(X)
print "Classes: ", le.classes_
print "Transform: ", le.transform(X)
print "Fit and Transform: ", le.fit_transform(X)


y = ["pune"] 	#Test Data
print "Test Teansform:", y, le.transform(y)