##import dataset
from sklearn import datasets
iris = datasets.load_iris()

## f(x) = y
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)


# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()
## the underlying 2 lines do the same as the 2 above
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
# print("predictions: %s" % predictions)

from sklearn.metrics import accuracy_score
print("accuracy: %s" % accuracy_score(y_test, predictions))