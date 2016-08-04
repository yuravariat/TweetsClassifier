import numpy as np
from sklearn import cross_validation, metrics
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()

# Regular, with split data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)


# cross validation
clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)

predicted = cross_validation.cross_val_predict(clf, iris.data, iris.target, cv=10)
metrics = metrics.accuracy_score(iris.target, predicted)


from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train',categories=['alt.atheism', 'talk.religion.misc'])

scores = cross_validation.cross_val_score(clf, newsgroups_train.data, newsgroups_train.target, cv=5)


test=1