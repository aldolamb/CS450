# import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from operator import itemgetter

# data = pd.read_csv('iris.data.txt')

iris = datasets.load_iris()
print(iris)

data_train, data_test, targets_train, targets_test \
    = train_test_split(iris.data, iris.target, train_size=.7, test_size=.3)

classifier = GaussianNB()
model = classifier.fit(data_train, targets_train)

targets_predicted = model.predict(data_test)

print(sum(1 for x,y in zip(targets_predicted,targets_test) if x == y) / len(targets_predicted) * 100, '%')

class HardCodedClassifier():
    def fit(self, data, targets):
        return HardCodedClassifier

    def predict(data_train):
        predictions = []
        for i in data_train:
            predictions.append(0)
        # predictions.sort(predictions)
        #sorted(predictions, key=itemgetter(1))
        return predictions

classifier = HardCodedClassifier()
model = classifier.fit(data_train, targets_train)
targets_predicted = model.predict(data_test)

#print(targets_predicted)

print(sum(1 for x, y in zip(targets_predicted,targets_test) if x == y) / len(targets_predicted) * 100, '%')

#cross validation in sklearn "cross val score"

# print(data_testz)

# Show the data (the attributes of each instance)
# print(iris.data)

# Show the target values (in numeric format) of each instance
# print(iris.target)

# Show the actual target names that correspond to each number
# print(iris.target_names)
