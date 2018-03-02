from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNNClassifier:
    def __init__(self):
        pass

    def fit(self, data, target):
        return KNNModel(data, target)


class KNNModel:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    # Predicts target data based on test data
    def predict(self, k, data):
        closest = []
        for row in data:
            # Display given row
            # print(row)

            # Calculates 'distance' to determine 'closest neighbors'
            distances = []
            for srow in self.data:
                distance = 0
                # Euclidean distance
                for j in range(0,srow.shape[0]):
                    distance += (row[j]-srow[j])**2
                distances.append(distance)

            # Sorts distances
            index = np.argsort(distances, axis=0)

            # Finds all unique instances of k closest neighbors
            classes = np.unique(self.target[index[:k]])

            # If only one class, return said class.
            # Otherwise count which class has the most instances.
            if len(classes) == 1:
                closest.append(int(classes[0]))
            else:
                count_freq_classes = np.zeros(max(classes)+1)
                for j in range(k):
                    count_freq_classes[self.target[index[j]]] += 1
                closest.append(np.argmax(count_freq_classes))
        return closest


def main():
    # Load iris data set
    iris = datasets.load_iris()

    # split data into Training and Testing
    train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target)

    # Calculate KNN using my algorithm
    clf = KNNClassifier()
    model = clf.fit(train_data, train_target)
    my_predictions = model.predict(4, test_data)

    # Calculate KNN of sklearn's algorithm
    classifier = KNeighborsClassifier(n_neighbors=4)
    model = classifier.fit(train_data, train_target)
    sklearn_predictions = model.predict(test_data)

    # accuracy of my algorithm
    i = 0
    for test in zip(my_predictions, test_target):
        k, j = test
        if k == j:
            i += 1
    i = (100 * i) / test_target.shape[0]
    print("Accuracy of my algorithm: " + str(i) + "%")

    # accuracy of sklearn's algorithm
    i = 0
    for test in zip(sklearn_predictions, test_target):
        k, j = test
        if k == j:
            i += 1
    i = (100*i)/test_target.shape[0]
    print("Accuracy of sklearn's algorithm: " + str(i) + "%")


if __name__ == "__main__":
    main()
