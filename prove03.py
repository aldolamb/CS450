import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#
# Car Evaluation
#
data = pd.read_csv('car.data.txt', skipinitialspace=True)

data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety',
                'evaluation']
data.buying.replace({'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}
                       , inplace=True)
data.maint.replace({'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}
                       , inplace=True)
data.doors.replace({'2': 2, '3': 3, '4': 4, '5more': 5}, inplace=True)
data.persons.replace({'2': 2, '4': 4, 'more': 6}, inplace=True)
data.lug_boot.replace({'small': 1, 'med': 2, 'big': 3}, inplace=True)
data.safety.replace({'low': 1, 'med': 2, 'high': 3}, inplace=True)
data.evaluation.replace({'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}, inplace=True)

target = data.evaluation
del data['evaluation']

cars_data = data.as_matrix()
cars_target = target.as_matrix()

#
# Prima Indian Diabetes
#
data = pd.read_csv('pima-indians-diabetes.data.txt', skipinitialspace=True)

data.columns = ['timesPregnant', 'plasmaGlucose', 'bloodPressure', 'skinThickness', 'serumInsulin', 'BMI',
                'pedigree', 'age', 'variable']
data.plasmaGlucose.replace({0: 123}, inplace=True)
data.bloodPressure.replace({0: 72}, inplace=True)
data.skinThickness.replace({0: 29}, inplace=True)
data.serumInsulin.replace({0: 156}, inplace=True)
data.BMI.replace({0: 32.4}, inplace=True)

target = data.variable
del data['variable']

diabetes_data = data.as_matrix()
diabetes_target = target.as_matrix()

#
# Automobile MPG
#
data = pd.read_csv('auto-mpg.data.txt', skipinitialspace=True, delim_whitespace=True)

data.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
                'model', 'origin', 'car_name']
data.displacement = data.displacement
data.horsepower.replace({'?': 104}, inplace=True)
data.horsepower = pd.to_numeric(data.horsepower)
data.weight = data.weight
data.model = data['model']

# data.mpg[(data.mpg <= 12)] = 0
# data.mpg[(data.mpg > 12) & (data.mpg <= 16)] = 1
# data.mpg[(data.mpg > 16) & (data.mpg <= 20)] = 2
# data.mpg[(data.mpg > 20) & (data.mpg <= 22)] = 3
# data.mpg[(data.mpg > 22) & (data.mpg <= 26)] = 4
# data.mpg[(data.mpg > 26) & (data.mpg <= 30)] = 5
# data.mpg[(data.mpg > 30) & (data.mpg <= 32)] = 6
# data.mpg[(data.mpg > 32) & (data.mpg <= 36)] = 7
# data.mpg[(data.mpg > 36) & (data.mpg <= 40)] = 8
# data.mpg[(data.mpg > 40) & (data.mpg <= 42)] = 9
# data.mpg[(data.mpg > 42)] = 10

del data['car_name']

data = data.astype(int)

target = data.mpg
del data['mpg']

mpg_data = data.as_matrix()
mpg_target = target.as_matrix()

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
                for j in range(0, srow.shape[0]):
                    distance += (row[j] - srow[j]) ** 2
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
                count_freq_classes = np.zeros(max(classes) + 1)
                for j in range(k):
                    count_freq_classes[self.target[index[j]]] += 1
                closest.append(np.argmax(count_freq_classes))
        return closest


def testData(data, target, trainSize):
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=(1-trainSize),
                                                                        train_size=trainSize)
    # Calculate KNN using my algorithm
    clf = KNNClassifier()
    model = clf.fit(train_data, train_target)
    my_predictions = model.predict(8, test_data)

    # Calculate KNN of sklearn's algorithm
    classifier = KNeighborsClassifier(n_neighbors=8)
    model = classifier.fit(train_data, train_target)
    sklearn_predictions = model.predict(test_data)

    # accuracy of my algorithm
    i = 0
    distances = 0
    for test in zip(my_predictions, test_target):
        k, j = test
        if k == j:
            i += 1
            distances += 0
        else:
            distances += abs(j - k)
    i = (100 * i) / test_target.shape[0]
    print("Accuracy of my algorithm: " + str(i) + "%")
    print("Average misplacement: ", (distances/my_predictions.__len__()))

    # accuracy of sklearn's algorithm
    i = 0
    for test in zip(sklearn_predictions, test_target):
        k, j = test
        if k == j:
            i += 1
    i = (100 * i) / test_target.shape[0]
    print("Accuracy of sklearn's algorithm: " + str(i) + "%")


def main():
    print("\nCars Evaluation: ")
    testData(cars_data, cars_target, 0.1)

    print("\nPrima Indian Diabetes: ")
    testData(diabetes_data, diabetes_target, 0.3)

    print("\nAuto MPG: ")
    testData(mpg_data, mpg_target, 0.3)


if __name__ == "__main__":
    main()
