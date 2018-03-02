import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy

data = pd.read_csv('adult.data.txt', skipinitialspace=True)

# rename columns
data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum',
                'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
                'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry',
                'pay']

# simplify datasets
data.workclass.replace({'Private': 1, 'Self-emp-not-inc': 1, 'Self-emp-inc': 1, 'Federal-gov': 1,
                        'Local-gov': 1, 'State-gov': 1,
                        'Without-pay': 0, 'Never-worked': 0, '?': 0}
                       , inplace=True)

data.maritalStatus.replace({'Married-civ-spouse': 2,
                            'Married-spouse-absent': 1, 'Married-AF-spouse': 1,
                            'Divorced': 0, 'Never-married': 0, 'Separated': 0, 'Widowed': 0}
                           , inplace=True)

data.occupation.replace({'Craft-repair': 0, 'Other-service': 0, 'Handlers-cleaners': 0, 'Transport-moving': 0,
                         'Priv-house-serv': 0, 'Sales': 1, 'Tech-support': 1, 'Machine-op-inspct': 1,
                         'Farming-fishing': 1, 'Protective-serv': 1, 'Armed-Forces': 1,
                         'Exec-managerial': 2, 'Prof-specialty': 2, 'Adm-clerical': 2, '?': 0}
                        , inplace=True)

data.relationship.replace({'Wife': 0, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 1,
                           'Other-relative': 0, 'Unmarried': 1}
                          , inplace=True)

data.race.replace({'White': 1, 'Asian-Pac-Islander': 0, 'Amer-Indian-Eskimo': 0,
                   'Other': 0, 'Black': 0}
                  , inplace=True)

data.sex.replace({'Female': 1, 'Male': 0}, inplace=True)

data.nativeCountry.replace({'United-States': 1,
                            'Cambodia': 0, 'England': 0, 'Puerto-Rico': 0, 'Canada': 0, 'Germany': 0,
                            'Outlying-US(Guam-USVI-etc)': 0, 'India': 0, 'Japan': 0, 'Greece': 0, 'South': 0,
                            'China': 0, 'Cuba': 0, 'Iran': 0, 'Honduras': 0, 'Philippines': 0, 'Italy': 0,
                            'Poland': 0, 'Jamaica': 0, 'Vietnam': 0, 'Mexico': 0, 'Portugal': 0, 'Ireland': 0,
                            'France': 0, 'Dominican-Republic': 0, 'Laos': 0, 'Ecuador': 0, 'Taiwan': 0,
                            'Haiti': 0, 'Columbia': 0, 'Hungary': 0, 'Guatemala': 0, 'Nicaragua': 0, 'Scotland': 0,
                            'Thailand': 0, 'Yugoslavia': 0, 'El-Salvador': 0, 'Trinadad&Tobago': 0, 'Peru': 0,
                            'Hong': 0, 'Holand-Netherlands': 0, '?': 0}
                           , inplace=True)

data.pay.replace({'>50K': 1, '<=50K': 0}, inplace=True)

del data['education']

target = data.pay
del data['pay']

data = data.as_matrix()
target = target.as_matrix()
# target = np.split(data, [14])

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
    # iris = datasets.load_iris()
    # train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target)
    train_size = 0.001
    train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=train_size, test_size=1-train_size)
    print(train_target)
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
    i = (100 * i) / test_target.shape[0]
    print("Accuracy of sklearn's algorithm: " + str(i) + "%")


if __name__ == "__main__":
    main()