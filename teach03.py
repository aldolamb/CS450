import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy

data = pd.read_csv('adult.data.txt', skipinitialspace=True)

data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum',
                'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
                'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry',
                'pay']

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

data = data.as_matrix()

target = np.split(data, 8)

class KNNClassifier:
    def __init__(self):
        pass

    def fit(self, data, target):
        return KNNModel(data, target)


class KNNModel:
    def __init__(self, data, target):
        self.data = data
        self.target = target


    def predict(self, k, data):
        closest = []
        for row in data:
            distances = []
            for srow in self.data:
                distance = 0
                for j in range(0,srow.shape[0]):
                    distance += (row[j]-srow[j])**2
                distances.append(distance)
            index = np.argsort(distances,axis=0)
            classes = np.unique(self.target[index[:k]])
            if len(classes)==1:
                closest.append(int(classes[0]))
            else:
                countfreqclasses = np.zeros(max(classes)+1)
                for j in range(k):
                    countfreqclasses[self.target[index[j]]] += 1
                closest.append(np.argmax(countfreqclasses))
        return closest



def main():
    iris = datasets.load_iris()
    train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target)
    clf = KNNClassifier()
    model = clf.fit(train_data, train_target)
    targetspredicted = model.predict(4, test_data)
    i = 0
    classifier = KNeighborsClassifier(n_neighbors=4)
    model = classifier.fit(train_data, train_target)
    predictions = model.predict(test_data)

    for test in zip(targetspredicted, test_target):
        k, j = test
        if k == j:
            i += 1
    i = (100 * i) / test_target.shape[0]
    print(str(i) + "% accuracy (mine)")
    i = 0
    for test in zip(predictions, test_target):
        k, j = test
        if k == j:
            i += 1
    i = (100*i)/test_target.shape[0]
    print(str(i) + "% accuracy (theirs)")



if __name__ == "__main__":
    main()
