from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import math

iris = datasets.load_iris()
data = pd.DataFrame(iris.data)

data.columns = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']


def rel_max(col):
    temp = col.sort_values()
    mid = col.mean()

    first_half, second_half = np.split(temp, np.bincount(np.digitize(temp, [mid])).cumsum())[:-1]
    first_mid = first_half.mean()
    second_mid = second_half.mean()

    first_half1, second_half1 = np.split(first_half, np.bincount(np.digitize(first_half, [first_mid])).cumsum())[:-1]
    first_half2, second_half2 = np.split(second_half, np.bincount(np.digitize(second_half, [second_mid])).cumsum())[:-1]

    print(first_half1.size, ', ', second_half1.size, ', ', first_half2.size, ', ', second_half2.size)

    return first_mid, mid, second_mid


mid1, mid2, mid3 = rel_max(data.sepalLength)

data.sepalLength[(data.sepalLength <= mid1)] = 0
data.sepalLength[(data.sepalLength > mid1) & (data.sepalLength <= mid2)] = 1
data.sepalLength[(data.sepalLength > mid2) & (data.sepalLength <= mid3)] = 2
data.sepalLength[(data.sepalLength > mid3)] = 3

mid1, mid2, mid3 = rel_max(data.sepalWidth)

data.sepalWidth[(data.sepalWidth <= mid1)] = 0
data.sepalWidth[(data.sepalWidth > mid1) & (data.sepalWidth <= mid2)] = 1
data.sepalWidth[(data.sepalWidth > mid2) & (data.sepalWidth <= mid3)] = 2
data.sepalWidth[(data.sepalWidth > mid3)] = 3

mid1, mid2, mid3 = rel_max(data.petalLength)

data.petalLength[(data.petalLength <= mid1)] = 0
data.petalLength[(data.petalLength > mid1) & (data.petalLength <= mid2)] = 1
data.petalLength[(data.petalLength > mid2) & (data.petalLength <= mid3)] = 2
data.petalLength[(data.petalLength > mid3)] = 3

mid1, mid2, mid3 = rel_max(data.petalWidth)

data.petalWidth[(data.petalWidth <= mid1)] = 0
data.petalWidth[(data.petalWidth > mid1) & (data.petalWidth <= mid2)] = 1
data.petalWidth[(data.petalWidth > mid3)] = 3
data.petalWidth[(data.petalWidth > mid2) & (data.petalWidth <= mid3)] = 2

data = data.astype(int)

# print(data)
# for i in data.as_matrix().T:
#     print(i)


class Tree(object):
    def __init__(self):
        self.left = None;
        self.right = None;
        self.data = None;


def calc_entropy(p):
    entropies = {}
    for i in p.T:
        total = i.shape[0]
        num0 = np.count_nonzero(i == 0)/total
        num1 = np.count_nonzero(i == 1)/total
        num2 = np.count_nonzero(i == 2)/total
        num3 = np.count_nonzero(i == 3)/total
        entropy = (-num0 * math.log(num0, 4) - num1 * math.log(num1, 4) -
                   num2 * math.log(num2, 4) - num3 * math.log(num3, 4))

        print(total, num0, num1, num2, num3, '\n', entropy, '\n', i)

        entropies[entropy] = i;

        # print(i)
        # return -p * np.log2(p)
    least = sorted(entropies, reverse=True).pop()
    print(entropies.get(least))

    print(entropies)


def main():
    calc_entropy(data.as_matrix())


if __name__ == "__main__":
    main()