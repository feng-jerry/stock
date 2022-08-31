import csv
import os

import dateutil.parser
import numpy as np
from joblib import dump
from sklearn import metrics
from sklearn.linear_model import RidgeClassifier

WINDOW_SIZE = 5
FORCAST_SIZE = 10

X_train = []
X_val = []
X_test = []
y_train = []
y_val = []
y_test = []

DIR = './SH50/data'

MODE = '_' + str(WINDOW_SIZE) + '_' + str(FORCAST_SIZE)

for filename in os.listdir(DIR):

    with open(DIR + '/' + filename) as csvfile:
        data_date = []
        data_price = []

        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if row[0] == 'trade_date':
                continue

            t = dateutil.parser.parse(row[0])
            data_date.append(t)
            data_price.append(float(row[4]))

        for i in range(len(data_date) - WINDOW_SIZE - FORCAST_SIZE):
            today = i + WINDOW_SIZE - 1
            if data_price[today + FORCAST_SIZE] > data_price[today]:
                y_append = 1
            else:
                y_append = 0

            max_price = max(data_price[i: i + WINDOW_SIZE])
            min_price = min(data_price[i: i + WINDOW_SIZE])
            X_append = []
            if max_price == min_price:
                X_append = [0.5] * WINDOW_SIZE
            else:
                for j in data_price[i: i + WINDOW_SIZE]:
                    X_append.append((j - min_price) / (max_price - min_price) * 2 - 1)

            if data_date[today].year <= 2018 or (data_date[today].year == 2019 and data_date[today].month <= 3):
                X_train.append(X_append)
                y_train.append(y_append)
            elif data_date[today].year == 2019 and data_date[today].month >= 4:
                X_val.append(X_append)
                y_val.append(y_append)
            else:
                X_test.append(X_append)
                y_test.append(y_append)

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

print(np.amax(X_train))
print(np.amin(X_train))

clf = RidgeClassifier()
clf.fit(X_train, y_train)

dump(clf, 'ridge' + MODE + '.joblib')

print('balanced_accuracy_score', metrics.balanced_accuracy_score(y_val, clf.predict(X_val)))