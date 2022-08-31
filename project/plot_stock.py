import csv
import os


import dateutil
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

DIR = './SH50/data'

most_up = -1
most_down = 1

most_up_stock = ''
most_down_stock = ''

for filename in os.listdir(DIR):
    with open(DIR + '/' + filename) as csvfile:

        data_date = []
        data_price = []
        data_X = []
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if row[0] == 'trade_date':
                continue
            t = dateutil.parser.parse(row[0])
            data_date.append(t)
            data_price.append(float(row[4]))
            data_X.append(len(data_price))
        model = LinearRegression()
        model.fit(np.array(data_X).reshape(-1, 1), data_price)
        plt.cla()
        plt.plot(data_date, data_price)
        plt.plot()
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(filename[0:6])
        a = model.coef_[0]
        if a > 0:
            if a > most_up:
                most_up = a
                most_up_stock = filename
            # plt.savefig('./plot/up/' + filename[0:6])
        else:
            if a < most_down:
                most_down = a
                most_down_stock = filename
            # plt.savefig('./plot/down/' + filename[0:6])


print(most_up, most_down)