import csv
import os
from collections import defaultdict

import dateutil
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

DIR = './SH50/data'

dataset = []
stock_name = []

date_map = defaultdict(int)

for filename in os.listdir(DIR):
    with open(DIR + '/' + filename) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if row[0] == 'trade_date':
                continue
            t = dateutil.parser.parse(row[0])
            date_map[t] += 1

for filename in sorted(os.listdir(DIR)):
    with open(DIR + '/' + filename) as csvfile:

        data_date = []
        data_price = []
        data_X = []
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if row[0] == 'trade_date':
                continue
            t = dateutil.parser.parse(row[0])
            if date_map[t] != 49:
                continue
            data_date.append(t)
            data_price.append(float(row[4]))
            data_X.append(len(data_price))

        dataset.append(data_price)
        stock_name.append(filename[0:6])

d = pd.DataFrame(data=np.array(dataset).T,
                 columns=stock_name)

corr = d.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 10, as_cmap=True)

car = np.array(corr)

maxi = -1.0
mini = 1.0
for i in range(len(car)):
    for j in range(len(car[i])):
        if car[i][j] > maxi and car[i][j] != 1.0:
            maxi = car[i][j]
            print('$max', i, j)
        if car[i][j] < mini:
            mini = car[i][j]
            print('^min', i, j)

# f, ax = plt.subplots(figsize=(12, 10))

# plt.title('Correlation between stocks\' closing prices')
# sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, cbar_kws={"shrink": .5})
# plt.savefig('Corr')
