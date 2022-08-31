import csv
import os
import pickle
from collections import defaultdict
from datetime import datetime

import dateutil.parser
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

for WINDOW_SIZE in [5, 10, 20, 30, 60]:
# for WINDOW_SIZE in [5]:
    for FORCAST_SIZE in [5, 10]:
    # for FORCAST_SIZE in [5]:
        if FORCAST_SIZE == 10 and WINDOW_SIZE in [5, 10]:
            continue
        begin = datetime.now()
        MODE = '_' + str(WINDOW_SIZE) + '_' + str(FORCAST_SIZE)

        Method = ['RANDOM_WALK', 'MOVING_AVERAGE', 'RIDGE', 'LGBM', 'TRANSFORMER']
        # Method = ['RANDOM_WALK', 'MOVING_AVERAGE', 'RIDGE', 'TRANSFORMER']

        DIR = './SH50/data'

        with open('./model/transformer' + MODE + '.pickle', 'rb') as pickle_file:
            transformer_pickle = pickle.load(pickle_file)
        clf = joblib.load('ridge' + MODE + '.joblib')
        clf_lgbm = joblib.load('lgbm' + MODE + '.joblib')

        agg_money = defaultdict(float)
        agg_money_by_date = defaultdict(lambda: defaultdict(float))
        rank = defaultdict(list)
        date_set = set()
        for filename in os.listdir(DIR):
            with open(DIR + '/' + filename) as csvfile:
                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    if row[0] == 'trade_date':
                        continue
                    t = dateutil.parser.parse(row[0])
                    date_set.add(t)

        date_list = sorted(list(date_set))
        date_to_order = defaultdict(int)
        for i in range(len(date_list)):
            date_to_order[date_list[i]] = i

        for filename in sorted(os.listdir(DIR)):

            with open(DIR + '/' + filename) as csvfile:

                data_price = []
                data_date = []
                money = defaultdict(lambda: [100 / FORCAST_SIZE] * FORCAST_SIZE)
                stock = defaultdict(lambda: [0] * FORCAST_SIZE)
                index = 0  # index for money and stock, 0 ~ (FORCAST_SIZE-1)
                total_by_day = defaultdict(list)
                date_to_price = defaultdict(float)

                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    if row[0] == 'trade_date':
                        continue
                    t = dateutil.parser.parse(row[0])
                    data_date.append(t)
                    data_price.append(float(row[4]))
                    date_to_price[t] = float(row[4])

                X_ridge = []
                for i in range(len(data_price) - WINDOW_SIZE - FORCAST_SIZE):
                    today = i + WINDOW_SIZE - 1

                    max_price = max(data_price[i: i + WINDOW_SIZE])
                    min_price = min(data_price[i: i + WINDOW_SIZE])
                    X_ridge_append = []
                    if max_price == min_price:
                        X_ridge_append = [0.5] * WINDOW_SIZE
                    else:
                        for j in data_price[i: i + WINDOW_SIZE]:
                            X_ridge_append.append((j - min_price) / (max_price - min_price) * 2 - 1)
                    X_ridge.append(X_ridge_append)

                X_ridge = np.array(X_ridge)
                prediction_ridge = clf.predict(X_ridge)
                prediction_lgbm = clf_lgbm.predict(X_ridge)

                X_date = []

                for i in range(len(data_price) - WINDOW_SIZE - FORCAST_SIZE):
                    today = i + WINDOW_SIZE - 1
                    if not data_date[today].year == 2020:
                        continue

                    if date_to_order[data_date[today]] - date_to_order[data_date[today - 1]] != 1:  # 出现停牌
                        k = 1
                        while date_list[date_to_order[data_date[today - 1]] + k] != data_date[today]:
                            for j in Method:
                                prev_day_value = sum(money[j]) + sum(stock[j]) * data_price[today - 1]
                                agg_money_by_date[j][date_list[date_to_order[data_date[today - 1]] + k]] += prev_day_value
                            k += 1

                    X_date.append(data_date[today])
                    for j in Method:
                        today_value = sum(money[j]) + sum(stock[j]) * data_price[today]
                        total_by_day[j].append(today_value)
                        agg_money_by_date[j][data_date[today]] += today_value

                    prediction = defaultdict(int)

                    prediction['RANDOM_WALK'] = 1

                    prediction['MOVING_AVERAGE'] = 0
                    if sum(data_price[i: i + WINDOW_SIZE]) / WINDOW_SIZE < \
                            sum(data_price[i + WINDOW_SIZE - 2: i + WINDOW_SIZE]) / 2:
                        prediction['MOVING_AVERAGE'] = 1

                    prediction['RIDGE'] = prediction_ridge[i]

                    prediction['LGBM'] = prediction_lgbm[i]

                    # prediction['TRANSFORMER'] = (prediction_transformer[i][0] > .5)
                    prediction['TRANSFORMER'] = transformer_pickle[(filename[0:6], today)]

                    # reg = LinearRegression().fit(np.array([t for t in range(WINDOW_SIZE)]).reshape(-1, 1),
                    #                              data_price[i: i + WINDOW_SIZE])

                    # if reg.coef_[0] > THRESH:
                    #     prediction['MIXED'] = 1
                    # else:
                    #     prediction['MIXED'] = prediction['TRANSFORMER']

                    for j in Method:
                        if prediction[j] == 0:  # 跌
                            money[j][index] += stock[j][index] * data_price[today]
                            stock[j][index] = 0  # 卖空
                        elif prediction[j] == 1:  # 涨
                            stock[j][index] += money[j][index] / data_price[today]
                            money[j][index] = 0
                    index += 1
                    if index == 5:
                        index = 0

            X_date.append(data_date[-1])
            end_total_money = defaultdict(float)
            for j in Method:
                end_total_money[j] = sum(money[j]) + sum(stock[j]) * data_price[-1]
                total_by_day[j].append(end_total_money[j])
                agg_money[j] += end_total_money[j]
                agg_money_by_date[j][data_date[-1]] += end_total_money[j]

            end_list = [end_total_money[j] for j in Method]
            sorted_end_list = sorted(end_list, reverse=True)
            for j in Method:
                rank[j].append(sorted_end_list.index(end_total_money[j]) + 1)

        sns.set_theme()
        plt.cla()
        plt.figure(figsize=(6, 4.5), dpi=100)
        print(f'WINDOW = {WINDOW_SIZE}, FORECAST = {FORCAST_SIZE}')
        for j in Method:
            print(j, agg_money[j] / 4900 - 1)
            rate_list = [agg_money_by_date[j][d] / 4900 - 1 for d in X_date]
            plt.plot(X_date, rate_list, label=j)
        plt.title(f'Yield on the 49 Stocks of Test Set\nWINDOW = {WINDOW_SIZE}, FORECAST = {FORCAST_SIZE}')
        plt.ylabel("Yield")
        plt.xlabel('Time')
        plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', ncol=1)
        plt.savefig('huice' + MODE, bbox_inches='tight')

        end = datetime.now()
        print(end - begin)
        # print(f'WINDOW = {WINDOW_SIZE}, FORECAST = {FORCAST_SIZE}')
        # for j in Method:
        #     print(j + str(sum(rank[j]) / len(rank[j])))
