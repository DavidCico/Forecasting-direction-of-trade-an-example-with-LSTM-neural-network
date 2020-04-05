import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
from sklearn.linear_model import LinearRegression  # linear regression
from functions import DailyExtract


currentDir = Path(os.getcwd())
file = currentDir / "TRAIN_DATA.csv"
data = pd.read_csv(open(file))

pd.options.mode.chained_assignment = None


datelist = sorted(list(set(data['TRADEDATE'])))
train_range = 20
traindata = df([])
for i in range(train_range):
    dataset = DailyExtract(datelist[i], data)
    traindata = traindata.append(dataset)
traindata.reset_index(drop=True, inplace=True)
row = traindata[traindata['TRADEDATE'] < datelist[train_range - 2]].shape[0]

X = traindata.iloc[:, 2:]
X = X.drop(['TenDPriceChg', 'BidAskSpd'], axis=1)
input_x = X.reindex(sorted(X.columns), axis=1)
input_x['intercept'] = 1
y = traindata['TenDPriceChg']

X_train, X_test, y_train, y_test = input_x.iloc[:row, ], input_x.iloc[row:, ], y.iloc[:row], y.iloc[row:]

print('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n,  y_test.shape={}'.format(X_train.shape, y_train.shape,
                                                                                          X_test.shape, y_test.shape))
linreg = LinearRegression()
model = linreg.fit(X_train, y_train)
print(model)
print(linreg.intercept_)
print(linreg.coef_)


y_pred = linreg.predict(X_train)
print(y_pred)
positive = 0
negative = 0
stable = 0
for i in range(len(y_pred)):
    if y_pred[i] >= 0.01 and y_train.values[i] > 0:
        positive += 1
    elif y_pred[i] <= -0.01 and y_train.values[i] < 0:
        negative += 1
    elif abs(y_pred[i]) < 0.01 and y_train.values[i] < 0:
        stable += 1
accuracy = (positive + negative + stable) / len(y_pred)
# calculate RMSE by hand
print("Predication Accuracy:", accuracy)

plt.figure()
plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
plt.plot(range(len(y_pred)), y_train, 'r', label="train")
plt.legend(loc="upper right")
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
plt.show()

y_pred = linreg.predict(X_test)
print(y_pred)
positive = 0
negative = 0
stable = 0
for i in range(len(y_pred)):
    if y_pred[i] >= 0.01 and y_test.values[i] > 0:
        positive += 1
    elif y_pred[i] <= -0.01 and y_test.values[i] < 0:
        negative += 1
    elif abs(y_pred[i]) < 0.01 and y_test.values[i] < 0:
        stable += 1
accuracy = (positive + negative + stable) / len(y_pred)
# calculate RMSE by hand
print("Predication Accuracy:", accuracy)

plt.figure()
plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
plt.plot(range(len(y_pred)), y_test, 'r', label="test")
plt.legend(loc="upper right")
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
plt.show()
