import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from sklearn.linear_model import LinearRegression  # linear regression

currentDir = Path(os.getcwd())
file = currentDir / "TRAIN_DATA.csv"
data = pd.read_csv(open(file))

pd.options.mode.chained_assignment = None


def DailyExtract(date):
    raw = data[data['TRADEDATE'] == date].reset_index(drop=True)
    midprice = (raw['BID']+raw['ASK'])/2
    raw['midprice'] = midprice
    length = len(midprice)
    raw['BidAskSpd'] = raw['ASK'] - raw['BID']
    ## compute ten days mid-price change
    TenDPriceChg = midprice.rolling(20).mean()[20:].reset_index(drop=True) \
                   - midprice[:length-20].reset_index(drop=True)
    raw['TenDPriceChg'] = TenDPriceChg
    # compute Volume Order Imbalance (VOI)

    ## compute  BidVolDelta
    BidPriceChg = raw[['BID']].diff()
    raw['BidPriceChg'] = BidPriceChg['BID']
    BidVolChg = raw[['BIDSIZE']].diff()
    raw['BidVolChg'] = BidVolChg['BIDSIZE']
    raw['BidVolDelta'] = 0
    raw['BidVolDelta'][raw['BidPriceChg'] > 0 ] = raw[raw['BidPriceChg'] >0 ]['BIDSIZE']
    raw['BidVolDelta'][raw['BidPriceChg'] == 0] = raw[raw['BidPriceChg'] == 0]['BidVolChg']

    ## compute  AskVolDelta
    AskPriceChg = raw[['ASK']].diff()
    raw['AskPriceChg'] = AskPriceChg['ASK']
    AskVolChg = raw[['ASKSIZE']].diff()
    raw['AskVolChg'] = AskVolChg['ASKSIZE']
    raw['AskVolDelta'] = 0
    raw['AskVolDelta'][raw['AskPriceChg'] < 0] = raw[raw['AskPriceChg'] < 0]['ASKSIZE']
    raw['AskVolDelta'][raw['AskPriceChg'] == 0] = raw[raw['AskPriceChg'] == 0]['AskVolChg']

    ## compute VOI
    raw['VOI'] = raw['BidVolDelta'] - raw['AskVolDelta']

    # compute the mid-price basis (MPB)
    ## compute the average traded price
    TradeAtBid = raw[['BIDVOLUME']].diff()
    raw['TradeAtBid'] = TradeAtBid['BIDVOLUME']
    TradeAtAsk = raw[['ASKVOLUME']].diff()
    raw['TradeAtAsk'] = TradeAtAsk['ASKVOLUME']
    raw['TradeValue'] = raw['BID'] * raw['TradeAtBid'] + raw['ASK'] * raw['TradeAtAsk']
    raw['AvgTradePrice'] = np.where( raw['TradeValue']!=0,
                                     raw['TradeValue']/(raw['TradeAtBid']+ raw['TradeAtAsk']), np.nan)
    raw['AvgTradePrice'].iloc[0] = raw['midprice'].iloc[0]
    raw[['AvgTradePrice']] = raw[['AvgTradePrice']].fillna(method='ffill')

    ## compute MPB
    raw['MPB'] = raw['AvgTradePrice'] - midprice.rolling(2).mean()
    raw['MPB'][0] = 0

    # lag by L
    L = 5
    namelist = []
    for _ in range(L):
        name = "VOI" + str(i + 1)
        namelist.append(name)
        raw[name] = raw['VOI'][i + 1:].reset_index(drop=True)
        raw[name] = raw[name] / raw['BidAskSpd']
        name = "BIDASKIMBALANCE" + str(i + 1)
        namelist.append(name)
        raw[name] = raw['BIDASKIMBALANCE'][i + 1:].reset_index(drop=True)
        raw[name] = raw[name] / raw['BidAskSpd']

    # scaled by bid ask spread
    raw['VOI'] = raw['VOI'] / raw['BidAskSpd']
    raw['BIDASKIMBALANCE'] = raw['BIDASKIMBALANCE'] / raw['BidAskSpd']
    raw['MPB'] = raw['MPB'] / raw['BidAskSpd']
    namelist = ['TRADEDATE', 'TIME', 'BidAskSpd', 'BIDASKIMBALANCE', 'VOI', 'MPB', 'TenDPriceChg'] + namelist

    output = raw[namelist]
    output = output[ output['BidAskSpd'] >= 0.00001]
    output = output.dropna()
    return output


datelist = sorted(list(set(data['TRADEDATE'])))
train_range = 20
train_ac = []
test_ac = []
for j in range(len(datelist)-train_range):
    traindata = df([])
    for i in range(train_range):
        dataset = DailyExtract(datelist[i+j])
        traindata = traindata.append(dataset)
    traindata.reset_index(drop=True, inplace=True)
    row = traindata[traindata['TRADEDATE'] < datelist[j+ train_range - 1]].shape[0]

    X = traindata.iloc[:, 2:]
    X = X.drop(['TenDPriceChg', 'BidAskSpd'], axis=1)
    input_x = X.reindex(sorted(X.columns), axis=1)
    input_x['intercept'] = 1
    y = traindata['TenDPriceChg']

    X_train, X_test, y_train, y_test = input_x.iloc[:row, ], input_x.iloc[row:, ], y.iloc[:row], y.iloc[row:]
    # print('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n,  y_test.shape={}'.format(X_train.shape,
    #                                                                                           y_train.shape,
    #                                                                                           X_test.shape,
    #                                                                                           y_test.shape))
    linreg = LinearRegression()
    model = linreg.fit(X_train, y_train)

    # training set predictions
    y_pred = linreg.predict(X_train)
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
    train_ac.append(accuracy)
    # calculate RMSE by hand
    print("Training Accuracy:", accuracy)

    '''
    ROC curve
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_train, 'r', label="train")
    plt.legend(loc="upper right") 
    plt.xlabel("the number of sales")
    plt.ylabel('value of sales')
    plt.show()
    '''

    y_pred = linreg.predict(X_test)
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
    test_ac.append(accuracy)
    # calculate RMSE by hand
    print("Testing Accuracy out of sample:", accuracy)

    '''
    ROC curve
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.xlabel("the number of sales")
    plt.ylabel('value of sales')
    plt.show()
    '''


# train_ac = df(train_ac)
# test_ac = df(test_ac)
# savepath = Path(os.getcwd())/"PreData/LinearRegress/" / "{}d-train_acc.csv".format(train_range)
# train_ac.to_csv(savepath)
# savepath = Path(os.getcwd())/"PreData/LinearRegress/" / "{}d-test_acc.csv".format(train_range)
# test_ac.to_csv(savepath)











