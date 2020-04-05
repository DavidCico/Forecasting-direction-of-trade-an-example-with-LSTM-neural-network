import numpy as np


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def HiddenVOI(i, raw):
    # compute the hidden Volume Order Imbalance (VOI)
    ## compute  BidVolDelta
    name = 'BID' + str(i)
    BidPriceChg = raw[[name]].diff()
    col1 = name + 'PriceChg'
    raw[col1] = BidPriceChg[name]
    name = 'BIDVOLUME' + str(i)
    BidVolChg = raw[[name]].diff()
    col2 = 'Bid' + str(i) + 'VolChg'
    raw[col2] = BidVolChg[name]
    BidDelta = 'Bid' + str(i) + 'VolDelta'
    raw[BidDelta] = 0
    raw[BidDelta][raw[col1] > 0] = raw[raw[col1] > 0][name]
    raw[BidDelta][raw[col1] == 0] = raw[raw[col1] == 0][col2]

    ## compute  AskVolDelta
    name = 'ASK' + str(i)
    AskPriceChg = raw[[name]].diff()
    col1 = name + 'PriceChg'
    raw[col1] = AskPriceChg[name]
    name = 'ASKVOLUME' + str(i)
    AskVolChg = raw[[name]].diff()
    col2 = 'Ask' + str(i) + 'VolChg'
    raw[col2] = AskVolChg[name]
    AskDelta = 'Ask' + str(i) + 'VolDelta'
    raw[AskDelta] = 0
    raw[AskDelta][raw[col1] < 0] = raw[raw[col1] < 0][name]
    raw[AskDelta][raw[col1] == 0] = raw[raw[col1] == 0][col2]

    ## compute the hidden VOI
    HVOI = 'HiddenVOI' + str(i)
    raw[HVOI] = raw[BidDelta] - raw[AskDelta]
    pass


def VOI(raw):
    # compute Volume Order Imbalance (VOI)
    ## compute  BidVolDelta
    BidPriceChg = raw[['BID']].diff()
    raw['BidPriceChg'] = BidPriceChg['BID']
    BidVolChg = raw[['BIDSIZE']].diff()
    raw['BidVolChg'] = BidVolChg['BIDSIZE']
    raw['BidVolDelta'] = 0
    raw['BidVolDelta'][raw['BidPriceChg'] > 0] = raw[raw['BidPriceChg'] > 0]['BIDSIZE']
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
    pass


def DailyExtract(date, data):
    raw, namelist, hiddenVOIlist, oldlist = _prepare_raw_and_name_list(date, data)

    namelist = ['TRADEDATE', 'TIME', 'BidAskSpd', 'BIDASKIMBALANCE', 'VOI', 'MPB', 'TenDPriceChg'] \
               + namelist + hiddenVOIlist \
               + ['BIDTRADE', 'BIDTRADE1', 'BIDTRADE2', 'BIDTRADE3', 'ASKTRADE', 'ASKTRADE1', 'ASKTRADE2', 'ASKTRADE3'] \
               + ['TICKDIR']

    output = raw[namelist]
    output = output[output['BidAskSpd'] >= 0.00001]
    output = output.dropna()
    return output


def DailyExtractAll(date, data):
    raw, namelist, hiddenVOIlist, oldlist = _prepare_raw_and_name_list(date, data)

    # classify the tick direction
    raw.replace("UP", 1, inplace=True)
    raw.replace("UPCYCLE", 1, inplace=True)
    raw.replace("DOWN", -1, inplace=True)
    raw.replace("DOWNCYCLE", -1, inplace=True)
    raw.replace("FLAT", 0, inplace=True)
    for i in range(4):
        if i == 0:
            col2 = "LASTTICKDURATION"
        else:
            col2 = "LASTTICKDURATION" + str(i)
        totaltime = np.max(raw['MARKETTIME'])
        raw[col2] = raw[col2] / totaltime

    namelist = ['TRADEDATE', 'TIME', 'BidAskSpd', 'BIDASKIMBALANCE', 'VOI', 'MPB', 'TenDPriceChg'] \
               + namelist + hiddenVOIlist \
               + ['BIDTRADE', 'BIDTRADE1', 'BIDTRADE2', 'BIDTRADE3', 'ASKTRADE', 'ASKTRADE1', 'ASKTRADE2', 'ASKTRADE3']
    namelist = np.union1d(oldlist, namelist)
    output = raw[namelist]
    output = output[output['BidAskSpd'] >= 0.00001]
    output = output.dropna()
    return output


def _prepare_raw_and_name_list(date, data):
    raw = data[data['TRADEDATE'] == date].reset_index(drop=True)
    oldlist = raw.columns
    midprice = (raw['BID'] + raw['ASK']) / 2
    raw['midprice'] = midprice
    length = len(midprice)
    raw['BidAskSpd'] = raw['ASK'] - raw['BID']

    # compute ten days mid-price change
    TenDPriceChg = midprice.rolling(20).mean()[20:].reset_index(drop=True) \
                   - midprice[:length - 20].reset_index(drop=True)
    raw['TenDPriceChg'] = TenDPriceChg

    # Compute the VOI and hidden VOI
    VOI(raw)
    hiddenVOIlist = []
    for i in range(3):
        num = i + 1
        HiddenVOI(num, raw)
        HVOI = 'HiddenVOI' + str(i + 1)
        raw[HVOI] = raw[HVOI] / raw['BidAskSpd']
        hiddenVOIlist.append(HVOI)

    # compute the mid-price basis (MPB)
    ## compute the average traded price
    TradeAtBid = raw[['BIDVOLUME']].diff()
    raw['TradeAtBid'] = TradeAtBid['BIDVOLUME']
    TradeAtAsk = raw[['ASKVOLUME']].diff()
    raw['TradeAtAsk'] = TradeAtAsk['ASKVOLUME']
    raw['TradeValue'] = raw['BID'] * raw['TradeAtBid'] + raw['ASK'] * raw['TradeAtAsk']
    raw['AvgTradePrice'] = np.where(raw['TradeValue'] != 0,
                                    raw['TradeValue'] / (raw['TradeAtBid'] + raw['TradeAtAsk']), np.nan)
    raw['AvgTradePrice'].iloc[0] = raw['midprice'].iloc[0]
    raw[['AvgTradePrice']] = raw[['AvgTradePrice']].fillna(method='ffill')

    ## compute MPB
    raw['MPB'] = raw['AvgTradePrice'] - midprice.rolling(2).mean()
    raw['MPB'][0] = 0

    # lag by L
    L = 5
    namelist = []
    for i in range(L):
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

    return raw, namelist, hiddenVOIlist, oldlist
