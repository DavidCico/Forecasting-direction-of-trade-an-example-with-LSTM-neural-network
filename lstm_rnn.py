import numpy as np
import pandas as pd
from pandas import DataFrame as df
import os
from pathlib import Path
import tensorflow as tf

from sklearn.model_selection import train_test_split  # cross validation
from sklearn.linear_model import LinearRegression  # linear regression
from sklearn import metrics
import matplotlib.pyplot as plt

from functions import HiddenVOI, VOI, DailyExtract, DailyExtractAll, next_batch

pd.options.mode.chained_assignment = None
tf.set_random_seed(1)  # set random seed

############################### import data #################################


####### Small Feature Set  #######################

"""
 file = Path(os.getcwd()) / "TRAIN_DATA.csv"
 data = pd.read_csv(open(file))
 datelist = sorted(list(set(data['TRADEDATE'])))

 # beg_index = 20
 # end_index = 28
 def DataSet(beg_index, end_index):
     # train_range = 20
     traindata = df([])
     for i in range(beg_index, end_index):
         dataset = DailyExtract(datelist[i], data)
         traindata = traindata.append(dataset)

     traindata.reset_index(drop=True, inplace=True)
     # traindata = traindata[traindata['TICKDIR'] != 'FLAT']
     # row = traindata[traindata['TRADEDATE'] < datelist[train_range - 2]].shape[0]
     input_X = traindata.iloc[:, 2:]
     input_X = input_X.drop(['TenDPriceChg','TICKDIR'], axis=1)
     input_x = input_X.reindex(sorted(input_X.columns), axis=1)
     input_x['intercept'] = 1
     input_X = np.array(input_x)
     input_y = traindata['TICKDIR']
     # input_y = np.where(input_y > 0, 1, np.where(input_y<0,-1,0))
     # input_Y = np.array(pd.get_dummies(pd.Series(input_y)))
     input_Y = df([])
     zero = np.zeros(input_y.shape[0])
     zero[input_y == 0] = 1
     input_Y['zero'] = zero
     one = np.zeros(input_y.shape[0])
     one[input_y == 1] = 1
     input_Y['one'] = one
     negone = np.zeros(input_y.shape[0])
     negone[input_y == -1] = 1
     input_Y['negone'] = negone
     input_Y = np.rint(np.array(input_Y))
    return input_X, input_Y
"""

####### Large Feature Set #####################

file = Path(os.getcwd()) / "TRAIN_DATA.csv"
data = pd.read_csv(open(file))
datelist = sorted(list(set(data['TRADEDATE'])))


# beg_index = 25
# end_index = 26
def DataSet(beg_index, end_index):
    # train_range = 20
    traindata = df([])
    for i in range(beg_index, end_index):
        dataset = DailyExtractAll(datelist[i], data)
        traindata = traindata.append(dataset)
    traindata.reset_index(drop=True, inplace=True)
    # row = traindata[traindata['TRADEDATE'] < datelist[train_range - 2]].shape[0]

    # input_X = traindata
    # traindata = traindata[traindata['TICKDIR'] != 'FLAT']
    input_X = traindata.drop(['TenDPriceChg', 'TRADEDATE', 'TIME', 'MARKETTIME', 'TICKDIR'], axis=1)
    input_x = input_X.reindex(sorted(input_X.columns), axis=1)
    input_x['intercept'] = 1
    input_X = np.array(input_x)
    input_y = traindata['TICKDIR']
    # input_y = np.where(input_y > 0, 1, np.where(input_y<0,-1,0))
    # input_Y = np.array(pd.get_dummies(pd.Series(input_y)))
    input_Y = df([])
    zero = np.zeros(input_y.shape[0])
    zero[input_y == 0] = 1
    input_Y['zero'] = zero
    one = np.zeros(input_y.shape[0])
    one[input_y == 1] = 1
    input_Y['one'] = one
    negone = np.zeros(input_y.shape[0])
    negone[input_y == -1] = 1
    input_Y['negone'] = negone
    input_Y = np.rint(np.array(input_Y))

    return input_X, input_Y


#############################  parameters  #############################

# hyperparameters

lr = 0.001  # learning rate
training_iters = 1000  # train step
batch_size = 1000
# batch_size_test = 100
n_inputs = 26  # number of features, X.shape[1] =26, 56
n_steps = 10  # time steps
n_hidden_units = 80  # neurons in hidden layer
n_classes = 3  # 3 output classes (-1,0,1)

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

#  weights and biases
weights = {
    # shape (25, 80)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (80, 3)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (80, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (3, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


###############################  RNN  ###################################


def RNN(X, weights, biases):
    # X ==> (20 batches * 4 steps, 25 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (20 batches, 4 steps, 80 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # use basic LSTM Cell.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size / n_steps, dtype=tf.float32)  # state

    # output result
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results


############################  Train and Test  ################################

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

day_num = len(datelist)
# day_num = 12
train_range = 15
for i in range(day_num - train_range):
    # i = 11
    beg_index = i
    end_index = i + train_range
    input_X, input_Y = DataSet(beg_index, end_index)
    test_X, test_Y = DataSet(end_index, end_index + 10)
    # print(test_X.shape)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        step = 0
        Train_Acu = []
        Test_Acu = []
        while step < training_iters:
            # train batch
            batch_xs, batch_ys = next_batch(batch_size, input_X, input_Y)
            batch_xs = batch_xs.reshape([-1, n_steps, n_inputs])
            batch_ys = batch_ys[n_steps - 1:: n_steps]
            # test batch
            test_xs, test_ys = next_batch(batch_size, test_X, test_Y)
            test_xs = test_xs.reshape([-1, n_steps, n_inputs])
            test_ys = test_ys[n_steps - 1:: n_steps]
            # training
            sess.run([train_op], feed_dict={
                x: batch_xs,
                y: batch_ys,
            })
            if step % 20 == 0:
                acc1 = sess.run(accuracy, feed_dict={
                    x: batch_xs,
                    y: batch_ys, })
                print("Training Accuracy: ", acc1)
                Train_Acu.append(acc1)
                acc2 = sess.run(accuracy, feed_dict={
                    x: test_xs,
                    y: test_ys,
                })
                print("Test Accuracy on ", datelist[end_index], " : ", acc2)
                Test_Acu.append(acc2)
            step += 1
        AccuracySet = df({'Train': Train_Acu, 'Test': Test_Acu})
