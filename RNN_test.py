# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 19:20:13 2018

@author: mingyang.wang
"""

import os
import pandas as pd
import numpy as np
 
filepath = "E:\\Github\\CNN_tensorflow_180726\\data\\leida2"
 
# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    file = []
    for allDir in pathDir:
        if allDir[-6:-4] == "AD":
            child = os.path.join('%s\%s' % (filepath, allDir))
            file.append(child)
    return(file)
    
file = eachFile(filepath)
col = [0,1,2,3,4,5,6,7]
row_data = 5000
labels = [] 
data_list = []
first = 1
for i in file:
    data = pd.read_csv(i, header=None, sep="\t", usecols=col)
    if(data.shape[0]>row_data):
        labels.append(1)
        data = np.array(data[:row_data]).reshape((1, len(col)*row_data))
    else:
        labels.append(0)
        k = int(row_data/data.shape[0]) +1
        listd = []
        for j in range(k):
            listd.append(data)
        data = pd.concat(listd)
        data = np.array(data[:row_data]).reshape((1, len(col)*row_data))
    if first:
        data_all = data.copy() 
    else:
        data_all = np.row_stack((data_all, data))
    first = 0
labels = np.array(labels)

def make_one_hot(data1):
    return (np.arange(2)==data1[:,None]).astype(np.integer)
        
labels = make_one_hot(labels)
data_all = data_all.reshape((len(data_all),200,200))

import tensorflow as tf

# set random seed for comparing the two result calculations
tf.set_random_seed(1)


# hyperparameters
lr = 0.001
training_iters = 1000
batch_size = 50

n_inputs = 200   # MNIST data input (img shape: 28*28)
n_steps = 200   # time steps
n_hidden_units = 128   # neurons in hidden layer
n_classes = 2      # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (40, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 2)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (2, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):

    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, reuse=tf.AUTO_REUSE)

    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def batch_data(data, label):  # 定义batch数据生成器
    idx = 0
    while True:
        if idx+50>180:
            idx=0
        start = idx
        idx += 50
        yield data[start:start+50], label[start:start+50]

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(tf.local_variables_initializer())
    sess.run(init)
    step = 0
    batch = batch_data(data_all, labels)
    saver=tf.train.Saver(max_to_keep=3)
    model_file=tf.train.latest_checkpoint('E:\\Github\\CNN_tensorflow_180726\\data\\ckpt\\')
    saver.restore(sess,model_file)
    for i in range(4):
        batch_xs, batch_ys = next(batch)
    
        print(sess.run(accuracy, feed_dict={
        x:batch_xs,
        y:batch_ys,
        }))
