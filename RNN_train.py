# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 19:20:13 2018

@author: mingyang.wang
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
 
# 遍历指定目录，显示目录下的所有AD数据文件名
def get_file_name(filepath):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        childDir = os.path.join('%s\\%s' % (filepath, allDir))
        if os.path.isfile(childDir) and allDir[-6:-4] == "AD":
            files.append(childDir)
        elif os.path.isdir(childDir):
            get_file_name(childDir)

#遍历AD数据文件，加载数据
def get_data(file):
    col = [i for i in range(8)]
    row_data = 5000
    label = [] 
    for i in range(len(file)):
        data = pd.read_csv(file[i], header=None, sep="\t", usecols=col)
        if(data.shape[0]>row_data):
            label.append(1)
        else:
            label.append(0)
            listd = []
            for j in range(int(row_data/data.shape[0]) +1):
                listd.append(data)
            data = pd.concat(listd)
        data = np.array(data[:row_data]).reshape((1, len(col)*row_data))
        if i :
            data_all = np.row_stack((data_all, data))
        else:
            data_all = data.copy() 
    label = np.array(label)
    return data_all, label

#转换为独热编码
def make_one_hot(data):
    return (np.arange(2)==data[:,None]).astype(np.integer)

#数据随机化  
def shuffle_aligned_list(data):
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]

#定义batch数据生成器  
def batch_data(data, batch_size, shuffle=True):
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(data[0]):
            batch_count = 0
            if shuffle:
                data= shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

#建立RNN模型
def RNN(X, weights, biases, n_inputs, n_steps, n_hidden_units, batch_size):
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, reuse=tf.AUTO_REUSE)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results

#计算准确率
def Accuracy():
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) #准确率
    
#创建RNN模型预测信号
def RNN_model():  
    tf.reset_default_graph()
    lr = 0.001   
    training_iters = 1000
    batch_size = 100
    n_inputs = 200   
    n_steps = 200  
    n_hidden_units = 128   
    n_classes = 2      
    
    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])   #输入
    y = tf.placeholder(tf.float32, [None, n_classes])

    weights = {    # 参数
        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))}
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))}

    pred = RNN(x, weights, biases, n_inputs, n_steps, n_hidden_units, batch_size)  #预测
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  #损失函数
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)  #训练

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) #准确率
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver(max_to_keep=1)   #保存模型
        step = 0
        for i in range(training_iters):
            batch_xs, batch_ys = next(batch_data([train_data, train_label], batch_size))
            sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys,})
            if i % 100 == 0:
                print(sess.run(accuracy, feed_dict={x:test_data[:100], y:test_label[:100]}))
                saver.save(sess,'E:\\Github\\Radar_RNN\\ckpt\\Radar_rnn.ckpt',global_step=step+1)

if __name__ == "__main__":
    filepath = "E:\\data\\leida"
    files=[]
    get_file_name(filepath)
    datas, labels = get_data(files)
    datas = datas.reshape((datas.shape[0], 200, 200))
    labels = make_one_hot(labels)
    train_data, test_data, train_label, test_label = train_test_split(datas, labels, test_size=0.2, random_state=0)
    print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)
    RNN_model()