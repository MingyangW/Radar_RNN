# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 11:10:24 2018

@author: mingyang.wang
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 遍历指定目录，显示目录下的所有AD数据文件名
def get_file_name(filepath):
    if os.path.exists(filepath):
        pathDir =  os.listdir(filepath)
        for allDir in pathDir:
            childDir = os.path.join('%s\\%s' % (filepath, allDir))
            if os.path.isfile(childDir) and allDir[-6:-4] == "AD":
                files.append(childDir)
            elif os.path.isdir(childDir):
                get_file_name(childDir)
    else:
        print("Filepath is not exist!")

#遍历AD数据文件，加载数据
def get_data(file, col):
    col_list = [i for i in range(col)]
    row_data = 5000
    label = [] 
    for i in range(len(file)):
        data = pd.read_csv(file[i], header=None, sep="\t", usecols=col_list)
        if(data.shape[0]>row_data):
            label.append(1)
        else:
            label.append(0)
            listd = []
            for j in range(row_data//data.shape[0] +1):
                listd.append(data)
            data = pd.concat(listd)
        data = np.array(data[:row_data]).reshape((1, col*row_data))
        if i == 0 :
            data_all = data.copy()
        else:
            data_all = np.row_stack((data_all, data))
    label = np.array(label)
    return data_all, label

#遍历AD数据文件，加载带标签数据
def get_data2(filepath, col):
#    label_dict = {"9710":0, "9725":1, "9800":2, "9870":3}
    col_list = [i for i in range(col)]
    row_data = 5000
    label = []
    data_sign = 1
    datas = np.zeros((1, col*row_data))
    if os.path.exists(filepath):
        pathDir = os.listdir(filepath)  #进入文件目录
        label_dict = {i:j for i,j in zip(pathDir, range(len(pathDir)))}
        for allDir in range(len(pathDir)):
            childDir = os.path.join('%s\\%s' % (filepath, pathDir[allDir]))
            pathfile = os.listdir(childDir)
            for allfile in range(len(pathfile)):  #进入AD文件目录
                if pathfile[allfile][-6:-4] == "AD":
                    childfile = os.path.join('%s\\%s' % (childDir, pathfile[allfile]))
                    data = pd.read_csv(childfile, header=None, sep="\t", usecols=col_list)
                    listd = []
                    for j in range(row_data//data.shape[0] +1):
                        listd.append(data)
                    data = pd.concat(listd)
                    data = np.array(data[:row_data]).reshape((1, col*row_data))
                    if data_sign==1:
                        datas = data
                        data_sign = 0
                    else:
                        datas = np.row_stack((datas, data))
                    label.append(label_dict[pathDir[allDir]])
        labels = np.array(label)
        return datas, labels
    else:
        print("Filepath is not exist!")


#转换为独热编码
def make_one_hot(data, n):
    return (np.arange(n)==data[:,None]).astype(np.integer)

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

    
#创建RNN模型预测信号
def RNN_model(data, label, batch_size, n_classes, n_inputs, n_steps, is_train=True, training_iters=1000):  
    tf.reset_default_graph()
    lr = 0.001   
#    training_iters = training_iter
#    batch_size = batch
#    n_inputs = 200  
#    n_steps = 200  
    n_hidden_units = 128   
#    n_classes = 2      
    
    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])   #输入
    y = tf.placeholder(tf.float32, [None, n_classes])

    weights = {    # 参数
        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
        }
    
    pred = RNN(x, weights, biases, n_inputs, n_steps, n_hidden_units, batch_size)  #预测
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  #损失函数
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)  #训练

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) #准确率
    accursum = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver(max_to_keep=1)   #保存模型
        if  is_train:  #模型训练
            cost_list = []
            for i in range(training_iters):
                batch_xs, batch_ys = next(batch_data([data, label], batch_size))
                sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
                cost_list.append(sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}))
                if i % (training_iters//10) == 0:
                    batch_xs, batch_ys = next(batch_data([data, label], batch_size))
                    print(sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys}))
            saver.save(sess,'E:\\Github\\Radar_RNN\\ckpt_0827\\radar.ckpt',global_step=i)
            print("Training is OVER!")
            print("Train_data_count: %d, Batch_size: %d, Training_iters: %d" %(data.shape[0], batch_size, training_iters))
            plt.plot(cost_list)
            plt.show()
        else:  #模型预测
            model_file=tf.train.latest_checkpoint('E:\\Github\\Radar_RNN\\ckpt_0827\\')
            saver.restore(sess,model_file)
    #        print(sess.run(accuracy, feed_dict={x:data, y:label}))
    #        print(sess.run(pred, feed_dict={x:data, y:label}))
    #        print(labels[:100])
            class_accur = 0
            n_batch = label.shape[0] // batch_size
            for i in range(n_batch):
                accur_num = sess.run(accursum, 
                                     feed_dict={x:data[batch_size*i:batch_size*(i+1)],y:label[batch_size*i:batch_size*(i+1)]})
                class_accur += accur_num
            result = class_accur / (n_batch * batch_size)
            print("test_data num: %d , Accuracy: %.4f" % (n_batch * batch_size, result))

if __name__ == "__main__":
    file_path_train = "E:\\data\\leida"  #原始AD数据存放地址
    file_path_test = "E:\\data\\leida3"  #原始AD数据存放地址
    file_path = "E:\\data\\radar4"
    data_row_num = 200  #RNN识别脉冲数据结构，行
    data_col_num = 200  #RNN识别脉冲数据结构，列
    batch_size = 1   #训练批次数据量
    AD_col = 8  #原始aD数据有效数列
    n_class = 4  #类别数量
    train_iter = 10000
    is_train = False
    
    files=[]
#    get_file_name(file_path_train) 
#    get_file_name(file_path_test)
#    AD_data, AD_label = get_data(files, AD_col)
    AD_data, AD_label = get_data2(file_path, AD_col)
    AD_data_reshape = AD_data.reshape((AD_data.shape[0], data_row_num, data_col_num))
    AD_label_onehot = make_one_hot(AD_label, n_class)
    
    AD_train_data, AD_test_data, AD_train_label, AD_test_label = train_test_split(AD_data, AD_label_onehot, test_size=0.7, random_state=0)
    std = StandardScaler()
    AD_train_data_std = std.fit_transform(AD_train_data)
    AD_test_data_std = std.transform(AD_test_data)
    AD_train_data_std = AD_train_data_std.reshape((AD_train_data_std.shape[0], data_row_num, data_col_num))
    AD_test_data_std = AD_test_data_std.reshape((AD_test_data_std.shape[0], data_row_num, data_col_num))
    print(AD_train_data_std.shape, AD_test_data_std.shape, AD_train_label.shape, AD_test_label.shape)
    

    RNN_model(data = AD_data_reshape, 
              label = AD_label_onehot, 
              batch_size = batch_size, 
              n_classes = n_class, 
              n_inputs = data_col_num, 
              n_steps = data_row_num,
              is_train = is_train,
              training_iters = train_iter
              )

        