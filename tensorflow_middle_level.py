# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:35:01 2018

@author: zhangjuefei
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected

# 获取手写数字。
mnist = input_data.read_data_sets("D:\documents\ANN_presentation\charts\data")

# 训练集和测试集。
train = mnist.train
test = mnist.test

# 定义一些与网络结构和训练有关的常量。
n_inputs = 28 * 28
n_hidden_1 = 300
n_hidden_2 = 100
n_outputs = 10
n_epochs = 30
batch_size = 50
learning_rate = 0.01

# 清除运算图。
tf.reset_default_graph()

# 定义占位变量。
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

# 定义网络的计算图。
with tf.name_scope("dnn"):
    
    # 第一隐藏层。
    hidden_layer_1 = fully_connected(X, n_hidden_1, activation_fn=tf.nn.tanh, scope="hidden1")
    # 第二隐藏层。
    hidden_layer_2 = fully_connected(hidden_layer_1, n_hidden_2, activation_fn=tf.nn.tanh, scope="hidden2")
    # 输出层。
    logits = fully_connected(hidden_layer_2, n_outputs, activation_fn=None, scope="logits")

# 网络损失。
with tf.name_scope("loss"):

    # 根据输出层输出的 10 个值（logits）的 softmax 值和样本标注类别计算交叉熵。
    cross_entropy_of_softmax_outputs = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    
    # 一个 mini batch 的平均损失。
    loss = tf.reduce_mean(cross_entropy_of_softmax_outputs, name="loss")

# 训练运算。
with tf.name_scope("train"):

    # 创建一个随机梯度下降优化器，学习率为 0.01 。
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # 创建优化运算，用随机梯度下降优化器优化 loss （交叉熵）。每执行一次，进行一步梯度下降。
    train_op = optimizer.minimize(loss)

# 评估。
with tf.name_scope("eval"):

     # 第 y 个 logit 是不是 top 1 （最大），是的话说明网络输出第 y 类概率最大，分类正确。
    correct = tf.nn.in_top_k(logits, y, 1)
    
    # 计算平均准确率。
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 运行。
with tf.Session() as sess:
    
    # 创建初始化器，并初始化全部变量（权值和偏置）。
    init = tf.global_variables_initializer()
    init.run()
    
    # 一共执行 n_epochs 个 epoch 。
    for epoch in range(n_epochs):
        
        for iteration in range(train.num_examples // batch_size):
            
            # 每个 epoch 取 train.num_examples // batch_size 个 mini batch 。
            X_batch, y_batch = train.next_batch(batch_size)
            
            # 运行 train_op 一次，用一个 mini batch 的平均梯度更新一次权值和偏置。
            sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
        
        # 每一个 epoch 结束，用全部训练样本和测试样本计算当前网络在训练集和测试集上的准确率。
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: test.images, y: test.labels})
        print("epoch: {:d} train accuracy: {:.3f} test accuracy: {:.3f}".format(epoch, acc_train, acc_test))