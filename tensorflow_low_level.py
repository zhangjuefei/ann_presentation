# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:49:37 2018

@author: zhangjuefei
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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
activation_fn_hidden1 = tf.nn.relu
activation_fn_hidden2 = tf.nn.relu

# 清除运算图。
tf.reset_default_graph()

# 定义占位变量。
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None,), name="y")

# 创建神经网络各层的权值矩阵和偏置向量。本神经网络的形状是 28*28->300->200->10->softmax 。
with tf.name_scope("weights_and_biases"):
    # 第一层权值矩阵 784x300 ，偏置向量 300 。初始化为小绝对值的值。
    hidden1_weights = tf.Variable(initial_value=tf.random_uniform([n_inputs, n_hidden_1], -0.1, 0.1), dtype=tf.float32,
                                  name="hidden1_weights")
    hidden1_biases = tf.Variable(initial_value=tf.random_uniform([n_hidden_1], -0.1, 0.1), dtype=tf.float32,
                                 name="hidden1_biases")

    # 第二层权值矩阵 300x100 ，偏置向量 100 。
    hidden2_weights = tf.Variable(initial_value=tf.random_uniform([n_hidden_1, n_hidden_2], -0.1, 0.1),
                                  dtype=tf.float32, name="hidden2_weights")
    hidden2_biases = tf.Variable(initial_value=tf.random_uniform([n_hidden_2], -0.1, 0.1), dtype=tf.float32,
                                 name="hidden2_biases")

    # 输出层权值矩阵 100x10 ，偏置向量 10 。
    output_weights = tf.Variable(initial_value=tf.random_uniform([n_hidden_2, n_outputs], -0.1, 0.1), dtype=tf.float32,
                                 name="output_weights")
    output_biases = tf.Variable(initial_value=tf.random_uniform([n_outputs], -0.1, 0.1), dtype=tf.float32,
                                name="output_biases")

# 定义网络的计算图。
with tf.name_scope("network_flow"):
    # 第一层的计算。
    output_hidden1 = activation_fn_hidden1(tf.nn.bias_add(tf.matmul(X, hidden1_weights), hidden1_biases),
                                           name="hidden1")

    # 二层的计算。
    output_hidden2 = activation_fn_hidden2(tf.nn.bias_add(tf.matmul(output_hidden1, hidden2_weights), hidden2_biases),
                                           name="hidden2")

    # 输出层的计算，无激活函数（或者说激活函数是恒等函数）。
    logits = tf.nn.bias_add(tf.matmul(output_hidden2, output_weights), output_biases, name="logits")

# 网络损失函数。   
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
