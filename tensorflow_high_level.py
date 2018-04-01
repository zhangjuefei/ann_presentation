# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:37:45 2018

@author: zhangjuefei
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 获取手写数字
mnist = input_data.read_data_sets("D:\documents\ANN_presentation\charts\data")

# 训练集和测试集
train = mnist.train
test = mnist.test

# 定义一些与网络结构和训练有关的常量
n_inputs = 28 * 28
n_hidden_1 = 300
n_hidden_2 = 100
n_outputs = 10
n_epochs = 30
batch_size = 50
learning_rate = 0.01

steps = int(n_epochs * train.num_examples / batch_size)
print("steps: {:d}".format(steps))

# 数据哪系列作为特征列：这里全部实数列都做特征
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(train.images)

# 优化器：随机梯度下降，学习率：0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# 构建一个神经网络，有两个隐藏层，分别有 300 和 100 个神经元，激活函数是 tanh 。
dnn = tf.contrib.learn.DNNClassifier(
    hidden_units=[n_hidden_1, n_hidden_2],
    activation_fn=tf.nn.tanh,
    n_classes=n_outputs,
    optimizer=optimizer,
    feature_columns=feature_columns
)

# 训练，mini batch size:50 ，迭代数 10000（每一个样本参与一次训练算一次迭代）
dnn.fit(x=train.images, y=train.labels.astype("int"), batch_size=n_epochs, steps=steps)

# 预测以及正确率。
predict = list(dnn.predict(test.images))
print(dnn.evaluate(x=test.images, y=test.labels.astype("int")))
