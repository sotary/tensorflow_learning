# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:00:59 2018

@author: Administrator
"""

import tensorflow as tf
import numpy
# Parameters求解通过三个点的最小圆的圆心和半径
learning_rate = 0.1
training_epochs = 3000
display_step = 50
#为了保证分母不为0，加个常量
n_constant= tf.constant(0.000000000001)
# Training Data, 3 points that form a triangel
train_X = numpy.asarray([[0.98,0.90,0.92],[0.98,0.95,0.90],[0.97,0.93,0.88],[0.92,0.93,0.98]])
train_Y = numpy.asarray([[0.92,0.6,0.7],[0.91,0.7,0.56],[0.92,0.74,0.46],[0.94,0.74,0.66]])
# tf Graph Input
X = tf.placeholder("float",shape=[None, 3])
Y = tf.placeholder("float",shape=[None, 3])
# Set vaibale for center 这里人工定变量初始值为0.5,0.5,0
cx = tf.Variable(initial_value=[[0.5],[0.5],[0]], name="cx",dtype=tf.float32)
#相减
ctt= tf.subtract(tf.matmul(X,cx),tf.matmul(Y,cx))

c_sum=tf.reduce_sum(cx)
c_div=tf.div(cx,c_sum+n_constant)
ctt2= tf.subtract(tf.matmul(X,c_div),tf.matmul(Y,c_div))
distance = tf.negative(tf.subtract(tf.matmul(X,cx),tf.matmul(Y,cx)))
init = tf.global_variables_initializer()
# Start training
with tf.Session() as sess: 
       sess.run(init)
       c=sess.run(ctt,feed_dict={X: train_X, Y: train_Y})
       b=sess.run(c_div,feed_dict={X: train_X, Y: train_Y})
       d=sess.run(ctt2,feed_dict={X: train_X, Y: train_Y})
       print(c,b,d)