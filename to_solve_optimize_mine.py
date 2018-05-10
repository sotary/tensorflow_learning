# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:07:58 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 08:07:53 2018

@author: Administrator
"""


import tensorflow as tf
import numpy
# Parameters求解通过三个点的最小圆的圆心和半径
learning_rate = 0.1
training_epochs = 3000
display_step = 100
#为了保证分母不为0，加个常量
n_constant= tf.constant(0.000000000001)
# Training Data, 3 points that form a triangel
train_X = numpy.asarray([[0.98,0.90,0.92],[0.98,0.95,0.90],[0.97,0.93,0.88],[0.92,0.93,0.98]])
train_Y = numpy.asarray([[0.73,0.11,0.01],[0.88,0.01,0.006],[0.72,0.006,0.06],[0.74,0.04,0.016]])
# tf Graph Input
X = tf.placeholder("float",shape=[None, 3])
Y = tf.placeholder("float",shape=[None, 3])
# Set vaibale for center 这里人工定变量初始值为0.2,0.5,0.3
cx = tf.Variable(initial_value=[[0.2],[0.5],[0.3]], name="cx",dtype=tf.float32)
c_sum=tf.reduce_sum(tf.abs(cx))
c_div=tf.div(tf.abs(cx),c_sum+n_constant)
# Caculate the distance to the center and make them as equal as possible
distance = tf.abs(tf.subtract(tf.matmul(X,c_div),tf.matmul(Y,c_div)))
# x 为要传递的tensor,axes是个int数组,传递要进行计算的维度,返回值是两个张量: mean and variance,
mean1,variance = tf.nn.moments(tf.matmul(X,c_div),[0])
#方差除以距离，是我们的目标函数  mean
mean = tf.div(variance,distance)
#reduc_sum是将数据降维
cost = tf.reduce_sum(mean,0)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# Start training
with tf.Session() as sess: 
       sess.run(init) 
# Fit all training data 
       for epoch in range(training_epochs): 
              sess.run(optimizer, feed_dict={X: train_X, Y: train_Y}) 
              c = sess.run(cost, feed_dict={X: train_X, Y:train_Y}) 
             # if (c.all - 0) < 0.0000000001:  break 
       #Display logs per epoch step 
              if (epoch+1) % display_step == 0: 
                     c = sess.run(cost, feed_dict={X: train_X, Y:train_Y}) 
                     m = sess.run(mean, feed_dict={X: train_X, Y:train_Y}) 
                     print ("Epoch:", '%04d' % (epoch+1), "CX=", sess.run(cx),"cost=",c)
                     print ("Optimization Finished!")
                     training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y}) 
                     print ("Training cost=", training_cost, "C_div=", sess.run(c_div),  "R=", m)