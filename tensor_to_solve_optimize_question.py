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
display_step = 50
# Training Data, 3 points that form a triangel
train_X = numpy.asarray([3.0,6.0,9.0])
train_Y = numpy.asarray([7.0,9.0,7.0])
# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")
# Set vaibale for center 这里人工定变量初始值为3
cx = tf.Variable(3, name="cx",dtype=tf.float32)
cy = tf.Variable(3, name="cy",dtype=tf.float32)
# Caculate the distance to the center and make them as equal as possible
distance = tf.pow(tf.add(tf.pow((X-cx),2),tf.pow((Y-cy),2)),0.5)
mean = tf.reduce_mean(distance)
cost = tf.reduce_sum(tf.pow((distance-mean),2)/3)
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
              if (c - 0) < 0.0000000001:  break 
       #Display logs per epoch step 
              if (epoch+1) % display_step == 0: 
                     c = sess.run(cost, feed_dict={X: train_X, Y:train_Y}) 
                     m = sess.run(mean, feed_dict={X: train_X, Y:train_Y}) 
                     print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), "CX=", sess.run(cx), "CY=", sess.run(cy), "Mean=", "{:.9f}".format(m))
                     print ("Optimization Finished!")
                     training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y}) 
                     print ("Training cost=", training_cost, "CX=", round(sess.run(cx),2), "CY=", round(sess.run(cy),2), "R=", round(m,2))