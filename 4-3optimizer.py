#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 20:26:00 2018

@author: li
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


#定义训练的模型

  # Import data下载的样本会保存在/home/li/anaconda_project/MNIST_data下
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#print(mnist.train._images)

batch_size=100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
#print(n_batch)

x = tf.placeholder("float", shape=[None, 784],name='x')
y = tf.placeholder("float", shape=[None, 10],name='y')

keep_prob = tf.placeholder("float")
#学习率lr
lr = tf.Variable(0.001,"float")

#创建网络  W1服从正态分布
W1 = tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
b1 = tf.Variable(tf.zeros([500])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1,keep_prob)

W2 = tf.Variable(tf.truncated_normal([500,300],stddev=0.1))
b2 = tf.Variable(tf.zeros([300])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([300,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop,W3)+b3,name="final_result")

#二次代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#训练
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
#初始化
init = tf.global_variables_initializer()

#结果保存在一个布尔型列表中，argmax返回一个张量中最大值所在 的位置
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



#将预测结果的表达式放入collection
#tf.add_to_collection('pred_network', prediction)

#用saver 保存模型  
#saver = tf.train.Saver()   
  
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(2):
        #每次迭代周期，lr下降
        sess.run(tf.assign(lr,0.001*(0.95**epoch)))
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
            
        learning_rate = sess.run(lr)
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("Iter" +str(epoch)+",Testing Accuracy="+str(acc)+",Learning Rate="+str(learning_rate))
    saver = tf.train.Saver()     
    saver.save(sess, "./Model/4-3model")   
##将当前图设置为默认图  
#graph_def = tf.get_default_graph().as_graph_def()   
##将上面的变量转化成常量，保存模型为pb模型时需要,注意这里的final_result和前面的y_con2是同名，只有这样才会保存它，否则会报错，  
## 如果需要保存其他tensor只需要让tensor的名字和这里保持一直即可  
#output_graph_def = tf.graph_util.convert_variables_to_constants(sess,    
#                graph_def, ['final_result'])    
##保存前面训练后的模型为pb文件  
#with tf.gfile.GFile("grf.pb", 'wb') as f:    
#        f.write(output_graph_def.SerializeToString())  
#   
  



