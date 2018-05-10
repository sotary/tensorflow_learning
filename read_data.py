# -*- coding: utf-8 -*-
"""
Created on Thu May 10 10:07:31 2018

@author: Administrator
"""

import numpy
import pandas as pd
a = pd.read_csv('./data/data3.csv',header=None)
#b = numpy.loadtxt('../opencv_learning/data3.csv',delimiter=",")
d = numpy.load('./data/data_template.npy')
#c = numpy.array(a)
e = numpy.loadtxt('./data/zeros_tem_d.csv',delimiter=',')
k = numpy.array(a)
mu,sigma=0.5,0.2 #均值与标准差
rarray=numpy.random.normal(mu,sigma,(600,3))
rarray_col= rarray[:,0:1]
ones = numpy.ones((600,1))

zeros_col_1 = ones*0.55 + rarray[:,0:1]*0.2
zeros_col_2 = ones*0.05 + rarray[:,1:2]*0.09
zeros_col_3 = ones*0.02 + rarray[:,1:2]*0.08
#按列拼接，axis=1
zeros_col = numpy.concatenate((zeros_col_1[:,0:1],zeros_col_2[:,0:1],zeros_col_3[:,0:1]),axis=1)
numpy.savetxt('./data/zeros_tem_d_2.csv',zeros_col,delimiter=',')
zeros = numpy.ones((600,3))
zeros_8 = zeros*0.84 + rarray*0.14

numpy.savetxt('./data/zeros_g_2.csv',zeros_8,delimiter=',')