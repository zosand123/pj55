# 선형회귀:지도학습에서 수치를 예측
# h=wx+b  (예)cars.csv  x=speed
# h=w1x1+w2x2+b  -->x X w + b
# ...
# water.csv 파일은 총 정수기 대여 대수(전월),
# 10년이상 노후 정수기 대여 대수(전월), AS시간(당월)에 대한
# 데이터이다. 주변의 신규 아파트가 동시 입주함에 따라 가입자수가 늘어
# 다음달의 AS시간을 예측하고 이에 따라 신규인력을 채용하고자 한다.

import tensorflow as tf
import numpy as np
# data=np.loadtxt('data\\water.csv',delimiter=',',skiprows=1)
# # print(data)
# xdata=data[:,1:-1 ]
# ydata=data[:,-1:]
# # print(xdata)
# # print(ydata)
# x=tf.placeholder(tf.float32,[None,2])
# y=tf.placeholder(tf.float32,[None,1])
# w=tf.Variable(tf.random_normal([2,1]))
#    # [None, 2] X[2, 1] = [None, 1]
# b=tf.Variable(tf.random_normal([1]))
# h=tf.matmul(x,w)+b
# cost=tf.reduce_mean(tf.square(y-h))
# train=tf.train.GradientDescentOptimizer(0.00000000003).minimize(cost)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(4001):
#         sess.run(train,feed_dict={x:xdata,y:ydata})
#         if i%100==0:
#             print(sess.run(cost,feed_dict={x:xdata,y:ydata}))
#     print('w=',sess.run(w))
#     print('b=',sess.run(b))
# # 0.00000000005
# # 0.00000000003 1692.4844
# # 0.00000000001 14832.939
# # 0.000000000001 677545.4
# # 월말 최종 대여수를 보니 총 대여수가 300,000대, 그중 10년 이상
# # 노후 정수기 대수가 70,000대로 집계되었다.
# # 다음달의 AS시간을 예측하고 그에 따라 필요한 AS기사의 인원수를 예측하시오
#     result=sess.run(h,feed_dict={x:[[70000,230000]]})
#     print('예측값=',result[0][0])
#     print(result[0][0]/160)
# --------------------------------
# https://archive.ics.uci.edu/ml/machine-learning-databases/zoo
# one_hot(데이터,7) -->0~6, 다중분류
data=np.loadtxt('data\\zoo\\zoo.csv',delimiter=',')
# print(data.shape)   #(101, 17)
xdata=data[:,:-1]
ydata=data[:,-1:]
# print(xdata.shape)   #(101, 16)  데이터
# print(ydata.shape)   #(101, 1)   정답
x=tf.placeholder(tf.float32,[None,16])
y=tf.placeholder(tf.int32,[None,1])   #0-6  ==>onehot
onehot=tf.one_hot(y,7)
onehot2=tf.reshape(onehot,[-1,7])
w=tf.Variable(tf.random_normal([16,7]))
# [None,16] X [16,7] = [None,7]
b=tf.Variable(tf.random_normal([7]))
logits=tf.matmul(x,w)+b
h=tf.nn.softmax(logits)   #다중분류
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=onehot2))
train=tf.train.GradientDescentOptimizer(0.7).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(x,feed_dict={x:xdata}))
    # print(sess.run(y,feed_dict={y:ydata}))
    # print('-'*30)
    # print(sess.run(onehot,feed_dict={y:ydata}))
    # print(sess.run(onehot2,feed_dict={y:ydata}))
    # print('학습전 예측값=\n',sess.run(h,feed_dict={x:xdata}))
    # print(sess.run(tf.argmax(h,1),feed_dict={x:xdata}))
    # print(sess.run(tf.equal(tf.argmax(h,1),tf.argmax(onehot2,1)),
    #                feed_dict={x:xdata,y:ydata}))
    # print(sess.run(tf.cast(tf.equal(tf.argmax(h,1),tf.argmax(onehot2,1)),tf.float32),
    #          feed_dict={x:xdata,y:ydata}))
    # print('학습전 정확도=',
    # sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(h,1),tf.argmax(onehot2,1)),tf.float32)),
    #          feed_dict={x:xdata,y:ydata}))
    for i in range(2001):
        sess.run(train,feed_dict={x:xdata,y:ydata})
        if i%100==0:
            print(sess.run(cost,feed_dict={x:xdata,y:ydata}))
   # 평가하기
    correct=tf.equal(tf.argmax(h,1),tf.argmax(onehot2,1))  #True,False,...
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
    print('정확도=',sess.run(accuracy,feed_dict={x:xdata,y:ydata}))
    #예측하기
    print(sess.run(h,feed_dict={x:[[0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0]]}))
    print(sess.run(tf.argmax(h,1),
                   feed_dict={x:[[0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0]]}))
print('\n\n\n\n\n\n\n')