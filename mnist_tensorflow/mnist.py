import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle

learning_rate = 0.05
epochs = 500
batch_size = 100
total_size = 42000

mnist_train = pd.read_csv('input/mnist/train.csv')
mnist_test = pd.read_csv('input/mnist/test.csv')



# preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

W1 = tf.Variable(tf.random_normal([784,300],stddev=0.03),name="W1")
b1 = tf.Variable(tf.random_normal([300]),name="b1")

W2 = tf.Variable(tf.random_normal([300,10],stddev=0.03),name="W2")
b2 = tf.Variable(tf.random_normal([10]),name="b2")

hidden_out = tf.nn.relu(tf.add(tf.matmul(x,W1),b1))

y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out,W2),b2))

y_clipped = tf.clip_by_value(y_,1e-10,0.9999999)

cross_entropy = -tf.reduce_mean(tf.reduce_sum(y*tf.log(y_clipped)+(1-y) * tf.log(1-y_clipped),axis=1))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

init_op = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = total_size//batch_size
    for epoch in range(epochs):
        avg_cost = 0
        mnist_train = shuffle(mnist_train)
        batch_y = pd.get_dummies(mnist_train.ix[:, 0]).values
        batch_x = mnist_train.ix[:, 1:mnist_train.shape[1]].values
        batch_x = min_max_scaler.fit_transform(batch_x)

        for i in range(total_batch):
            _,c = sess.run([optimiser,cross_entropy],feed_dict={x:batch_x[i*batch_size:(i+1)*batch_size],y:batch_y[i*batch_size:(i+1)*batch_size]})
            avg_cost += c/total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))

    test_batch_x = mnist_test.values
    test_batch_x = min_max_scaler.fit_transform(test_batch_x)

    results = sess.run(y_clipped,feed_dict={x:test_batch_x})
    print("Training Completed")
    output = []
    for i in range(len(results)):
        output.append([i+1,results[i].argmax()])

    a = np.array(output)
    p = pd.DataFrame(a)
    p.to_csv("mnist_output.csv")




