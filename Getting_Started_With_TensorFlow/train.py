import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

linear_model = W * x  + b

loss = tf.reduce_sum(tf.square(y - linear_model))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

import numpy as np

x_train = np.arange(1, 5)
y_train = np.arange(0, -4, -1)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for i in range(10000):
    sess.run(train, {x: x_train, y: y_train})

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))