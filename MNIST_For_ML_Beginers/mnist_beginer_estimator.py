import tensorflow as tf

Estimator = tf.estimator.Estimator
EstimatorSpec = tf.estimator.EstimatorSpec

def model_fn(features, labels, mode):
    # Define model
    x = features['x'] # Input, shape: (None, 784)
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    labels = tf.cast(labels, tf.float32)

    # Define loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(y), 1))

    # Chose gradient descent as optimizer
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train_step = optimizer.minimize(cross_entropy)
    train = tf.group(train_step, tf.assign_add(global_step, 1))

    return EstimatorSpec(mode = mode, predictions = y, 
                        loss = cross_entropy, train_op = train,
                        eval_metric_ops = {'accuracy': tf.metrics.accuracy(tf.argmax(y, 1), 
                                            tf.argmax(labels, 1))})


# Load data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Config input function
train_input_fn = tf.estimator.inputs.numpy_input_fn({'x': mnist.train.images}, mnist.train.labels, 
                                                    batch_size=128, num_epochs=None, shuffle=False)

test_input_fn = tf.estimator.inputs.numpy_input_fn({'x': mnist.test.images}, mnist.test.labels, 
                                                    batch_size=128, num_epochs=1, shuffle=False)

estimator = Estimator(model_fn)

# Training
estimator.train(train_input_fn, steps=1000)

# Evaluation
test_metrics = estimator.evaluate(input_fn=test_input_fn)
print("test metrics: %r"% test_metrics)
