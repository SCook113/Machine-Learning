import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# This script tries to solve the
# mnist problem with one manually created layer

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])

# Variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# Create graph operations
y = tf.matmul(x, W) + b
# Loss function
y_true = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)
# Create session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # Initialize the variables
    sess.run(init)
    # Train the model for 1000 steps on the training set
    # Using built in batch feeder from mnist for convenience
    for step in range(1000):
        # Get next batch
        batch_x, batch_y = mnist.train.next_batch(100)
        # Run gradient descent on the cost function (cross entropy) for every batch
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})
    # Test the Train Model
    matches = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    acc = tf.reduce_mean(tf.cast(matches, tf.float32))
    print(sess.run(acc, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))
