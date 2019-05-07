import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(101)
tf.set_random_seed(101)

#######################################################################
# Example Neural Network
#######################################################################
#
# # Number of features
# n_features = 10
#
# # How many neurons in a layer
# n_dense_neurons = 3
#
# # Shape = None (We don't know yet)
# # Initialises a placeholder with shape: (???, n_features)
# x = tf.placeholder(tf.float32,(None, n_features))
#
# # Weights
# W = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))
#
# # Bias term
# b = tf.Variable(tf.ones([n_dense_neurons]))
#
# xW = tf.matmul(x,W)
# z = tf.add(xW,b)
#
# # Activation function
# a = tf.nn.sigmoid(z)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     layer_out = sess.run(a, feed_dict={x: np.random.random([1, n_features])})
# print(layer_out)

#######################################################################
# Simple regression example
#######################################################################
# Create noisy data with linear form
x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

# Plot the data
# plt.plot(x_data,y_label, '*')
# plt.show()

# Form of our function
# y = mx +b
# Initialise m and b to random numbers
m = tf.Variable(0.66)
b = tf.Variable(0.19)

# Our error function to minimize
# Initialize first
error = 0

# For every value in
for x, y in zip(x_data, y_label):
    # y_hat ist the prediction we make with your linear function
    # So for every point x we make a prediction
    y_hat = m * x + b
    # Here we set the cost function. It has the form of:
    # (y1-y1_hat)**2 + (y2-y2_hat)**2 + (y3-y3_hat)**2 + (y4-y4_hat)**2 + ...
    # This is the function we want to minimize
    error += (y - y_hat) ** 2

# As optimizer we choose gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# The optimizer only computes one step when we run it
# We tell the optimizer what to optimize
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    epochs = 100

    # We optimize 100 times
    for i in range(epochs):
        sess.run(train)

    # Fetch Back Results
    final_slope, final_intercept = sess.run([m, b])

# Plot the function with optimized values
print("Slope: ", final_slope)
print("Intercept: ", final_intercept)
x_test = np.linspace(-1, 11, 10)
y_pred_plot = final_slope * x_test + final_intercept
plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')
plt.show()
