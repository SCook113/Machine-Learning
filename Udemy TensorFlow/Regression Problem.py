import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# We will not split data in to train/test/split data in this tutorial,
# this is only for training purposes

# Seed are set in numpy and in tf
np.random.seed(101)
tf.set_random_seed(101)

x_data = np.linspace(0.0, 10.0, 1000000)
# Make data points noisy
noise = np.random.randn(len(x_data))

# This is the form of my function
# y = mx + b
# b = 5
y_true = (0.5 * x_data) + 5 + noise

# Here we make our data
x_df = pd.DataFrame(data=x_data, columns=['X_Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y_Data'])

my_data = pd.concat([x_df, y_df], axis=1)

# # Let's look at random samples of the data frame
# # x, y refer to the columns in the dataframe
# my_data.sample(200).plot(kind='scatter', x='X_Data', y='Y_Data')
# plt.show()

# Let's find the line that fits the data
batch_size = 8

# Initialize variables
m = tf.Variable(0.81)
b = tf.Variable(0.17)
# Initialize placeholders
xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])
# Initializse model
y_model = m * xph + b
# Initialize error function
error = tf.reduce_sum(tf.square(yph - y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batches = 1000

    for i in range(batches):
        # Grab 8 random indexes
        rand_ind = np.random.randint(len(x_data), size=batch_size)
        # Feed 8 random points to the graph
        feed = {xph: x_data[rand_ind], yph: y_true[rand_ind]}

        sess.run(train, feed_dict=feed)
    # Get back the calculated values for m and b
    model_m, model_b = sess.run([m, b])
    print(model_m, model_b)

y_hat = x_data * model_m + model_b
my_data.sample(200).plot(kind='scatter', x='X_Data', y='Y_Data')
plt.plot(x_data, y_hat, 'r')
plt.show()
