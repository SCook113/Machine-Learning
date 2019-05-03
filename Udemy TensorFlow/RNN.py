import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class TimeSeriesData():
    def __init__(self, num_points, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax - xmin) / num_points
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(selfself, x_series):
        return np.sin(x_series)

    def next_batch(self, batch_size, steps, return_batch_ts=False):

        # Grab random starting point for each batch
        rand_start = np.random.rand(batch_size, 1)
        # Convert to be on time series
        ts_start = rand_start * (self.xmax - self.xmin - (steps * self.resolution))
        # Create batch time series on the x axis
        batch_ts = ts_start + np.arange(0.0, steps + 1) * self.resolution
        # Create the Y data for the time series x axis from previous step
        y_batch = np.sin(batch_ts)
        # Formatting for RNN
        if return_batch_ts:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1), batch_ts
        else:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)


ts_data = TimeSeriesData(250, 0, 10)
# plt.plot(ts_data.x_data, ts_data.y_true)
# plt.show()

num_time_steps = 30
y1, y2, ts = ts_data.next_batch(1, num_time_steps, True)
# plt.plot(ts.flatten()[1:], y2.flatten(), '*')
# plt.show()

# Create the model
tf.reset_default_graph()
num_inputs = 1
num_neurons = 100
num_output = 1
learning_rate = 0.001
num_train_iterations = 2000
batch_size = 1

# PLACEHOLDERS
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_output])

# RNN Cell Layer
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_output)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# Loss function
loss = tf.reduce_mean(tf.square(outputs - y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

# Session
saver = tf.train.Saver()

# with tf.Session() as sess:
#     sess.run(init)
#
#     for iteration in range(num_train_iterations):
#         X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)
#         sess.run(train, feed_dict={X: X_batch, y: y_batch})
#
#         if iteration % 100 == 0:
#             mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
#             print(iteration, "\tMSE",mse)
#
#     saver.save(sess, "./models/rnn_time_series_model_codealong")

with tf.Session() as sess:
    saver.restore(sess, "./models/rnn_time_series_model_codealong")

    # SEED WITH ZEROS
    zero_seq_seed = [0. for i in range(num_time_steps)]
    for iteration in range(len(ts_data.x_data) - num_time_steps):
        X_batch = np.array(zero_seq_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        zero_seq_seed.append(y_pred[0, -1, 0])

# plt.plot(ts_data.x_data, zero_seq_seed, "b-")
# plt.plot(ts_data.x_data[:num_time_steps], zero_seq_seed[:num_time_steps], "r", linewidth=3)
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.show()

with tf.Session() as sess:
    saver.restore(sess, "./models/rnn_time_series_model_codealong")

    # SEED WITH Training Instance
    training_instance = list(ts_data.y_true[:30])
    for iteration in range(len(training_instance) -num_time_steps):
        X_batch = np.array(training_instance[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        training_instance.append(y_pred[0, -1, 0])
plt.plot(ts_data.x_data, ts_data.y_true, "b-")
plt.plot(ts_data.x_data[:num_time_steps],training_instance[:num_time_steps], "r-", linewidth=3)
plt.xlabel("Time")
plt.show()



