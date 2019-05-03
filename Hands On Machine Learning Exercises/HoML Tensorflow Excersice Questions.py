import tensorflow as tf

# 1. What are the main benefits of creating a computation graph rather than directly executing the
# computations? What are the main drawbacks?
# - Main benefits: Control and scalability.
# - Main drawbacks:

# 2. Is the statement a_val = a.eval(session=sess) equivalent to a_val = sess.run(a)?
# c1 = tf.constant(3)
# random_matrix = tf.random_uniform((2,2),-40,40,dtype='int32' )
#
# a = c1 * random_matrix
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     # a_val = a.eval(session=sess)
#     # print(a_val)
#     # a_val = sess.run(a)
#     # print(a_val)
# Apparently yes.

# 3. Is the statement a_val, b_val = a.eval(session=sess), b.eval(session=sess) equivalent to
# a_val, b_val = sess.run([a, b])?
# c1 = tf.constant(3)
# a = tf.random_uniform((2,2),-40,40,dtype='int32' )
# b = tf.random_uniform((2,2),-40,40,dtype='int32' )
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     # a_val, b_val = a.eval(session=sess), b.eval(session=sess)
#     # print(a_val)
#     # print(b_val)
#     a_val, b_val = sess.run([a, b])
#     print(a_val)
#     print(b_val)
# The output is the same but the graph is run twice in first version

# 4. Can you run two graphs in the same session?
# No

# 5. If you create a graph g containing a variable w, then start two threads and open a session in each
# thread, both using the same graph g, will each session have its own copy of the variable w or will it
# be shared?
# Variables allow concurrent read and write operations. The value read from a variable may change if it is concurrently updated. By default, concurrent assignment operations to a variable are allowed to run with no mutual exclusion.

# 6. When is a variable initialized? When is it destroyed
# A variable is created when you first run the tf.Variable.initializer operation for that variable in a session. It is destroyed when that tf.Session.close.

# 7. What is the difference between a placeholder and a variable?
# You use tf.Variable for trainable variables such as weights (W) and biases (B) for your model.
# Placeholders are used to feed training examples

# 8. What happens when you run the graph to evaluate an operation that depends on a placeholder but you
# don’t feed its value? What happens if the operation does not depend on the placeholder?
# c1 = tf.constant(3)
# c2 = tf.constant(3)
# p1 = tf.placeholder(tf.int32,[1,1])
#
# res1 = c1 * c2
# res2 = c1 * c2 * p1
# with tf.Session() as sess:
#     r1 = sess.run(res1)
#     print(r1)
# If it depends on a value there will be an InvalidArgumentError.

# 9. When you run a graph, can you feed the output value of any operation, or just the value of
# placeholders?
# c1 = tf.constant(3)
# c2 = tf.constant(3)
# p1 = tf.placeholder(tf.int32)
#
# res1 = c1 * c2
# res2 = c1 * c2 * p1
# with tf.Session() as sess:
#     r1 = sess.run(res2, feed_dict={p1: sess.run(res1)})
#     print(r1)