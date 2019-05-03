import tensorflow as tf

# # Using version 1.12.0
# print(tf.__version__)

# Tensor is a n-dimensional array

hello = tf.constant("Hello")
world = tf.constant("World")

# # 'with' makes sure the session closes itself when
# # we are done
# with tf.Session() as sess:
#     result = sess.run(hello+world)
# print(result)

# a = tf.constant(10)
# b = tf.constant(20)
# with tf.Session() as sess:
#     result2 = sess.run(a+b)
# print(result2)

# a = tf.constant([ [1,2],
#                   [3,4]])
# b = tf.constant([ [10],
#                   [100]])
#
# result3 = tf.matmul(a,b)
# with tf.Session() as sess:
#     res = sess.run(result3)
# print(res)

#######################################################################
# Graphs Lesson
#######################################################################
# n1 = tf.constant(1)
# n2 = tf.constant(2)
# n3 = n1 + n2
# with tf.Session() as sess:
#     res= sess.run(n3)
# print(res)
#
# graph_1 = tf.get_default_graph()
# graph_2 = tf.Graph()
#
# with graph_2.as_default():
#     print(graph_2 is tf.get_default_graph())

#######################################################################
# Variables / Placeholders Lesson
#######################################################################
sess = tf.Session()
my_tensor = tf.random_uniform((4, 4), 0, 1)
my_var = tf.Variable(initial_value=my_tensor)

# All variables must me initialised first
init = tf.global_variables_initializer()
sess.run(init)
res = sess.run(my_var)
# print(res)
