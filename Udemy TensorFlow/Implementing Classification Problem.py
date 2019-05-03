import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

"""
This Script shows how tensorflow would work on a classification problem
"""


def traverse_postorder(operation):
    """
    Makes sure computations are done in the correct order
    """
    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


class Operation():

    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

        # In Tensorflow there is a default graph. Here we add
        # our operation to the default graph
        _default_graph.operations.append(self)

    # Placeholder function for classes that inherit from Operation
    def compute(self):
        pass


#######################################################################
# Operations
#######################################################################
class add(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


class multiply(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var


class matmul(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var.dot(y_var)


class Sigmoid(Operation):

    def __init__(self, z):
        super().__init__([z])

    def compute(self, z_val):
        return 1 / (1 + np.exp(-z_val))


#######################################################################
# Placeholder
#######################################################################
class Placeholder():
    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)


#######################################################################
# Variable
#######################################################################
class Variable():
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        _default_graph.variables.append(self)


#######################################################################
# Placeholder
# Keeps track of all the variables, operations and placeholders
#######################################################################
class Graph():
    def __init__(self):
        self.operations = []
        self.variables = []
        self.placeholders = []

    def set_as_default(self):
        global _default_graph
        _default_graph = self


#######################################################################
# Session
# Keeps track of all the variables, operations and placeholders
#######################################################################
class Session:

    def run(self, operation, feed_dict={}):
        """
          operation: The operation to compute
          feed_dict: Dictionary mapping placeholders to input values (the data)
        """

        # Puts nodes in correct order
        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:

            if type(node) == Placeholder:

                node.output = feed_dict[node]

            elif type(node) == Variable:

                node.output = node.value

            else:  # Operation

                node.inputs = [input_node.output for input_node in node.input_nodes]

                node.output = node.compute(*node.inputs)

            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)

        # Return the requested node value
        return operation.output


#######################################################################
# Classification
# We now use our tf implementation for a classification task
#######################################################################
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Values from -10 to 10 with 100 samples to generate
sample_z = np.linspace(-10, 10, 100)
sample_a = sigmoid(sample_z)
# # Plot sigmoid function
# plt.plot(sample_z,sample_a)
# plt.show()

data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)

features = data[0]
labels = data[1]

# # Show generated data
# plt.scatter(features[:,0], features[:,1])
# plt.show()

# Line seperation the two data clusters we generated:
# (1,1) * f - 5 = 0
# Anything above this line belongs to one data cluster
# and anything below to the other

# # This is a representation of our equation
# res = np.array([1, 1]).dot(np.array([[8],[10]])) - 5
# print(res)

# The mathematical equation in words:
# 1. z = Compute our function above
#           w = weights
#           x = datapoints
#           b = intercept
# 2. put the outut of x in the sigmoid function
# If the point is beyond line => sigmoid is positive => data belongs to cluster one
# If the point is below line => sigmoid is negative => data belongs to cluster two
g = Graph()
g.set_as_default()
x = Placeholder()
w = Variable([1, 1])
b = Variable(-5)
z = add(matmul(w, x), b)
a = Sigmoid(z)
sess = Session()
print(sess.run(operation=a, feed_dict={x: [8, 10]}))
