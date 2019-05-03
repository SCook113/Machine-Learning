import numpy as np
"""
This Script is a (very simple) implementation how the basics of tensorflow works under the hood.
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

    def __init__(self,x,y):
        super().__init__([x,y])

    def compute(self,x_var,y_var):
        self.inputs = [x_var,y_var]
        return x_var + y_var

class multiply(Operation):

    def __init__(self,x,y):
        super().__init__([x,y])

    def compute(self,x_var,y_var):
        self.inputs = [x_var,y_var]
        return x_var * y_var


class matmul(Operation):

    def __init__(self,x,y):
        super().__init__([x,y])

    def compute(self,x_var,y_var):
        self.inputs = [x_var,y_var]
        return x_var.dot(y_var)

#######################################################################
# Placeholder
#######################################################################
class Placeholder():
    def __init__(self):
        self.output_nodes=[]
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

# # Show how simple computation works
# g = Graph()
# g.set_as_default()
# sess = Session()
# A = Variable(10)
# b = Variable(1)
# x = Placeholder()
# y = multiply(A,x)
# z = add(y,b)
# result = sess.run(operation=z, feed_dict={x : 10})
# print(result)

# A computation using matrices
g = Graph()
sess = Session()
g.set_as_default()
A = Variable([[10,20],[30,40]])
b = Variable([1,1])
x = Placeholder()
y = matmul(A,x)
z = add(y,b)
result = sess.run(operation=z, feed_dict={x:10})
print(result)
