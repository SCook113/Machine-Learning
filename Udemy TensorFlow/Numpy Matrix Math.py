import numpy as np

print("Matrix Multiplication")
#################################################
# Matrix Multiplikation
#################################################
# b = np.matrix(((5,5,5)))
# # Es wird immer Zeile * Spalte multipliziert
# x = np.matrix(((1,2,3)))
# # Representiert:
# # 1*th1 + 2*th2 + 3*th3
# #
# # Anders ausgedrückt: Input Werte für x
# W = np.matrix(((1,2,3),(1,2,3),(1,2,3)))
# # Representiert 3 Gleichungen.
# # Jede Spalte ist eine Gleichung:
# # Gleichung 1 :
# # x1*1 + x2*1 + x3*1
# # Gleichung 2 :
# # x1*2 + x2*12+ x3*2
# # Gleichung 3 :
# # x1*3 + x2*3 + x3*3
#
# # x * W representiert was dabei rauskommt wenn ich in jede
# # Gleichung in W die zahlen von x eingebe und die 3 gleichungen ausrechne
# # print("Nur ein Input", x * W)
# # = [[ 6 12 18]]
# # Hier hatte ich nur einen 'Input' für x
#
# # Wenn ich in x eine Zeile hinzufüge bekomme ich das ergebnis
# # erst von den drei gleichungen mit 'input 1' und dann
# # das Ergebnis von 'Input 2'
# # xx = np.matrix(((1,2,3),(10,10,10)))
# # print("Zwei Input", xx * W)
#
# # print(xx * W )
#
# # Representieren von einem NN mit einem layer und 3 neuronen
#
# input = np.matrix(((1,2,3)))
#
# weights_layer_1 = np.matrix(((1,2,3),(1,2,3),(1,2,3)))
# # Neuron 1 = 1x + 1x + 1x
# # Neuron 2 = 2x + 2x + 2x
# # Neuron 1 = 3x + 3x + 3x
# # Jede Spalte ist ein Neuron
#
# output_layer_1 = (input * weights_layer_1)
#
# # print(output_layer_1)
# # [[ 6 12 18]
# last_layer = np.matrix(('2 ; 2; 2'))
#
# ergebnis = output_layer_1 * last_layer
#
print("End of Matrix Multiplication")

print("Matrix and Vektors")
#################################################
# Matrix * Vector
#################################################
# M = np.array(((1,2,3),(1,2,3),(1,2,3)))
# # print(M)
# x = np.array(((3,3,3)))
# # print(x)
# # With the dot product i compute everything as befor
# print(x.dot(M))
# # With a simple multiplikation i multiply every column with the vactor
# print(x+x)
# print('Add:',M+x)
print("End Matrix and Vektors")

print("Kernel Calculations")
#################################################
# Calculate ouput size of conv net
#################################################
# # Input width
# i_w = 32
# i_H = 32
# k_w = 5
# k_h = 5
# S = 1
# P = 0
# O_w = ((i_w - k_w + 2 * P) / S) + 1
# O_h = ((i_H - k_h + 2 * P) / S) + 1
# print(O_w, O_h)
print("End Kernel Calculations")

# print(np.ones((5,5,1,32)))
# np.ones((3,2)) 3 zeilen, 2 spalten
# np.ones((3,2,4)) 3 matrizen mit form 2 zeilen, 4 salten
# np.ones((3,2,4,2))

print("Example Calculation")
#################################################
# Example Calculation
#################################################
# x_flattened = np.ones((1,30000))
# gewichte = np.ones((30000, 50))
# bias = np.ones((1, 50))
# print(x_flattened.dot(gewichte) + bias)
print("End of Example Calculation")

print("Example Crossproduct")
#################################################
# Example Calculation
#################################################
# x = np.array([3,6,1])
# crossproduct = np.cross(x,x)
# print(crossproduct) # [0 0 0]
# print(np.sqrt(crossproduct.dot(crossproduct))) # Länge = 0
print("End of Example Crossproduct")

print("Example Cosine Similarity")
#################################################
# Example Calculation
#################################################
from numpy import dot
from numpy.linalg import norm

x = np.array([0.1, 0, 0, 0.1, 0.1])
y = np.array([0, 0.1, 0.1, 0, 0])
cos_sim = dot(x, y) / (norm(x) * norm(y))
print(cos_sim)  # Ergibt 0
print("End of Example Cosine Similarity")
