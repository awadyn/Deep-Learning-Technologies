print("solving optimization problem...")
print("\n\n")

from cvxpy import *
import numpy as np

bow_1 = np.loadtxt("bow_1")
bow_2 = np.loadtxt("bow_2")
costs = np.loadtxt("costs")
print(type(bow_1[0]))
print(bow_1)
print("\n-----\n")
print(type(bow_2[0]))
print(bow_2)
print("\n-----\n")
print(type(costs[0][0]))
print(costs)
print("\n-----\n")

vocabulary_size = len(bow_1) 

T = Variable(vocabulary_size, vocabulary_size)
constraints = []

#constraints.append(sum_entries(T, axis=0).T == bow_1)
constraints.append(sum_entries(T, axis=1) == bow_2)
constraints.append(0 <= T)

variables = []
for i in range(0,vocabulary_size):
        for j in range(0,vocabulary_size):
                if( i == 0 and j == 0 ):
                        variables.append(T[i,j] * costs[i][j])
#                        objectiveFunction = T[i,j] * costs[i][j]
#                else:
#                        objectiveFunction += T[i,j] * costs[i][j]
objectiveFunction = sum(variables)

prob = Problem(Minimize(objectiveFunction), constraints)
prob.solve(verbose = True, solver = SCS)


print(type(T.value))
print(T.value)
print("-----\n\n\n\n")

np.savetxt("flow_matrix", T.value)
flow_matrix = np.loadtxt("flow_matrix")
print(type(flow_matrix[0][0]))
print(flow_matrix)

distance = 0
for i in range(0, vocabulary_size):
        for j in range(0, vocabulary_size):
                distance += flow_matrix[i][j] * costs[i][j]

print("wmd = " + str(distance))



