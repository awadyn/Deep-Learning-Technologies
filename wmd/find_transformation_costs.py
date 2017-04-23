import sys
import numpy as np

test_class = sys.argv[1]
train_class = sys.argv[2]


train_file = "malware_embeddings/malware_embeddings_" + train_class
test_file = "malware_test_embeddings/malware_test_embeddings_" + test_class

train = np.loadtxt(train_file)
test = np.loadtxt(test_file)

costs = np.ndarray(shape=(1680, 1680), dtype=float)

i = 0
j = 0
sum = 0

for i in range(0, 1680):
        for j in range(0, 128):
                sum = sum + pow((test[i][j] - train[i][j]), 2)
        costs[i][i] = sum
        sum = 0


np.savetxt("malware_costs/malware_costs_" + test_class + "_" + train_class, costs)

