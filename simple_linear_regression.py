# simple linear regression using tensorflow core library

import numpy as np
import tensorflow as tf


# define model parameters
w = tf.Variable([0.1], tf.float32)
b = tf.Variable([0.1], tf.float32)

# define model input and output placeholders
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# define model
linear_model = w * x + b

# define loss function: use mean squared error loss
loss = tf.reduce_sum(tf.square(linear_model - y))

# define optimizer with update constant
optimizer = tf.train.GradientDescentOptimizer(0.01)



# assign training data
x_train = [1, 2, 3, 4]
y_train = [-1, -2, -3, -4]

# assign optimizer for this model
train = optimizer.minimize(loss)



# begin session
# initialize variables w and b to their initial incorrect values
session_1 = tf.Session()
session_1.run(tf.global_variables_initializer())

# execute training loop
for i in range(1000):
        session_1.run(train, {x:x_train, y:y_train})

# evaluate accuracy
new_w, new_b = session_1.run([w, b])
new_loss = session_1.run(loss, {x:x_train, y:y_train})
print("w: %s, b: %s, loss: %s" %(new_w, new_b, new_loss))
