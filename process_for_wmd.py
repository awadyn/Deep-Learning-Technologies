from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import math
import os
import sys
import random
import zipfile

from sklearn.feature_extraction.text import CountVectorizer
from functools import partial
from nltk import regexp_tokenize as tokenizer
import re

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf



print("\n\nWelcome...\n\n\n\n")
print("To find the document distance between " + sys.argv[1] + "and " + sys.argv[2] + ":")
print("----> Find BOW representations for " + sys.argv[1] + "and " + sys.argv[2])
print("----> Find word embeddings for " + sys.argv[1] + "and " + sys.argv[2])
print("----> Find cost of transforming every word in " + sys.argv[1] + "to every word in " + sys.argv[2])
print("----> Assemble and solve Word Mover's Distance optimization problem")
print("----> Find document distance by applying Word Mover's Distance algorithm\n\n\n\n")

print("Let us begin...\n\n\n\n")




# file containing all words in our vocabulary
vocabulary_file = open("vocabulary", "r")
vocabulary = list(set(vocabulary_file.read().splitlines()))
vocabulary_size = len(vocabulary)
# files whose document distance we wish to measure
files = [sys.argv[1], sys.argv[2]]

print("finding BOW representations...")
print("\n\n")

corpus = []
for filename in files:
    file = open(filename, "r")
    data = file.read()
    corpus.append(data)
print("corpus: ")
print(corpus)
print("-----\n")

vectorizer = CountVectorizer(min_df=1, vocabulary=vocabulary)
vectorizer._validate_vocabulary()

X = vectorizer.fit_transform(corpus)
bows = X.toarray().astype(np.float32)

# from BOW to nBOW
for b in range(0, len(bows)):
        for i in range(0, len(bows[b])):
                bows[b][i] /= len(vocabulary)

bow_1 = bows[0]
bow_2 = bows[1]

print(type(bows[0][0]))
print("bow_1:")
print(bow_1)
print("bow_2:")
print(bow_2)
print("-----\n\n\n\n")




print("finding word embeddings...")
print("\n\n")


word_embeddings = []
for file in files:
  
  # Read the data into a list of strings.
  def read_data(filename):
    file = open(filename, "r")
    data = file.read().split()
    return data
  words = read_data(file)
  
  
  # Step 2: Build the dictionary and replace rare words with UNK token.
  def build_dataset(words):
    count = [['UNK', -1]]
    most_common_size = len(collections.Counter(words).most_common(vocabulary_size - 1))
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
      if word in dictionary:
        index = dictionary[word]
      else:
        index = 0  # dictionary['UNK']
        unk_count += 1
      data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
  
  data, count, dictionary, reverse_dictionary = build_dataset(words)
  del words  # Hint to reduce memory.
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
  print("-----")
  
  
  
  data_index = 0
  # Step 3: Function to generate a training batch for t, reverse_dictionaryhe skip-gram model.
  def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
      buffer.append(data[data_index])
      data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
      target = skip_window  # target label at the center of the buffer
      targets_to_avoid = [skip_window]
      for j in range(num_skips):
        while target in targets_to_avoid:
          target = random.randint(0, span - 1)
        targets_to_avoid.append(target)
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[target]
      buffer.append(data[data_index])
      data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels
  
  batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
  for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
  
  # Step 4: Build and train a skip-gram model.
  
  embedding_size = 128  # Dimension of the embedding vector.
  batch_size = 16
  skip_window = 4       # How many words to consider left and right.
  num_skips = 2         # How many times to reuse an input to generate a label.
  
  # We pick a random validation set to sample nearest neighbors. Here we limit the
  # validation samples to the words that have a low numeric ID, which by
  # construction are also the most frequent.
  valid_size = 16     # Random set of words to evaluate similarity on.
  valid_window = 100  # Only pick dev samples in the head of the distribution.
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)
  num_sampled = 64    # Number of negative examples to sample.
  
  graph = tf.Graph()
  
  global nce_weights
  
  with graph.as_default():
  
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  
    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
      # Look up embeddings for inputs.
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)
  
      # Construct the variables for the NCE loss
      nce_weights = tf.Variable(
          tf.truncated_normal([vocabulary_size, embedding_size],
                              stddev=1.0 / math.sqrt(embedding_size)))
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
      print("\n\n\ntype(nce_weights)" + str(type(nce_weights)))
  
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))
  
    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
  
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
  
    # Add variable initializer.
    init = tf.global_variables_initializer()
  
  # Step 5: Begin training.
  num_steps = 60001
  
  with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")
  
    average_loss = 0
    for step in xrange(num_steps):
      batch_inputs, batch_labels = generate_batch(
          batch_size, num_skips, skip_window)
      feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
  
      # We perform one update step by evaluating the optimizer op (including it
      # in the list of returned values for session.run()
      _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
      average_loss += loss_val
  
      if step % 2000 == 0:
        if step > 0:
          average_loss /= 2000
        # The average loss is an estimate of the loss over the last 2000 batches.
        print("Average loss at step ", step, ": ", average_loss)
        average_loss = 0
  
    final_embeddings = normalized_embeddings.eval()
    word_embeddings.append(final_embeddings)

embeddings_1 = word_embeddings[0]
embeddings_2 = word_embeddings[1]
print(type(embeddings_1[0][0]))
print("embeddings_1")
print(embeddings_1)
print("embeddings_2")
print(embeddings_2)
print("-----\n\n\n\n")




print("finding cost of transformation...")
print("\n\n")


num_words_1 = len(embeddings_1)
num_words_2 = len(embeddings_2)
embedding_size = len(embeddings_1[0])
costs = np.ndarray(shape=(num_words_1, num_words_2), dtype=np.float32)

i = 0
j = 0
sum = 0.0
for i in range(0, num_words_1):
        for k in range(0, num_words_2):
                for j in range(0, embedding_size):
                        sum += pow((embeddings_2[i,j] - embeddings_1[k,j]), 2)
                costs[i,k] = sum
                sum = 0.0
        print("found costs[" + str(i) + "]")


print(type(costs[0][0]))
print("costs:")
print(costs)


np.savetxt("bow_1", bow_1)
np.savetxt("bow_2", bow_2)
np.savetxt("costs", costs)

#print("solving optimization problem...")
#print("\n\n")
#from cvxpy import *
#
#
#T = Variable(vocabulary_size, vocabulary_size)
#print(type(T))
#constraints = []
#
#constraints.append(sum_entries(T, axis=0).T == bow_1)
#constraints.append(sum_entries(T, axis=1) == bow_2)
#constraints.append(0 <= T)
#
#objectiveFunction = sum(T)
#
##for i in range(0,rows):
##        for j in range(0,cols):
##                if( i == 0 and j == 0 ):
##                        objectiveFunction = T[i,j] * costs[i,j]
##                else:
##                        objectiveFunction += T[i,j] * costs[i,j]
##
#
# 
#prob = Problem(Minimize(objectiveFunction), constraints)
#prob.solve(verbose = True, solver = SCS)
#
#
#print(type(T.value))
#print(T.value)
#print("-----\n\n\n\n")

#flow_matrix = np.loadtxt("sparse_flow_matrix")
#distance = 0
#for i in range(0,rows):
#        for j in range(0,cols):
##                print("cost = " + str(costs[i,j]))
##                print("flow_matrix(i,j) = " + str(flow_matrix[i,j]))
#                distance += flow_matrix[i,j] * costs[i,j]
#
#print("wmd = " + str(distance))

