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



print("\n\nWelcome...\n\n")
print("To find the document distance between " + sys.argv[1] + " and known malwares:")
print("----> Find BOW representations for " + sys.argv[1] + " and known malwares")
print("----> Find word embeddings for " + sys.argv[1] + " and known malwares")
print("----> Find cost of transforming every word in " + sys.argv[1] + " to every word in the known malwares")
print("----> Find document distance by applying Word Centroid Distance algorithm\n\n")






# file containing all words in our vocabulary
vocabulary_file = open("vocabulary", "r")
vocabulary = list(set(vocabulary_file.read().splitlines()))
vocabulary_size = len(vocabulary)

# malware whose document distance from classified malwares we wish to measure
test_malware = sys.argv[1]

# specifies whether there is a need to find embeddings or not
find_embeddings = sys.argv[2]

# classified/training malwares
malware_dir = "parsed_malwares"
embeddings_dir = "malware_embeddings"
malwares = []
for root, dirs, filenames in os.walk(malware_dir):
        malwares.append(test_malware)
        for malware in filenames:
                malwares.append(os.path.join(root, malware))






print("finding BOW representations...")
print("\n\n")


corpus = []
for malware in malwares:
        data = open(malware, "r").read()
        corpus.append(data)

pattern = re.compile('^[A-Za-z0-9]+(?:\\.[A-Za-z0-9]+)*', flags=re.M)
vectorizer = CountVectorizer(min_df=1, vocabulary=vocabulary, analyzer=partial(tokenizer, pattern=pattern))
vectorizer._validate_vocabulary()

X = vectorizer.fit_transform(corpus)
bows = X.toarray().astype(np.float32)

# from BOW to nBOW
for b in range(0, len(bows)):
        for i in range(0, len(bows[b])):
                bows[b][i] /= len(vocabulary)

# bows: m x n matrix of nbow vectors, where m is the number
# of malware classes + 1
#print(type(bows[0][0]))
print("nBOW non-zero elements:")
for bow in bows:
        print(len(np.nonzero(bow)[0]))
#        print(bow)
#        print("\n")
print("-----\n\n")







print("finding word embeddings...")
print("\n\n")

word_embeddings = []

# check flag
if find_embeddings == "-embeddings":
        malwares_for_embedding = malwares
else:
        if find_embeddings == "-no-embeddings":
                # only find embeddings for test file
                malwares_for_embedding = []
                malwares_for_embedding.append(malwares[0])

# find embeddings
for malware in malwares_for_embedding:
        # store name to save embeddings after generating them
        malware_name = malware[malware.index('/') + 1 :]
        
        # Step 1: Read the data into a list of strings.
        def read_data(filename):
                data = open(filename, "r").read().split()
                return data

        words = read_data(malware)
        
        
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
                                # dictionary['UNK']
                                index = 0  
                                unk_count += 1
                        data.append(index)
                count[0][1] = unk_count
                reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
                return data, count, dictionary, reverse_dictionary  

        data, count, dictionary, reverse_dictionary = build_dataset(words)
        # Hint to reduce memory.
        del words
#        print('Most common words (+UNK)', count[:5])
#        print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
#        print("-----\n")
        

        # Step 3: Function to generate a training batch
        data_index = 0
        def generate_batch(batch_size, num_skips, skip_window):
                global data_index
                assert batch_size % num_skips == 0
                assert num_skips <= 2 * skip_window
                batch = np.ndarray(shape=(batch_size), dtype=np.int32)
                labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
                # [ skip_window target skip_window ]
                span = 2 * skip_window + 1
                buffer = collections.deque(maxlen=span)
                for _ in range(span):
                        buffer.append(data[data_index])
                        data_index = (data_index + 1) % len(data)
                for i in range(batch_size // num_skips):
                        # target label at the center of the buffer
                        target = skip_window
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
        

        # Step 4: Build and train a skip-gram model.
        batch_size = 16
        skip_window = 10         # How many words to consider left and right.
        num_skips = 2           # How many times to reuse an input to generate a label.
        embedding_size = 128    # Dimension of the embedding vector.
        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        valid_size = 16         # Random set of words to evaluate similarity on.
        valid_window = 100      # Only pick dev samples in the head of the distribution.
        num_sampled = 64        # Number of negative examples to sample.
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        
        graph = tf.Graph()  
        global nce_weights
        with graph.as_default():
                # Input data.
                train_inputs = tf.placeholder(tf.int64, shape=[batch_size])
                train_labels = tf.placeholder(tf.int64, shape=[batch_size, 1])
                valid_dataset = tf.constant(valid_examples, dtype=tf.int64)
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
                optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
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
        num_steps = 50001
        with tf.Session(graph=graph) as session:
                # We must initialize all variables before we use them.
                init.run()
                print("Initialized")
                print("Finding word embeddings for " + malware)
                print("-----\n")
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
                np.savetxt(embeddings_dir + '/'+ malware_name + '.embeddings', final_embeddings)
                word_embeddings.append(final_embeddings)
                        
if find_embeddings == '-no-embeddings':
        for malware in malwares[1:]:
                malware_name = malware[malware.index('/') + 1 :]
                word_embeddings.append(np.loadtxt(embeddings_dir + '/' + malware_name + '.embeddings'))


#print("word_embeddings:")
#for embedding in word_embeddings:
#        print(embedding)
#        print("\n")
#print("-----\n\n\n\n")







print("finding word centroid distance...")
print("\n\n")


embedding_size = len(word_embeddings[0][0])
print("size(word_embeddings[0]) = " + str(len(word_embeddings[0])) + " x " + str(embedding_size))

dist_file = open("distances/distances.log", "a")
distances = []
for embedding in range(1, len(word_embeddings)):
        print("wcd between " + malwares[0] + " and " + malwares[embedding] + ":")
        j = 0
        k = 0
        sum_1 = 0.0
        sum_2 = 0.0
        centroid_1 = []
        centroid_2 = []
        for k in range(0, embedding_size):
                        for j in range(0, vocabulary_size):
                                sum_1 += bows[0][j] * word_embeddings[0][j][k]
                                sum_2 += bows[embedding][j] * word_embeddings[embedding][j][k]
                        centroid_1.append(sum_1)
                        centroid_2.append(sum_2)
                        sum_1 = 0.0
                        sum_2 = 0.0
        distance = 0.0
        for k in range(0, embedding_size):
                distance += pow((centroid_2[k] - centroid_1[k]), 2)
        distances.append(distance)
        
        print(str(distance))
        dist_file.write("wcd between " + malwares[0] + " and " + malwares[embedding] + "\n")
        dist_file.write(str(distance))
        dist_file.write("\n")





print("finding malware classification...")
print("\n\n")


min_distance = distances[0]
for distance in distances:
        if distance < min_distance:
                min_distance = distance

print("classification: ")
print(str(distances.index(min_distance) + 1))
dist_file.write("\nclassification\n")
dist_file.write(str(distances.index(min_distance) + 1))
dist_file.write("\n\n\n\n")

dist_file.close()

