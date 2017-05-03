import sys
import os

import gensim
import nltk


malware_test_doc = sys.argv[1]
malware_test_words = open(malware_test_doc, 'r').read().splitlines()


malware_train_docs = dict()
for root, dirs, filenames in os.walk('parsed_train_malwares'):
        for malware in filenames:
                malware_train_docs[malware] = open(os.path.join(root, malware), 'r').read().splitlines()                
for malware_class in range(1, 10):
        malware_dir = 'parsed_malwares/' + str(malware_class)
        for root, dirs, filenames in os.walk(malware_dir):
                for malware in filenames:
                        malware_train_docs[malware] = open(os.path.join(root, malware), 'r').read().splitlines()


vocabulary = []
vocabulary.append(malware_test_words)
for malware in malware_train_docs:
        vocabulary.append(malware_train_docs[malware])


model = gensim.models.Word2Vec(vocabulary, min_count=1)


out_file = open('results/' + malware_test_doc[malware_test_doc.index('/')+1 :], 'a')
out_file.write('measuring distance between ' + malware_test_doc[malware_test_doc.index('/')+1 :] + ' and instances of 9 malware classes...\n')
out_file.write('size of ' + malware_test_doc + '  =  ' + str(len(malware_test_words)) + '\n\n')


average_distances = []
for malware_class in range(1, 10):
        malware_train_dir = 'parsed_malwares/' + str(malware_class)
        sum = 0.0
        for root, dirs, filenames in os.walk(malware_train_dir):
                for malware in filenames:
                        distance = model.wmdistance(malware_train_docs[malware], malware_test_words)
#                        print(malware_test_doc + '  -->  ' + malware + '  =  ' + str(distance))
                        sum = sum + distance
        if len(filenames) == 0:
                average = float("inf")
        else:
                average = sum / len(filenames)
        average_distances.append(average)
        print('average  =  ' + str(average))
        out_file.write('class ' + str(malware_class) + ' average distance  =  ' + str(average) + '\n')


min_distance = average_distances[0]
min_index = 0
for index, distance in enumerate(average_distances):
        if distance <= min_distance:
                min_distance = distance
                min_index = index


print("classification  =  " + str(min_index + 1))
out_file.write('\nclassification  =  ' + str(min_index + 1))
out_file.close()

