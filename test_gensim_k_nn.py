import sys
import os

import gensim
import nltk
import pickle

from collections import Counter


print('\n\nbuilding model...')
malware_train_docs = dict()
for root, dirs, filenames in os.walk('parsed_train_malwares'):
        for malware in filenames:
                malware_train_docs[malware] = open(os.path.join(root, malware), 'r').read().splitlines()                



vocabulary = []
for malware in malware_train_docs:
        vocabulary.append(malware_train_docs[malware])



model = gensim.models.Word2Vec(vocabulary, min_count=1)



print('\n\ncalculating distances...\n')
distances = dict()
malware_train_dir = 'parsed_train_malwares'
for root, dirs, filenames in os.walk(malware_train_dir):
        for test_malware in filenames:
                malware_test_doc = test_malware
                malware_test_words = open(os.path.join(root, malware_test_doc), 'r').read().splitlines()
                distances[test_malware] = dict()
                train_malwares = filenames[:]
                train_malwares.remove(test_malware)
                for malware in train_malwares:
                        distance = model.wmdistance(malware_train_docs[malware], malware_test_words)
                        distances[test_malware][malware] = distance
                print('finished calculating distances from train malwares to ' + test_malware)



# write distances dictionary to file
with open('distances_10.pickle', 'wb') as handle:
        pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)



print('\n\nfinding nearest neighbors...\n')
for root, dirs, filenames in os.walk(malware_train_dir):
        for test_malware in filenames:
                out_file = open('results/' + test_malware, 'a')
                
                sorted_distances = sorted((distances[test_malware]).values())
                closest_malwares_10 = []
                closest_classes_10 = []
                closest_malwares_20 = []
                closest_classes_20 = []
                closest_malwares_30 = []
                closest_classes_30 = []
                for i in range(0, 10):
                        for malware, distance in distances[test_malware].items():
                                if distance == sorted_distances[i]:
                                        closest_malwares_10.append(malware)
                                        closest_classes_10.append(malware[malware.index('_') + 6 : malware.index('_') + 7])
                counter = Counter(closest_classes_10)
                closest_class = counter.most_common()[0][0]
                print('\n10 closest malwares to ' + test_malware + ' :\n')
                print(counter.most_common())
                print('10_NN classification  =  ' + closest_class)
                print('\n')
                out_file.write('10_NN classification  =  ' + closest_class + '\n')
                
                for i in range(0, 20):
                        for malware, distance in distances[test_malware].items():
                                if distance == sorted_distances[i]:
                                        closest_malwares_20.append(malware)
                                        closest_classes_20.append(malware[malware.index('_') + 6 : malware.index('_') + 7])
                counter = Counter(closest_classes_20)
                closest_class = counter.most_common()[0][0]
                print('\n20 closest malwares to ' + test_malware + ' :\n')
                print(counter.most_common())
                print('20_NN classification  =  ' + closest_class)
                print('\n')
                out_file.write('20_NN classification  =  ' + closest_class + '\n')

                for i in range(0, 30):
                        for malware, distance in distances[test_malware].items():
                                if distance == sorted_distances[i]:
                                        closest_malwares_30.append(malware)
                                        closest_classes_30.append(malware[malware.index('_') + 6 : malware.index('_') + 7])
                counter = Counter(closest_classes_30)
                closest_class = counter.most_common()[0][0]
                print('\n30 closest malwares to ' + test_malware + ' :\n')
                print(counter.most_common())
                print('30_NN classification  =  ' + closest_class)
                print('\n')
                out_file.write('30_NN classification  =  ' + closest_class + '\n')

                print('\n\n\n')

