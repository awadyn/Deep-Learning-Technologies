import sys
import os

import gensim
import nltk

# setup vocabulary
vocabulary_file = open("vocabulary", "r")
vocabulary = list(set(vocabulary_file.read().splitlines()))

# setup test malware to classify
test_malware_dir = "parsed_test_malwares"
test_malwares = []
for root, dirs, filenames in os.walk(test_malware_dir):
        for test_malware in filenames:
                test_malwares.append(os.path.join(root, test_malware))

# setup malwares to compare against
malware_dir = "parsed_malwares"
malwares = []
for root, dirs, filenames in os.walk(malware_dir):
        for malware in filenames:
                malwares.append(os.path.join(root, malware))


print('Setup vocabulary and malware directories...')


# setup word2vec model
model = gensim.models.Word2Vec([vocabulary], workers=2, min_count=1)


print('Setup word2vec model...')


# setup malware docs to compare against
malware_docs = []
for malware in malwares:
        malware_file = open(malware, "r")
        malware_docs.append(malware_file.read().splitlines()) 


print('Setup malware docs to compare against...')


for i, test_malware in enumerate(test_malwares):
        # setup test malware doc
        test_malware_file = open(test_malwares[i], "r")
        test_malware_doc = test_malware_file.read().splitlines()
        
        
        print('Setup test malware doc: ' + test_malwares[i])
        print('\n\n')
        
        
        print('Measuring distances: \n')
        distances = []
        for j, malware_doc in enumerate(malware_docs):
                distance = model.wmdistance(test_malware_doc, malware_doc)
                distances.append(distance)
                print(test_malwares[i] + ' --> ' + malwares[j])
                print(distance)
        min_distance = distances[0]
        for distance in distances:
                if min_distance > distance:
                        min_distance = distance
        print('\n')
        print('Min(distances): ' + str(min_distance))
        print('Classification: ' + str(distances.index(min_distance) + 1))
        




