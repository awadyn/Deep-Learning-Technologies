import numpy as np
import sys
import os

parsed_malwares_dir = "parsed_malwares"

total_words = []

for root, dirs, filenames in os.walk(parsed_malwares_dir):
    for filename in filenames:
            print('adding malware_words of ' + filename + 'to vocabulary...')
            file = open(os.path.join(root, filename), 'r')
            data = file.read().splitlines()
            for word in data:
                    total_words.append(word)

vocabulary = list(set(total_words))


print("len(vocabulary): " + str(len(vocabulary)))
print("vocabulary:")
print(vocabulary)

output_file = open("vocabulary", "w")
for word in vocabulary:
        output_file.write(word)
        output_file.write("\n")
