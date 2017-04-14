import numpy as np
import sys

total_words = []

for filename in sys.argv[1:]:
        file = open(filename, "r")
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
