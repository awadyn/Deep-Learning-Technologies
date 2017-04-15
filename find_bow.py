from sklearn.feature_extraction.text import CountVectorizer
from functools import partial
import numpy as np
from nltk import regexp_tokenize as tokenizer
import re

vocab_file = open("malware_vocabulary", "r")
vocab = list(set(vocab_file.read().splitlines()))

for malware_class in [1, 2, 3, 7, 8]:
    test_file = open("parsed_malwares/malware_lang_" + str(malware_class), "r")
    test = test_file.read()
    corpus = [test]

    pattern = re.compile('^[A-Za-z0-9]+(?:\\.[A-Za-z0-9]+)*', flags=re.M)
    vectorizer = CountVectorizer(min_df=1, vocabulary=vocab, analyzer=partial(tokenizer, pattern=pattern))

    vectorizer._validate_vocabulary()
    X = vectorizer.fit_transform(corpus)
    X.toarray()

    np.savetxt("malware_bows/malware_bow_" + str(malware_class), X.toarray())
