#!/bin/bash

./preprocess_for_tests_k_nn.sh

# running classification
echo "Running classification"
python test_gensim_k_nn.py

