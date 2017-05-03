#!/bin/bash

if test "$(ls -A 'parsed_test_malwares')"
then
        echo "Moving test files back to parsed_train_malwares"
        mv "parsed_test_malwares/"* "parsed_train_malwares/"
fi

for file in `ls parsed_train_malwares`
do
        # choose test malware
        echo "Choosing test malware ${file}"
        mv parsed_train_malwares/"${file}" parsed_test_malwares/
        echo
        echo
        # preprocess train malwares, isolating similar size malwares
        echo "Preprocessing train malwares"
        ./preprocess_for_test.sh "parsed_test_malwares/${file}"
        echo
        echo
        # running classification
        echo "Running classification"
        python test_gensim.py "parsed_test_malwares/${file}"
        echo
        echo
        # reversing preprocess for next test
        echo "Reversing preprocessing to prepare for next test"
        ./reverse_preprocess.sh "parsed_test_malwares/${file}"
        echo
        echo
done
