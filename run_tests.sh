#!/bin/bash

# remove pre-existing log files
if [ -e "distances/distances.log" ]
then
        rm "distances/distances.log"
fi
touch "distances/distances.log"

TEST_MALWARE_DIR="parsed_test_malwares"

COUNTER=1
for file in `ls ${TEST_MALWARE_DIR}`
do
        if [ ${COUNTER} -eq 1 ]
        then
                python wcd.py ${TEST_MALWARE_DIR}/${file} -embeddings
        else
                python wcd.py ${TEST_MALWARE_DIR}/${file} -no-embeddings
        fi
        let COUNTER=${COUNTER}+1
done
