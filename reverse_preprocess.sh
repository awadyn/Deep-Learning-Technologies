#!/bin/bash

TEST_FILE=$1

SOURCE_DIR="parsed_malwares"
DESTINATION_DIR="parsed_train_malwares"

for class in 1 2 3 4 5 6 7 8 9
do
        if test "$(ls -A ${SOURCE_DIR}/${class})"
        then
                echo "Moving files in ${SOURCE_DIR}/${class} back to ${DESTINATION_DIR}"
                mv "${SOURCE_DIR}/${class}/"* "${DESTINATION_DIR}/"
        fi
done

echo "Moving ${TEST_FILE} back to ${DESTINATION_DIR}"
mv "${TEST_FILE}" "${DESTINATION_DIR}/"

