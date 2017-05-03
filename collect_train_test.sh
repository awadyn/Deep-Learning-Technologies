#!/bin/bash

#### collect train and test data
#### adjust number of files collected here

SOURCE_DIR="/Volumes/Elements/299/malwares_by_class"
DESTINATION_TRAIN_DIR="malware_by_class"
DESTINATION_TEST_DIR="malware_by_class_test"

TRAIN_FLAG=$1
TRAIN_SAMPLE_SIZE=$2
TEST_FLAG=$3
TEST_SAMPLE_SIZE=$4

for class in 1 2 3 4 5 6 7 8 9
do
        # create directories if they do not exist
        [[ -d ${DESTINATION_TRAIN_DIR}/${class} ]] || mkdir ${DESTINATION_TRAIN_DIR}/${class}
        [[ -d ${DESTINATION_TEST_DIR}/${class} ]] || mkdir ${DESTINATION_TEST_DIR}/${class}
        
        # clean up directories if they contain files
        if test "$(ls -A ${DESTINATION_TRAIN_DIR}/${class})"
        then
                echo "Deleting pre-existing train files"
                rm ${DESTINATION_TRAIN_DIR}/${class}/*
        fi
        if test "$(ls -A ${DESTINATION_TEST_DIR}/${class})"
        then
                echo "Deleting pre-existing test files"
                rm ${DESTINATION_TEST_DIR}/${class}/*
        fi

        # copy files to directories
        echo "Copying ${TRAIN_SAMPLE_SIZE} train files to ${DESTINATION_TRAIN_DIR}/${class}"
        for file in `ls ${SOURCE_DIR}/${class} | head -${TRAIN_SAMPLE_SIZE}`
        do
                cp ${SOURCE_DIR}/${class}/${file} ${DESTINATION_TRAIN_DIR}/${class}
        done
        
        echo "Copying ${TEST_SAMPLE_SIZE} test files to ${DESTINATION_TEST_DIR}/${class}"
        for file in `ls ${SOURCE_DIR}/${class} | tail -${TEST_SAMPLE_SIZE}`
        do
                cp ${SOURCE_DIR}/${class}/${file} ${DESTINATION_TEST_DIR}/${class}
#                test_size=`wc -l ${SOURCE_DIR}/${class}/${file} | tr -s " " | cut -f2 -d " "`
#                counter=0
#                for file in `ls ${SOURCE_DIR}/${class}`
#                do
#                        train_size=`wc -l ${SOURCE_DIR}/${class}/${file} | tr -s " " | cut -f2 -d " "`
#                        if [ "${test_size}" -gt $((${train_size} - 5000)) ]
#                        then
#                                if [ "${test_size}" -lt $((${train_size} + 5000)) ]
#                                then
#                                        echo "found good file..."
#                                        cp ${SOURCE_DIR}/${class}/${file} ${DESTINATION_TRAIN_DIR}/${class}
#                                        let counter=${counter}+1
#                                else
#                                        echo "file not good..."
#                                fi
#                        fi
#                        if [ "${counter}" -eq "${TRAIN_SAMPLE_SIZE}" ]
#                        then
#                                echo "Copied ${TRAIN_SAMPLE_SIZE} train files to ${DESTINATION_TRAIN_DIR}/${class}"
#                                break
#                        fi
#                done
        done

        echo
done
