#!/bin/bash

TEST_FILE=$1

SOURCE_DIR="parsed_train_malwares"
DESTINATION_DIR="parsed_malwares"

test_size=`wc -l ${TEST_FILE} | tr -s " " | cut -f2 -d " "`
if [ "${test_size}" -le 1000 ]
then
        ((valid_range=${test_size}))
else
        ((valid_range=${test_size}/3))
fi
echo "test parsed malware size  =  ${test_size}"
echo "valid range  =  ${valid_range}"

for class in 1 2 3 4 5 6 7 8 9
do

        # create directories if they do not exist
        [[ -d ${DESTINATION_DIR}/${class} ]] || mkdir ${DESTINATION_DIR}/${class}

        # clean up directories if they contain files
        if test "$(ls -A ${DESTINATION_DIR}/${class})"
        then
#                echo "Moving pre-existing files back"
                #rm ${DESTINATION_DIR}/${class}/*
                mv ${DESTINATION_DIR}/${class}/* ${SOURCE_DIR}
        fi

        for file in `ls ${SOURCE_DIR}/malware_lang_${class}_*`
        do
                train_size=`wc -l ${file} | tr -s " " | cut -f2 -d " "`
                if [ "${train_size}" -le 1 ]
                then
                        rm "${file}"
                fi
                if [ "${test_size}" -gt $((${train_size} - ${valid_range})) ]
                then
                        if [ "${test_size}" -lt $((${train_size} + ${valid_range})) ]
                        then
#                                echo "train parsed malware in range  =  ${train_size}"
                                mv "${file}" "${DESTINATION_DIR}/${class}"
                        fi
                fi
        done
#        echo
#        echo
done



