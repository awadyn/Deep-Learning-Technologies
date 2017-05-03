#!/bin/bash

SOURCE_DIR="parsed_train_malwares"

for class in 1 2 3 4 5 6 7 8 9
do
        for file in `ls ${SOURCE_DIR}/malware_lang_${class}_*`
        do
                train_size=`wc -l ${file} | tr -s " " | cut -f2 -d " "`
                if [ "${train_size}" -le 1 ]
                then
                        rm "${file}"
                fi
        done
done

