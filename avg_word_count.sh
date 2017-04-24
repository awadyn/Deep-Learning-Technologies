#!/bin/bash
##########################################################################
#### find average number of lines for a malware in this malware class ####
##########################################################################

for class in `ls malware_by_class`
do
        COUNTER=1
        for file in `ls malware_by_class/${class}`
        do
                if [ "${COUNTER}" -eq 10 ]
                then
                        wc -l "malware_by_class/${class}/${file}"
                fi
                let COUNTER=${COUNTER}+1
        done
done

#for class in `ls parsed_train_malwares`
#do
#        echo "word count for class ${class}"
#        echo `wc -l parsed_train_malwares/${class}`
#        echo
#done

echo
echo

for class in `ls parsed_test_malwares`
do
        echo `wc -l parsed_test_malwares/${class}`
done



