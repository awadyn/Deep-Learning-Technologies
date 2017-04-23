#!/bin/bash
##########################################################################
#### find average number of lines for a malware in this malware class ####
##########################################################################

for class in `ls parsed_malwares`
do
        echo "word count for class ${class}"
        echo `wc -l parsed_malwares/${class}`
        echo
done

for class in `ls parsed_test_malwares`
do
        echo "word count for class ${class}"
        echo `wc -l parsed_test_malwares/${class}`
        echo
done



