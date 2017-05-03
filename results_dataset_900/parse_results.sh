#!/bin/bash

# clean up
rm classifications_*

num_classifications=0
num_correct=0
for class in 1 2 3 4 5 6 7 8 9
do
        num_classifications_class=0
        num_correct_class=0
        less malware_lang_results_${class} | sed 's/.*classification  =  \([0-9]\).*/\1/' | sed -n '/^[0-9]/p' >> classifications_${class}
        for line in `less classifications_${class}`
        do
                let num_classifications=${num_classifications}+1
                let num_classifications_class=${num_classifications_class}+1
                if [ "${line}" = "${class}" ]
                then
                        let num_correct=${num_correct}+1
                        let num_correct_class=${num_correct_class}+1
                fi
        done
        echo "accuracy for class ${class}:"
        echo "(${num_correct_class}*100)/${num_classifications_class}" | bc -l
done

echo ${num_classifications}
echo ${num_correct}
echo
echo "accuracy:"
echo "(${num_correct}*100)/${num_classifications}" | bc -l
