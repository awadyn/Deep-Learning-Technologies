#!/bin/bash


#### psuedo-parses all malware assembly code
#### genertes text like files of opcode.arg.arg sequences
#### abstracts out some specifics for a more generic textual representation


ASSEMBLY_TO_HEX="assembly_hex.txt"
ASSEMBLY="assembly.txt"
TEMP_FILE="temp_lang"
LOG_FILE="logs.txt"


# clean up directories
if [ -d "parsed_malwares" ]
then
        rm "parsed_malwares/"*
fi
if [ -d "parsed_test_malwares" ]
then
        rm "parsed_test_malwares/"*
fi
if [ -d "temp_parsed_malwares" ]
then
        rm "temp_parsed_malwares/"*
fi
if [ -d "temp_parsed_test_malwares" ]
then
        rm "temp_parsed_test_malwares/"*
fi
# remove pre-existing log file
rm "${LOG_FILE}"
        

echo
echo
echo "Parsing training malware assembly files..."
for class in 1 2 3 4 5 6 7 8 9
do
        MALWARE_CLASS=${class}
                
        INPUT_TRAIN_DIR="malware_by_class/${MALWARE_CLASS}"
        OUTPUT_TRAIN_FILE="temp_parsed_malwares/malware_lang_temp"
        FINAL_OUTPUT_TRAIN_FILE="parsed_malwares/malware_lang_${MALWARE_CLASS}"

        COUNTER=1
        for file in "${INPUT_TRAIN_DIR}/"*.asm;
        do
                
                cp "${file}" "${TEMP_FILE}_${COUNTER}"
                
                echo "parsing ${file}..."
                
                sed -i -e "s/.text[^ ]*//g; /;/ d; /.rdata/ d; $(sed 's:.*:s/&//g:' assembly_hex.txt)" "${TEMP_FILE}_${COUNTER}"
                sed -n "$(sed 's:.*:/& / p:' assembly.txt)" "${TEMP_FILE}_${COUNTER}"  >> "${OUTPUT_TRAIN_FILE}_${COUNTER}" 
                
                # clean up
                rm "${TEMP_FILE}_${COUNTER}" 2>> "${LOG_FILE}"
        
                # remove all leading and trailing white spaces and new lines #
                tr -d '\011\015' < "${OUTPUT_TRAIN_FILE}_${COUNTER}" > "${TEMP_FILE}_${COUNTER}"
                tr -s " " < "${TEMP_FILE}_${COUNTER}" > "${OUTPUT_TRAIN_FILE}_${COUNTER}"
                sed -i -e 's/^[ \t]*//;s/[ \t]*$//; s/, /./g; s/ /./g; s/eax/gpr/g; s/ebx/gpr/g; s/ecx/gpr/g; s/edx/gpr/g; s/edi/gpr/g; s/esi/gpr/g; s/\[.*\(ebp\).*\]/\1/; s/\[.*\(esp\).*\]/\1/; s/\[.*\(gpr\).*\]/\1/; s/call.*/call/g' "${OUTPUT_TRAIN_FILE}_${COUNTER}"
        
                # let's not append them
                cat "${OUTPUT_TRAIN_FILE}_${COUNTER}" >> "${FINAL_OUTPUT_TRAIN_FILE}"
                echo "done..."
        
                #parsed_len=`wc -l ${OUTPUT_TRAIN_FILE}_${COUNTER} | tr -s " " | cut -f2 -d " "`
                #max_len=100000
                #if [ "${parsed_len}" -gt "${max_len}" ]
                #then
                #        echo "found file with good length..."
                #        echo ${len}
                #        head -100000 ${OUTPUT_TRAIN_FILE}_${COUNTER} | cat >> "${FINAL_OUTPUT_TRAIN_FILE}_${COUNTER}"
                #fi

                # clean up
                rm "${TEMP_FILE}_${COUNTER}" 2>> "${LOG_FILE}"
                rm "${TEMP_FILE}_${COUNTER}-e" 2>> "${LOG_FILE}"
                rm "${OUTPUT_TRAIN_FILE}_${COUNTER}-e" 2>> "${LOG_FILE}"
        
                let COUNTER=${COUNTER}+1
        done
done

echo
echo
echo "Parsing testing malware assembly files..."
for class in 1 2 3 4 5 6 7 8 9
do
        MALWARE_CLASS=${class}
        
        INPUT_TEST_DIR="malware_by_class_test/${MALWARE_CLASS}"
        OUTPUT_TEST_FILE="temp_parsed_test_malwares/malware_lang_temp"
        FINAL_OUTPUT_TEST_FILE="parsed_test_malwares/malware_lang_${MALWARE_CLASS}"
        
        COUNTER=1
        for file in "${INPUT_TEST_DIR}/"*.asm;
        do
                
                cp "${file}" "${TEMP_FILE}_${COUNTER}"
                
                echo "parsing ${file}..."
                
                sed -i -e "s/.text[^ ]*//g; /;/ d; /.rdata/ d; $(sed 's:.*:s/&//g:' assembly_hex.txt)" "${TEMP_FILE}_${COUNTER}"
                sed -n "$(sed 's:.*:/& / p:' assembly.txt)" "${TEMP_FILE}_${COUNTER}"  >> "${OUTPUT_TEST_FILE}_${COUNTER}" 
                
                # clean up
                rm "${TEMP_FILE}_${COUNTER}" 2>> "${LOG_FILE}"
        
                # remove all leading and trailing white spaces and new lines #
                tr -d '\011\015' < "${OUTPUT_TEST_FILE}_${COUNTER}" > "${TEMP_FILE}_${COUNTER}"
                tr -s " " < "${TEMP_FILE}_${COUNTER}" > "${OUTPUT_TEST_FILE}_${COUNTER}"
                sed -i -e 's/^[ \t]*//;s/[ \t]*$//; s/, /./g; s/ /./g; s/eax/gpr/g; s/ebx/gpr/g; s/ecx/gpr/g; s/edx/gpr/g; s/edi/gpr/g; s/esi/gpr/g; s/\[.*\(ebp\).*\]/\1/; s/\[.*\(esp\).*\]/\1/; s/\[.*\(gpr\).*\]/\1/; s/call.*/call/g' "${OUTPUT_TEST_FILE}_${COUNTER}"
        
                cat "${OUTPUT_TEST_FILE}_${COUNTER}" >> "${FINAL_OUTPUT_TEST_FILE}_${COUNTER}"
                echo "done..."
        
                # clean up
                rm "${TEMP_FILE}_${COUNTER}" 2>> "${LOG_FILE}"
                rm "${TEMP_FILE}_${COUNTER}-e" 2>> "${LOG_FILE}"
                rm "${OUTPUT_TEST_FILE}_${COUNTER}-e" 2>> "${LOG_FILE}"
        
                let COUNTER=${COUNTER}+1
        done
done
