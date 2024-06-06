#!/bin/bash

#folders=("GV" "PH" "DC" "EBG" "PZ" "BG" "FE")

folders=("gv" "ph" "dc" "bg" "fe" "pz" "ebg")
#trial=("T1" "T2" "T3" "T4")
#dir=("10" "100" "200" "500" "800")
    # Iterate over each destination folder
        # Copy files from source folder to destination folder
#for T in "${trial[@]}"; do
	for fold in "${folders[@]}";do
     		#for d in "${dir[@]}"; do
				for file in "$fold"/*;do
					size=$(du -m "$file" | cut -f1)
					if [ "$size" -gt 50 ]; then
                                		echo "File: $file - Size: ${size}MB"
					fi
				done

	done
