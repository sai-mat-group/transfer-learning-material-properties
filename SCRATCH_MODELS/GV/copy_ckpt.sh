#!/bin/bash


# Source folders

# Destination folders

trial=("T5")
dir=("10" "100" "200" "500" "800")
    # Iterate over each destination folder
for T in "${trial[@]}"; do
       for d in "${dir[@]}"; do  
	     	cp -r "$T"/"$d"/checkpoints/* "$T"/"$d"/.
       done
done  
