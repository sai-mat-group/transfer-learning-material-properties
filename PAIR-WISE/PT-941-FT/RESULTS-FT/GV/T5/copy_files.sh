#!/bin/bash


folders=("bg" "ph" "dc" "fe" "pz" "ebg")
dir=("10" "100" "200" "500" "800")
    # Iterate over each destination folder
        # Copy files from source folder to destination folder
for fold in "${folders[@]}";do
     	for d in "${dir[@]}"; do
		cp "$fold"/"$d"/checkpoints/prediction_results_train_set.csv "$fold"/"$d"/. 	
		cp "$fold"/"$d"/checkpoints/history_* "$fold"/"$d"/.
	done
done
