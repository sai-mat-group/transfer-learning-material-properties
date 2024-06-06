#!/bin/bash


# Source folders

# Destination folders

trial=("T3")
dir=("10" "100" "200" "500" "800")
    # Iterate over each destination folder
for T in "${trial[@]}"; do
       for d in "${dir[@]}"; do  
	     	cp -r ../../DC/"$T"/"$d"/checkpoints/prediction_results_train_set.csv "$T"/"$d"/.
		cp -r ../../DC/"$T"/"$d"/checkpoints/history_* "$T"/"$d"/.
		cp -r ../../DC/"$T"/"$d"/prediction_results_test_set.csv "$T"/"$d"/.
		#cp -r ../../DC/"$T"/"$d"/prediction_results_train_set.csv "$T"/"$d"/.
		#cp -r ../../DC/"$T"/"$d"/history_* "$T"/"$d"/.
       done
done  
