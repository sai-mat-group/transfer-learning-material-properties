#!/bin/bash



# Destination folders
dir=("10" "100" "200" "500" "800")
# Iterate over each source folder
    # Iterate over each destination folder
for d in "${dir[@]}"; do
                        cp "$d"/checkpoints/checkpoint_500.pt "$d"/.
			cp "$d"/checkpoints/history* "$d"/.
			cp "$d"/checkpoints/prediction_results_train_set.csv "$d"/.
done



