#!/bin/bash


# Source folders
source_folders=("50K")

# Destination folders
#dir=("DC_3" "DC_4" "DC_5" "FE_3" "FE_4" "FE_5")
dir=("BG-FE" "DC-BG" "FE-BG" "PH-GV")
#dir=("FE-BG")
# Iterate over each source folder
for d in "${dir[@]}"; do
			mv "$d"/checkpoint_final.pt "$d"/checkpoint_final-1.pt
			mv "$d"/checkpoint_500.pt "$d"/checkpoint_final.pt
			mv "$d"/checkpoint_final-1.pt "$d"/checkpoint_500.pt
                        cp "$d"/checkpoints/history* "$d"/.
			cp "$d"/checkpoints/prediction_results_train_set.csv "$d"/.
			#cp /data/pawanj/reshma/TL/LD/LD_TRIALS/BG/"$folder"/"$T"/"$dest"/"$d"/checkpoints/prediction_results_train_set.csv BG/"$folder"/"$T"/"$dest"/"$d"/.
             		#cp /data/pawanj/reshma/TL/LD/LD_TRIALS/BG/"$folder"/"$T"/"$dest"/"$d"/checkpoints/history* BG/"$folder"/"$T"/"$dest"/"$d"
done

