#!/bin/bash
cd ph
cd 10
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 100
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 200
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 500
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 800
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../../
#----------
#--------------------------------------
cd gv
cd 10
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 100
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 200
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 500
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 800
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../../
#-----------------------------------------------
cd dc
cd 10
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 100
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 200
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 500
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 800
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../../

#-------------------------------------------
cd fe
cd 10
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 100
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 200
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 500
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../

cd 800
cp checkpoint_final.pt checkpoint_final_old.pt
cp checkpoints/checkpoint_500.pt checkpoint_final.pt
python3 load-freeze-weights.py
cd ../../


