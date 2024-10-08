# Optimal pre-train/fine-tune strategies for accurate material property predictions

This repository contains the pre-trained and fine-tuned atomistic line graph neural
network ([ALIGNN](https://www.nature.com/articles/s41524-021-00650-1)) architectures and the corresponding results for the calculations carried out as a part of the research paper titled "Optimal pre-train/fine-tune strategies for accurate material property predictions". The archived version of the manuscript is available at [arXiv](http://arxiv.org/abs/2406.13142).

The objective of the research is to study the influence of important handles such as the pre-train (PT) and fine-tune (FT) dataset sizes, FT strategies, and other important hyperparameters on employing pair-wise transfer learning (TL) in the prediction of material properties using graph neural network (GNN) based architectures. Importantly, we also develop a GNN framework that is simultaneously PT on multiple properties (MPT), enabling the construction of generalized GNN models. We used the ALIGNN architecture (__version__ = "2023.04.07", see below for installation instructions) for the PT and FT exercises performed in this work.

The repository contains four important folders containing the model checkpoints (.pt files) and the results (.csv and .json files) corresponding to different sections of the paper: "HP-TUNE" (Sections 2.5, 3.3 on the arXiv preprint), "MULTI_PROPERTY" (Sections 3.5, 3.6), "PAIR_WISE" (Sections 3.1, 3.2, 3.4), and "SCRATCH_MODELS". The detailed instructions to replicate any calculation can be found in the NOTE file.

The training files required for each calculation are stored in the sub-folder "TRAINING-FILES" within each main folder. The dataset files used for each calculation are stored in the "DATA-JSON-TRIAL-RUNS" sub-folder within the "PAIR_WISE" folder. The calculations pertaining to bandgap, shear modulus, formation energy, phonons, dielectric constant, piezoelectric modulus, and experimental band gap are compiled within sub-folders named BG (or bg), GV (gv), FE (fe), PH (ph), DC (dc), PZ (pz), and EBG (ebg), respectively, under the main folders. The results pertaining to different dataset sizes are named 10, 100, 200, 500, and 800. The PT and FT checkpoints are named checkpoint_final.pt and checkpoints_500.pt, respectively. The FT checkpoints stored in the same format as the results and the cumulative dataset can be found in [OneDrive](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/reshmadevi_iisc_ac_in/Es_cvJqdvFNOh5qTVD1CG9QBTW-hMCVej_Vuln4kEaxzSw?e=KuwOl4). 

In case you use any of the data here, we would appreciate a citation to our manuscript at [arXiv](http://arxiv.org/abs/2406.13142), and the eventually published one at (DoI-to-be-updated).


## Instructions for ALIGNN installation

The code uses an modified version of the [ALIGNN](https://github.com/usnistgov/alignn) architecture - long term we hope to include this model in the main "ALIGNN" project, but to replicate the training resutls of this paper one can use the modified source code here as follows.

The "alignn-src" folder with the setup.py file is available on the repository, which is the version of ALIGNN that we used to generate all the models shown in our work. The "README" file within the "alignn-src" folder is a replica of an older version of the README file in the original [ALIGNN repository](https://github.com/usnistgov/alignn). Hence, to reproduce the results in our manuscript, please use the following installation instructions instead of what is provided in the README file under "alignn-src". 

### For installation on typical Intel/AMD CPUs

Create a conda environment and activate it
```
conda create --name alignn python=3.9
conda activate alignn
```
Create a copy of the "alignn-src" directory that is available in this repository. Run the setup.py file within the directory.
```
cd alignn-src
python3 setup.py develop
pip install dgl-cu111
```
If pymatgen functionality is required, then use
```
pip install pymatgen
```

### For installation on NVIDIA GPUs

The preferred version of CUDA toolkit is 11.8, as available at the [NVIDIA developers](https://developer.nvidia.com/cuda-11-8-0-download-archive) website. Please install the CUDA toolkit and any necessary NVIDIA driver(s) before proceeding.

Create a conda environment and activate it
```
conda create --name alignn python=3.9
conda activate alignn --stack
```

Create a copy of the "alignn-src" directory that is available in this repository.  The versions of pytorch (2.2.1), dgl (11.3), and pytorch-cuda (11.8) need to be specifically stated during installation on GPUs, since they are compatible with the version of ALIGNN we used and did not create conflicts. Run the setup.py file within copied alignn-src directory, after the pytorch and dgl libraries have been installed.
```
cd alignn-src
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c dglteam dgl-cuda11.3 dgl
python3 setup.py develop
```
If pymatgen functionality is required, then use
```
pip install pymatgen
```
