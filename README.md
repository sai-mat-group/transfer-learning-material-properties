# transfer-learning-material-properties

This repository contains the pre-trained and fine-tuned atomistic line graph neural
network (ALIGNN) architectures and the corresponding results for the calculations carried out as a part of the research paper titled "Optimal pre-train/fine-tune strategies for accurate material property predictions". The archived version of the manuscript is available at [arXiv](http://arxiv.org/abs/2406.13142).

The objective of the research is to study the influence of important handles such as the pre-train (PT) and fine-tune (FT) dataset sizes, FT strategies, and other important hyperparameters on employing pair-wise transfer learning (TL) in the prediction of material properties using graph neural network (GNN) based architectures. Importantly, we also develop a GNN framework that is simultaneously PT on multiple properties (MPT), enabling the construction of generalized GNN models.

The repository contains four important folders containing the model checkpoints (.pt files) and the results (.csv and .json files) corresponding to different sections of the paper: HP-TUNE (Sec 2.5,3.3), MULTI_PROPERTY (Sec 3.5,3.6), PAIR_WISE (Sec 3.1,3.2,3.4), and SCRATCH_MODELS. The detailed instructions to replicate any calculation can be found in the NOTE file.

The training files required for each calculation are stored in the sub-folder TRAINING-FILES within each of the main folders. The dataset files used for each calculation are stored in the DATA-JSON-TRIAL-RUNS sub-folder within the PAIR_WISE folder. The calculations pertaining to bandgap, shear modulus, formation energy, phonons, dielectric constant, piezoelectric modulus, and experimental band gap are represented by the folders BG, GV, FE, PH, DC, PZ, and EBG, respectively. The results pertaining to different dataset sizes are named 10, 100, 200, 500, and 800. The PT and FT checkpoints are named checkpoint_final.pt and checkpoints_500.pt, respectively. The FT checkpoints stored in the same format as the results and the cumulative dataset can be found in [OneDrive](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/reshmadevi_iisc_ac_in/Es_cvJqdvFNOh5qTVD1CG9QBTW-hMCVej_Vuln4kEaxzSw?e=KuwOl4). 

In case you use any of the data here, we would appreciate a citation to our manuscript at [arXiv](http://arxiv.org/abs/2406.13142), and the eventually published one at (DoI-to-be-updated).

###############INSTRUCTIONS FOR ALIGNN(__version__ = "2023.04.07") INSTALLAION################################
The "alignn-src" folder with the setup.py file is available on the repository.
```
conda create --name alignn python=3.9
conda activate alignn
cd alignn-src
python3 setup.py develop
pip install dgl-cu111
pip install pymatgen
```
Alternatively,
```
conda create --name alignn python=3.9
conda activate alignn
cd alignn-src
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c dglteam dgl-cuda11.3 dgl
python3 setup.py develop
pip install pymatgen
```
