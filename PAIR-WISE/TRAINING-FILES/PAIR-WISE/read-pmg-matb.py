import json
import numpy as np
import glob
import os

from pymatgen.core.structure import Structure 
from pymatgen.io.jarvis import JarvisAtomsAdaptor

from tqdm import tqdm

data = 'gv_std_90_10.json'
with open(data, "rb") as f:
    dataset = json.loads(f.read())


# ### Now set up alignn model
from alignn.data import get_train_val_loaders
from jarvis.db.jsonutils import loadjson
from alignn.config import TrainingConfig

config = loadjson('config_git.json')
config = TrainingConfig(**config)

(
    train_loader,
    val_loader,
    test_loader,
    prepare_batch,
) = get_train_val_loaders(
    dataset_array=dataset,
    target=config.target,
    n_train=config.n_train,
    n_val=config.n_val,
    n_test=config.n_test,
    train_ratio=config.train_ratio,
    val_ratio=config.val_ratio,
    test_ratio=config.test_ratio,
    batch_size=config.batch_size,
    atom_features=config.atom_features,
    neighbor_strategy=config.neighbor_strategy,
    standardize=config.atom_features != "cgcnn",
    id_tag=config.id_tag,
    pin_memory=config.pin_memory,
    workers=config.num_workers,
    save_dataloader=config.save_dataloader,
    use_canonize=config.use_canonize,
    filename=config.filename,
    cutoff=config.cutoff,
    max_neighbors=config.max_neighbors,
    output_features=config.model.output_features,
    classification_threshold=config.classification_threshold,
    target_multiplication_factor=config.target_multiplication_factor,
    standard_scalar_and_pca=config.standard_scalar_and_pca,
    keep_data_order=config.keep_data_order,
    output_dir=config.output_dir,
)


from alignn.train import train_dgl
train_dgl(
    config,
    train_val_test_loaders=[
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ],
)

with open('checkpoints/history_val.json', 'r') as f:
  data = json.load(f)

min_val = np.argmin(data['loss'])
min_chck = './checkpoints/checkpoint_' + str(min_val+1) + '.pt'
last_chck = './checkpoints/checkpoint_' + str(len(data['loss'])) + '.pt'
os.rename(min_chck, 'checkpoint_min_val.pt')
os.rename(last_chck, 'checkpoint_final.pt')

validation_file=data
n_early_stopping=50
"""Check if early stopping reached."""
early_stopping_reached = False
maes = validation_file["mae"]
best_mae = 1e9
no_improvement = 0
best_epoch = len(maes)
for ii, i in enumerate(maes):
    if i > best_mae:
        no_improvement += 1
        if no_improvement == n_early_stopping:
            print("Reached Early Stopping at", i, "epoch=", ii)
            early_stopping_reached = True
            best_mae = i
            best_epoch = ii
            break
    else:
        no_improvement = 0
        best_mae = i


print(best_epoch)
best_val = './checkpoints/checkpoint_' + str(best_epoch) + '.pt'
os.rename(best_val, 'checkpoint_best_val.pt')


dir_path = "./checkpoints/"
# pattern for file names to be deleted
file_pattern = "*.pt"
# get a list of file paths using the glob module
file_paths = glob.glob(os.path.join(dir_path, file_pattern))
# loop over each file path and delete the file
for file_path in file_paths:

    os.remove(file_path)
