import json
import random
import torch

import numpy as np
import pandas as pd

from jarvis.core.graphs import Graph
from jarvis.core.atoms import Atoms

from pymatgen.io.jarvis import JarvisAtomsAdaptor
from pymatgen.core import Structure

#from alignn.models.alignn import ALIGNN
from alignn_multi import ALIGNN_ft, ALIGNN

#from alignn.data import get_torch_dataset
from data_multi import get_torch_dataset

from tqdm import tqdm

import dgl


def atoms_to_graph(atoms, cutoff=6.0, max_neighbors=12,
    atom_features="cgcnn", use_canonize=True):
    """Convert structure dict to DGLGraph."""
    #structure = Atoms.from_dict(atoms)
    structure = JarvisAtomsAdaptor.get_atoms(Structure.from_dict(atoms))
    return Graph.atom_dgl_multigraph(
        structure,
        cutoff=cutoff,
        atom_features=atom_features,
        max_neighbors=max_neighbors,
        compute_line_graph=True,
        use_canonize=use_canonize,
    )

def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]

def collate_line_graph(samples):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, torch.stack(labels)
        else:
            return batched_graph, batched_line_graph, torch.tensor(labels)

### Parameters that we need to set
data = 'MP_cor_jarvis_2d_std_90_10.json'
epochs = 500
load_prev_best_model = True
train_split = 90
test_split = 0
val_split = 10
batch_size = 16
lim = False      # Just used if you want a smaller subset for testing
learning_rate = 1e-3 
prop_indices = [3]  # This is the index of the property - taken from the OH column
n_early_stopping = 500
n_outputs = 7 # Dimensions of the original MPR model, 6
n_outputs_new = 1 # Dimensions of the original MPR model, 6
n_hidden = 2 # Number of hidden layers in the head
print_outputs = False
checkpoint_dir = './fine-tune/' # Where all your checkpoints will be saved
checkpoint_fp = 'best_model-all.pt' # The checkpoint of the general model to load up initially
layers_to_copy = ['atom_embedding', 'edge_embedding', 'angle_embedding', 'alignn_layers', 'gcn_layers', 'readout'] # The layers from which to load pre-trained parameters
#checkpoint_fp = './fine-tune-diel/best_model.pt' # The checkpoint of the general model to load up initially
### No need to edit beyond here



device = "cpu"
if torch.cuda.is_available():
    print('Found GPU and CUDA')
    device = torch.device("cuda")

with open(data, "rb") as f:
    dataset = json.loads(f.read())
if lim:
    dataset = dataset[:lim]

train_lim = int(len(dataset) / 100 * train_split)
val_lim = int(len(dataset) / 100 * val_split)
test_lim = int(len(dataset) / 100 * test_split)

print('Taining data: ', train_lim, 'Validaion data:  ', val_lim)

# Read in the old model
model = ALIGNN(n_outputs=n_outputs, 
        print_outputs=print_outputs, n_hidden=n_hidden)
model.to(device)
model.load_state_dict(torch.load(checkpoint_fp, map_location=torch.device(device))['model'])

# Set up the new model
model_new = ALIGNN_ft(n_outputs=n_outputs_new, 
        print_outputs=print_outputs, n_hidden=n_hidden)
model_new.to(device)

# Copy weights from selected layers to new model
for l in layers_to_copy:
    l_o = getattr(model, l)
    l_n = getattr(model_new, l)
    l_n.load_state_dict(l_o.state_dict())

# Set up the optimiser, loss and device

criterion = torch.nn.L1Loss()
params = group_decay(model)
optimizer = torch.optim.AdamW(params, lr=learning_rate)
    
for datum in tqdm(dataset):
#    datum['atoms'] = Atoms.to_dict(JarvisAtomsAdaptor.get_atoms(Structure.from_dict(datum['structure'])))
     datum['atoms'] = datum['atoms']
#    datum['has_prop'] = torch.FloatTensor(datum['OH']) 
#    datum['has_prop'] = datum['has_prop'][prop_indices]
     datum['target'] = torch.FloatTensor(datum['prop_list'])
#    datum['target'] = datum['target'][prop_indices]

train_data = get_torch_dataset(dataset[:train_lim], target='target', neighbor_strategy="k-nearest", atom_features="cgcnn", line_graph=True)
val_data = get_torch_dataset(dataset[train_lim:train_lim+val_lim], target='target', neighbor_strategy="k-nearest", atom_features="cgcnn", line_graph=True)

from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
    )
from torch import nn
from torch.utils.data import DataLoader

collate_fn = collate_line_graph
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

from ignite.metrics import Loss, MeanAbsoluteError

criterion = torch.nn.L1Loss()
params = group_decay(model_new)
optimizer = torch.optim.AdamW(params, lr=1e-4)

metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}

# ## Set up trainer and evaluator

trainer = create_supervised_trainer(
        model_new,
        optimizer,
        criterion,
        prepare_batch=train_loader.dataset.prepare_batch,
        device=device,
        deterministic=False,
        # output_transform=make_standard_scalar_and_pca,
        )

evaluator = create_supervised_evaluator(
        model_new,
        metrics=metrics,
        prepare_batch=val_loader.dataset.prepare_batch,
        device=device,
        #
       )

train_evaluator = create_supervised_evaluator(
        model_new,
        metrics=metrics,
        prepare_batch=val_loader.dataset.prepare_batch,
        device=device,
        #
       )

print('TRAINER  DEVICE', device)

# ## Set up checkpoint saving

from ignite.handlers import Checkpoint, DiskSaver

# what to check
def cp_score(engine):
    """Lower MAE is better."""
    return -engine.state.metrics["mae"]


# ## Logging performance

from ignite.handlers import EarlyStopping
from ignite.handlers.stores import EpochOutputStore
from jarvis.db.jsonutils import dumpjson
from ignite.contrib.handlers.tqdm_logger import ProgressBar

history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
        }

eos = EpochOutputStore()
eos.attach(evaluator)
train_eos = EpochOutputStore()
train_eos.attach(train_evaluator)

## A learning rate scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            # pct_start=pct_start,
            pct_start=0.3,
            )

# what to save
to_save = {
        "model": model_new,
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "trainer": trainer,
          }

# save last two epochs
evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        Checkpoint(
            to_save,
            DiskSaver(
                checkpoint_dir, create_dir=True, require_empty=False
            ),
            n_saved=2,
            global_step_transform=lambda *_: trainer.state.epoch,
            ),
        )
# save best model
evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        Checkpoint(
            to_save,
            DiskSaver(
                checkpoint_dir, create_dir=True, require_empty=False
            ),
            filename_pattern="best_model.{ext}",
            n_saved=1,
            global_step_transform=lambda *_: trainer.state.epoch,
            score_function=cp_score,
            ),
        )

# collect evaluation performance
@trainer.on(Events.EPOCH_COMPLETED)
def log_results(engine):
    """Print training and validation metrics to console."""
    train_evaluator.run(train_loader)
    evaluator.run(val_loader)

    tmetrics = train_evaluator.state.metrics
    vmetrics = evaluator.state.metrics
    for metric in metrics.keys():
        tm = tmetrics[metric]
        vm = vmetrics[metric]
        if metric == "roccurve":
            tm = [k.tolist() for k in tm]
            vm = [k.tolist() for k in vm]
        if isinstance(tm, torch.Tensor):
            tm = tm.cpu().numpy().tolist()
            vm = vm.cpu().numpy().tolist()

        history["train"][metric].append(tm)
        history["validation"][metric].append(vm)
        
    dumpjson(
         filename=checkpoint_dir+"/history_val.json",
         data=history["validation"],
        )
    dumpjson(
        filename=checkpoint_dir+"/history_train.json",
        data=history["train"],
        )
    
    pbar = ProgressBar()
    pbar.log_message("  #######  ")
    pbar.log_message(f"Train_MAE: {tmetrics['mae']:.4f}")
    pbar.log_message(f"Val_MAE: {vmetrics['mae']:.4f}")
    pbar.log_message("    ")
    
    es_handler = EarlyStopping(
        patience=n_early_stopping,
        score_function=cp_score,
        trainer=trainer,
        )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, es_handler)

pbar = ProgressBar()
pbar.attach(trainer, output_transform=lambda x: {"loss": x})
trainer.add_event_handler(
    Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

trainer.run(train_loader, max_epochs=epochs)
