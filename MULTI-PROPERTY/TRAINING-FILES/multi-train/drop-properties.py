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
from alignn_multi import ALIGNN

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
        graphs, line_graphs, has_prop, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, torch.stack(has_prop), torch.stack(labels)
        else:
            return batched_graph, batched_line_graph, torch.stack(has_prop), torch.tensor(labels)


########## PARAMETETER TO SET
data = '../../data/No-dup-complete_dataset_100.json'
epochs = 30
load_prev_best_model = False # Restart training from checkpoints, set location with checkpoint_fp below
train_split = 90
test_split = 0
val_split = 10
batch_size = 4
lim = False # Limit the size of the dataset? False or integer (the number to cap at)
learning_rate = 1e-3
drop_indices = [3] # The indices of the properties to drop from the dataset (see README for details.)
n_early_stopping = 30
n_outputs=6 # Number of oupput propperties to predict = 7 - len(drop_indices)
n_hidden=1 # Number of hidden layers in the predictor head
print_outputs=False # Only use for debugging
checkpoint_fp = 'dropped-column-3/checkpoint_140.pt' # If restarting from a checkpoint, the location of the chkpt



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
for i in drop_indices: ## Loop across each of the indices
    checkpoint_dir = './dropped-column-%s' % str(i)
    prop_indices = [0, 1, 2, 3, 4, 5, 6]
    prop_indices.remove(i)
    model = ALIGNN(n_outputs=n_outputs, 
            print_outputs=print_outputs, n_hidden=n_hidden)
    model.to(device)
# First set up the optimiser, loss and device

    criterion = torch.nn.L1Loss()
    params = group_decay(model)
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    
    for datum in tqdm(dataset):
        datum['atoms'] = Atoms.to_dict(JarvisAtomsAdaptor.get_atoms(Structure.from_dict(datum['structure'])))
        datum['has_prop'] = torch.FloatTensor(datum['OH']) 
        datum['has_prop'] = datum['has_prop'][prop_indices]
        datum['target'] = torch.FloatTensor(datum['prop_list'])
        datum['target'] = datum['target'][prop_indices]

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
    params = group_decay(model)
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}

# ## Set up trainer and evaluator

    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        prepare_batch=train_loader.dataset.prepare_batch,
        device=device,
        deterministic=False,
        # output_transform=make_standard_scalar_and_pca,
        )

    evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        prepare_batch=val_loader.dataset.prepare_batch,
        device=device,
        #
       )

    train_evaluator = create_supervised_evaluator(
        model,
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
            "model": model,
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
    if load_prev_best_model:
        model.load_state_dict(torch.load(checkpoint_fp, map_location=torch.device(device))['model'])
        checkpoint = torch.load(checkpoint_fp, map_location=device) 
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    trainer.run(train_loader, max_epochs=epochs)

