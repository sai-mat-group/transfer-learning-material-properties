import json
import torch

from pymatgen.core.structure import Structure 
from pymatgen.io.jarvis import JarvisAtomsAdaptor

from tqdm import tqdm

data = 'dc_std_10_test.json'  # The data to re-train on 
model_path = 'checkpoint_500.pt' # The model checkpoint to load initially 
freeze_before = 8 # The layer at which to unfreeze the weights

with open(data, "rb") as f:
    dataset = json.loads(f.read())

# ### Now set up alignn model
from alignn.data_trial import get_train_val_loaders
from jarvis.db.jsonutils import loadjson
from alignn.config import TrainingConfig
from alignn.models.alignn import ALIGNN

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


from alignn.train_trial import train_dgl
## Check for GPU and CUDA
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

model = ALIGNN()
model.load_state_dict(torch.load(model_path,  map_location=torch.device('cpu'))["model"])
layer_count = 0

for child in model.children():
    layer_count += 1
    if layer_count < freeze_before:
        for param in child.parameters():
            param.requires_grad = False


#model.to(device)

train_dgl(
    config,
    model,
    train_val_test_loaders=[
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ],
)
