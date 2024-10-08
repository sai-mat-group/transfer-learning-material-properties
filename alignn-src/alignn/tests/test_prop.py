"""Training script test suite."""
import time
import matplotlib.pyplot as plt
import numpy as np
from alignn.train import train_dgl
from alignn.pretrained import get_prediction
from alignn.pretrained import get_multiple_predictions
from sklearn.metrics import mean_absolute_error
import os
from jarvis.core.atoms import Atoms
from alignn.train_folder import train_for_folder
from alignn.train_folder_ff import train_for_folder as train_for_folder_ff
from jarvis.db.figshare import get_jid_data
from alignn.ff.ff import AlignnAtomwiseCalculator, default_path, revised_path

plt.switch_backend("agg")

config = {
    "dataset": "dft_2d",
    "target": "formation_energy_peratom",
    # "target": "optb88vdw_bandgap",
    "n_train": 50,
    "n_test": 25,
    "n_val": 25,
    "num_workers": 0,
    "atom_features": "cgcnn",
    "neighbor_strategy": "k-nearest",
    "epochs": 2,
    "save_dataloader": False,
    "batch_size": 10,
    "weight_decay": 1e-05,
    "learning_rate": 0.01,
    "criterion": "mse",
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "num_workers": 4,
    "model": {
        "name": "alignn",
    },
}


# def test_runtime_training():
#    cmd1 = 'python alignn/train_folder.py --root_dir "alignn/examples/sample_data" --config "alignn/examples/sample_data/config_example.json"'
#    os.system(cmd1)
#    cmd2 = 'python alignn/train_folder.py --root_dir "alignn/examples/sample_data" --classification_threshold 0.01 --config "alignn/examples/sample_data/config_example.json"'
#    os.system(cmd2)
#    cmd3 = 'python alignn/train_folder.py --root_dir "alignn/examples/sample_data_multi_prop" --config "alignn/examples/sample_data/config_example.json"'
#    os.system(cmd3)


def test_minor_configs():
    tmp = config
    # tmp["log_tensorboard"] = True
    tmp["n_early_stopping"] = 2
    tmp["model"]["name"] = "alignn"
    config["write_predictions"] = True
    result = train_dgl(tmp)


def test_models():
    """Test CGCNN end to end training."""
    config["model"]["name"] = "dense_alignn"
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Toal time:", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["write_predictions"] = True
    config["model"]["name"] = "alignn"
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "alignn_layernorm"
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "cgcnn"
    config["write_predictions"] = False
    config["save_dataloader"] = False
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "densegcn"
    config["write_predictions"] = False
    config["save_dataloader"] = False
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "icgcnn"
    config["write_predictions"] = False
    config["save_dataloader"] = False
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "alignn_cgcnn"
    config["write_predictions"] = False
    config["save_dataloader"] = False
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    # Classification
    config["model"]["name"] = "dense_alignn"
    config["classification_threshold"] = 0.0
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Toal time:", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "alignn"
    config["classification_threshold"] = 0.0
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "cgcnn"
    config["write_predictions"] = False
    config["save_dataloader"] = False
    config["classification_threshold"] = 0.0
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "alignn_cgcnn"
    config["write_predictions"] = False
    config["save_dataloader"] = True
    config["classification_threshold"] = 0.0
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    """

    config["model"]["name"] = "simplegcn"
    config["write_predictions"] = False
    config["save_dataloader"] = False
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()
    """
    """
    x = []
    y = []
    for i in result["EOS"]:
        x.append(i[0].cpu().numpy().tolist())
        y.append(i[1].cpu().numpy().tolist())
    x = np.array(x, dtype="float").flatten()
    y = np.array(y, dtype="float").flatten()
    plt.plot(x, y, ".")
    plt.xlabel("DFT")
    plt.ylabel("ML")
    plt.savefig("compare.png")
    plt.close()
    """


def test_pretrained():
    box = [[2.715, 2.715, 0], [0, 2.715, 2.715], [2.715, 0, 2.715]]
    coords = [[0, 0, 0], [0.25, 0.2, 0.25]]
    elements = ["Si", "Si"]
    Si = Atoms(lattice_mat=box, coords=coords, elements=elements)
    prd = get_prediction(atoms=Si)
    print(prd)
    cmd1 = "python alignn/pretrained.py"
    os.system(cmd1)
    get_multiple_predictions(atoms_array=[Si, Si])


def test_alignn_train():
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/sample_data/")
    )
    config = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../examples/sample_data/config_example.json",
        )
    )
    train_for_folder(root_dir=root_dir, config_name=config)

    root_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../examples/sample_data_multi_prop/"
        )
    )
    config = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../examples/sample_data/config_example.json",
        )
    )
    train_for_folder(root_dir=root_dir, config_name=config)

    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/sample_data/")
    )
    config = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../examples/sample_data/config_example.json",
        )
    )
    train_for_folder(
        root_dir=root_dir, config_name=config, classification_threshold=0.01
    )

    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/sample_data_ff/")
    )
    config = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../examples/sample_data_ff/config_example_atomwise.json",
        )
    )
    train_for_folder_ff(root_dir=root_dir, config_name=config)


def test_calculator():
    atoms = Atoms.from_dict(
        get_jid_data(dataset="dft_3d", jid="JVASP-32")["atoms"]
    )
    model_path = default_path()
    calc = AlignnAtomwiseCalculator(path=model_path)
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = calc
    energy = ase_atoms.get_potential_energy()
    forces = ase_atoms.get_forces()
    stress = ase_atoms.get_stress()
    print("round(energy,3)", round(energy, 3))
    print("max(forces.flatten()),3)", max(forces.flatten()))
    print("max(stress.flatten()),3)", max(stress.flatten()))
    # assert round(energy,3)==round(-60.954999923706055,3)
    # assert round(max(forces.flatten()),2)==round(0.08332983,2)
    # assert round(max(stress.flatten()),2)==round(0.002801671050217803,2)


# test_minor_configs()
# test_pretrained()
# test_runtime_training()
# test_alignn_train()
# test_models()
# test_calculator()
