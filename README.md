# Grover featurizer
This repository provides an interface to the pre-trained Grover-Large model, as developed by the Tencent AI Lab: https://github.com/tencent-ailab/grover.

GROVER  encodes molecular structures into a vectorial format amenable for downstream QSAR model training. It is based on Message Passing Networks and has been pre-trained on 10 million molecules (https://arxiv.org/abs/2007.02835). All credit goes to the original authors. The original MolBERT repository is released under an MIT License.

## Install

```bash
git clone git@github.com:ersilia-os/groverfeat.git
cd groverfeat
pip install -e .
```

By default, RDKit and h5py are not installed. Those are necessary **only** if you want to standardize molecules and generate the default chemical library. If you want to have these functionalities, simply do:
```bash
pip install rdkit-pypi
pip install h5py
```

## Calculate features

```python
from groverfeat import Featurizer

smiles_list = ["C", "CCCC"]

f = Featurizer()
X = f.transform(smiles_list)
```

## Create a chemical library
The repository contains ChEMBL 29 as data. You can create a library of precalculated features for this collection:

```python
from groverfeat.library import ReferenceLibrary

rl = ReferenceLibrary()
rl.save("reference_library.h5")
```

This creates an HDF5 file with two datasets, `Values` (corresponding to the features) and `Inputs` (corresponding to the SMILES string).
You can retrieve them as follows:

```python
X, smiles = rl.read("reference_library.h5")
```

Or, if you want to access these data from elsewhere, you can do:

```python
import h5py

with h5py.File("reference_library.h5", "r") as f:
    X = f["Values"][:]
    smiles = [inp.decode("utf-8") for inp in f["Inputs"]]
```

# About us
This repository is created and maintained by the Ersilia team.
The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization incorporated with the Charity Commission for England and Wales (number 1192266). Our mission is to reduce the imbalance in biomedical research productivity between countries by supporting research in underfunded settings.

You can support us via our [Open Collective](https:/opencollective.com/ersilia).
