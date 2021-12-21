__version__ = "0.0.1"

import os
import numpy as np
from tqdm import tqdm
import subprocess
import csv
import tempfile
import shutil

PATH = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(PATH, "..", "model", "grover_large.pt")
REFERENCE_SMILES = os.path.join(PATH, "..", "data", "reference_library.csv")

EMBEDDING_SIZE = 5000

class Featurizer(object):
    def __init__(self,  chunksize: int = 1000):
        self.chunksize = chunksize

    def chunker(self, n):
        size = self.chunksize
        for i in range(0, n, size):
            yield slice(i, i + size)

    def _transform(self, smiles_list):

        tempfolder = tempfile.mkdtemp()
        datafile = os.path.join(tempfolder, "data.csv")
        featuresfile = os.path.join(tempfolder, "features.npz")
        fpsfile = os.path.join(tempfolder, "fp.npz")

        with open(datafile, "w") as f:
            write = csv.writer(f)
            write.writerow(["SMILES"])
            for smi in smiles_list:
                write.writerow([smi])
        cmd = "python {0}/scripts/save_features.py --data_path {1} --save_path {2} --features_generator rdkit_2d_normalized --restart".format(PATH, datafile, featuresfile)
        cmd2 = "python {0}/main.py fingerprint --data_path {1} --features_path {2} --checkpoint_path {3} --fingerprint_source both --output {4}".format(PATH, datafile, featuresfile, MODEL, fpsfile)
        subprocess.Popen(cmd, shell=True, env=os.environ).wait()
        subprocess.Popen(cmd2, shell=True, env=os.environ).wait()
        with open(fpsfile, "rb") as f:
            x = np.load(f)["fps"]
        shutil.rmtree(tempfolder)
        return x


    def transform(self, smiles_list):
        X = np.zeros((len(smiles_list), EMBEDDING_SIZE), np.float32)
        for chunk in tqdm(self.chunker(X.shape[0])):
            X[chunk] = self._transform(smiles_list[chunk])
        return X