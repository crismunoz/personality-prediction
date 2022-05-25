import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import numpy as np
import re
import pickle
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
# add parent directory to the path as well, if running from the finetune folder
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)

sys.path.insert(0, os.getcwd())

import utils.gen_utils as utils


def get_inputs(inp_dir, dataset, embed, embed_mode, mode, layer, n_hl):
    """Read data from pkl file and prepare for training."""
    data_x = []
    for chunk_id in range(2):
        file = open(
            inp_dir + dataset + "-" + embed + "-" + embed_mode + "-" + mode + "-" + str(chunk_id) +".pkl", "rb"
        )
        data = pickle.load(file)
        author_ids, data_x_1 = list(zip(*data))
        file.close()
        data_x+=data_x_1

    # alphaW is responsible for which BERT layer embedding we will be using
    if layer == "all":
        alphaW = np.full([n_hl], 1 / n_hl)

    else:
        alphaW = np.zeros([n_hl])
        alphaW[int(layer) - 1] = 1

    # just changing the way data is stored (tuples of minibatches) and
    # getting the output for the required layer of BERT using alphaW
    inputs = []
    n_batches = len(data_x)
    for ii in range(n_batches):
        inputs.extend(np.einsum("k,kij->ij", alphaW, data_x[ii]))

    inputs = np.array(inputs)
    return inputs


def training(dataset, dataset_trained, best_folds, inputs, hidden_dim):
    """Train MLP model for each trait on 10-fold corss-validtion."""
    if dataset == "kaggle":
        trait_labels = ["E", "N", "F", "J"]
    else:
        trait_labels = ["EXT", "NEU", "AGR", "CON", "OPN"]

    expdata = {}

    for trait_idx in range(len(trait_labels)):       
                    
        # Build the model
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Dense(50, input_dim=hidden_dim, activation="relu")
        )
        model.add(tf.keras.layers.Dense(n_classes))

        # Load weights
        model_dir = os.path.join(os.getcwd(),'models',dataset_trained)
        trait = trait_labels[trait_idx]
        checkpoint_filepath = os.path.join(model_dir,f'{trait}_{best_folds[trait]}','model')
        model.load_weights(checkpoint_filepath)

        # Predict the text
        x_train = inputs
        batch_size = 10
        num_batchs = (x_train.shape[0]+batch_size-1)//batch_size

        ypreds = []
        for ib in tqdm(range(num_batchs), total=num_batchs):
            start = ib*batch_size
            stop = (ib+1)*batch_size
            ypred = model.predict(x_train[start:stop], verbose=0)
            ypreds.append(ypred)
        
        ypreds = np.concat(ypreds, axix=0)

        expdata[trait_labels[trait_idx]] = ypreds

    df = pd.DataFrame.from_dict(expdata)
    return df


def logging(df, log_expdata=True):
    """Save results and each models config and hyper parameters."""
    (
        df["network"],
        df["lr"],
        df["batch_size"],
        df["epochs"],
        df["model_input"],
        df["embed"],
        df["layer"],
        df["mode"],
        df["embed_mode"],
        df["jobid"],
    ) = (
        network,
        lr,
        batch_size,
        epochs,
        MODEL_INPUT,
        embed,
        layer,
        mode,
        embed_mode,
        jobid,
    )

    pd.set_option("display.max_columns", None)
    print(df.head(5))

    # save the results of our experiment
    if log_expdata:
        Path(path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(path + "expdata.csv"):
            df.to_csv(path + "expdata.csv", mode="a", header=True)
        else:
            df.to_csv(path + "expdata.csv", mode="a", header=False)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)
        df.to_csv(path + "dataset_inference.csv", header=True)




if __name__ == "__main__":
    (
        inp_dir,
        dataset,
        lr,
        batch_size,
        epochs,
        _,
        embed,
        layer,
        mode,
        embed_mode,
        jobid,
    ) = utils.parse_args_inference()
    # embed_mode {mean, cls}
    # mode {512_head, 512_tail, 256_head_tail}
    log_expdata = False
    network = "MLP"
    MODEL_INPUT = "LM_features"
    dataset_trained = 'essays'
    
    if dataset_trained=='essays':
        best_folds = {'AGR': 2, 'CON': 6, 'EXT': 9, 'NEU': 7, 'OPN': 3}
    
    elif dataset_trained=='status':
        best_folds = {'AGR': 8, 'CON': 1, 'EXT': 1, 'NEU': 1, 'OPN': 1}

    print("{} : {} : {} : {} : {}".format(dataset, embed, layer, mode, embed_mode))
    n_classes = 2
    seed = jobid
    np.random.seed(seed)
    tf.random.set_seed(seed)

    start = time.time()
    path = "explogs/"

    if re.search(r"base", embed):
        n_hl = 12
        hidden_dim = 768

    elif re.search(r"large", embed):
        n_hl = 24
        hidden_dim = 1024

    inputs = get_inputs(inp_dir, dataset, embed, embed_mode, mode, layer, n_hl)
    df = training(dataset, dataset_trained, best_folds, inputs, hidden_dim)
    logging(df, log_expdata)
