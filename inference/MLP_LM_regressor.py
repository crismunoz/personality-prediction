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
    file = open(
        inp_dir + dataset + "-" + embed + "-" + embed_mode + "-" + mode + ".pkl", "rb"
    )
    data = pickle.load(file)
    author_ids, data_x, data_y = list(zip(*data))
    file.close()

    # alphaW is responsible for which BERT layer embedding we will be using
    if layer == "all":
        alphaW = np.full([n_hl], 1 / n_hl)

    else:
        alphaW = np.zeros([n_hl])
        alphaW[int(layer) - 1] = 1

    # just changing the way data is stored (tuples of minibatches) and
    # getting the output for the required layer of BERT using alphaW
    inputs = []
    targets = []
    ids = []
    n_batches = len(data_y)
    for ii in range(n_batches):
        inputs.extend(np.einsum("k,kij->ij", alphaW, data_x[ii]))
        targets.extend(data_y[ii])
        ids.extend(author_ids[ii])

    inputs = np.array(inputs)
    full_targets = np.array(targets)
    full_ids = np.array(ids)

    return inputs, full_targets,full_ids

def get_inputs_chunks(inp_dir, dataset, embed, embed_mode, mode, layer, n_hl):
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


def training(dataset, dataset_trained, best_folds, inputs, hidden_dim, n_classes):
    """Train MLP model for each trait on 10-fold corss-validtion."""
    if dataset == "kaggle":
        trait_labels = ["E", "N", "F", "J"]
    else:
        #trait_labels = ["EXT", "NEU", "AGR", "CON", "OPN"]
        trait_labels = ["EXT"]#, "NEU", "AGR", "CON", "OPN"]

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
        batch_size = 40
        num_batchs = (x_train.shape[0]+batch_size-1)//batch_size

        ypreds = []
        for ib in tqdm(range(num_batchs), total=num_batchs):
            start = ib*batch_size
            stop = (ib+1)*batch_size
            ypred = model.predict(x_train[start:stop], verbose=0)
            ypreds.append(ypred)
        
        ypreds = np.concatenate(ypreds, axis=0)

        expdata[trait_labels[trait_idx]] = np.reshape(ypreds,[-1])
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
        df.to_csv(path + f"dataset_inference_{dataset_trained}_{dataset}.csv", header=True)




if __name__ == "__main__":
    (
        inp_dir,
        dataset,
        dataset_trained,
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
    
    if dataset_trained=='status_regressor':
        best_folds = {'AGR': 3, 'CON': 3, 'EXT': 6, 'NEU': 1, 'OPN': 2}
    


    print("{} : {} : {} : {} : {}".format(dataset, embed, layer, mode, embed_mode))
    n_classes = 1
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
    df = training(dataset, dataset_trained, best_folds, inputs, hidden_dim, n_classes)
    logging(df, log_expdata)
