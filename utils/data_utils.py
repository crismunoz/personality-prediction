import numpy as np
import pandas as pd
import re
import csv
import preprocessor as p
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import *
import math

from utils.author_100recent import get_100_recent_posts
import utils.dataset_processors as dataset_processors


class MyMapDataset(Dataset):
    def __init__(self, dataset, tokenizer, token_length, DEVICE, mode):
        if dataset == "essays":
            datafile = "data/essays/essays.csv"
            author_ids, input_ids, targets = dataset_processors.essays_embeddings(
                datafile, tokenizer, token_length, mode
            )
        elif dataset == "kaggle":
            datafile = "data/kaggle/kaggle.csv"
            author_ids, input_ids, targets = dataset_processors.kaggle_embeddings(
                datafile, tokenizer, token_length
            )
        elif dataset == "pandora":
            author_ids, input_ids, targets = dataset_processors.pandora_embeddings(
                datafile, tokenizer, token_length
            )
        elif dataset == "status":
            datafile = "data/myPersonality/mypersonality_final.csv"
            author_ids, input_ids, targets = dataset_processors.status_embeddings(
                datafile, tokenizer, token_length, mode
            )
        elif dataset == "status_regressor":
            datafile = "data/myPersonality/mypersonality_final.csv"
            author_ids, input_ids, targets = dataset_processors.status_regressor_embeddings(
                datafile, tokenizer, token_length, mode
            )

        author_ids = torch.from_numpy(np.array(author_ids)).long().to(DEVICE)
        input_ids = torch.from_numpy(np.array(input_ids)).long().to(DEVICE)
        targets = torch.from_numpy(np.array(targets))

        if dataset in ["pandora" , "status_regressor"]:
            targets = targets.float().to(DEVICE)
        else:
            targets = targets.long().to(DEVICE)

        self.author_ids = author_ids
        self.input_ids = input_ids
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.author_ids[idx], self.input_ids[idx], self.targets[idx])


class MyNewMapDataset(Dataset):
    def __init__(self, dataset, datafile, tokenizer, token_length, DEVICE, mode, chunk_id, total_chunks):
        
        if dataset=='generated_text':
            author_ids, input_ids = dataset_processors.generated_text_embeddings(datafile, tokenizer, token_length, mode, chunk_id, total_chunks)

        author_ids = torch.from_numpy(np.array(author_ids)).long().to(DEVICE)
        input_ids = torch.from_numpy(np.array(input_ids)).long().to(DEVICE)

        self.author_ids = author_ids
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.author_ids[idx], self.input_ids[idx])