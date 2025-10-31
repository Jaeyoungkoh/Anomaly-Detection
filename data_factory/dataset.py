from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import timedelta
import dateutil

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

class BuildDataset(Dataset):
  
    def __init__(self, data, window_size, slide_size, attacks, model_type):
        
        self.ts = np.arange(len(data))
        self.tag_values = np.array(data, dtype=np.float32)
        self.window_size = window_size
        self.model_type = model_type

        self.valid_idxs = []
        if self.model_type == 'reconstruction':
            for L in range(0, len(self.ts) - window_size + 1, slide_size):
                R = L + window_size - 1
                try:
                    if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(
                            self.ts[L]
                    ) == timedelta(seconds=window_size - 1):
                        self.valid_idxs.append(L)
                except:
                    if self.ts[R] - self.ts[L] == window_size - 1:
                        self.valid_idxs.append(L)
        elif self.model_type == 'prediction':
            for L in range(0, len(self.ts) - window_size, slide_size):
                R = L + window_size
                try:
                    if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(
                            self.ts[L]
                    ) == timedelta(seconds=window_size):
                        self.valid_idxs.append(L)
                except:
                    if self.ts[R] - self.ts[L] == window_size:
                        self.valid_idxs.append(L)

        print(f"# of valid windows: {len(self.valid_idxs)}")

        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.with_attack = True
        else:
            self.with_attack = False

    def __len__(self):
        return len(self.valid_idxs)

    def __getitem__(self, idx: str) -> dict:
        i = self.valid_idxs[idx]
        last = i + self.window_size

        x = self.tag_values[i:last]
        z = None # z를 None으로 초기화

        if self.model_type == 'reconstruction':
            # item["ts"] = self.ts[i:last]
            y = self.tag_values[i:last]
            if self.with_attack:
                z = self.attacks[i:last]

        elif self.model_type == 'prediction':
            # item["ts"] = self.ts[last]
            y = self.tag_values[last]
            if self.with_attack:
                z = self.attacks[last]

        return np.float32(x), np.float32(y), np.float32(z)


def load_dataset(args):

    valid_split_rate = args.valid_split_rate

    try:
        assert args.dataset in ['SWaT', 'SMD', 'SMAP_MSL', 'COLLECTOR']
    except AssertionError as e:
        raise ValueError(f"Invalid dataname: {args.dataset}")

    if args.dataset == 'SWaT':
        trainset = pd.read_pickle(args.train_path).drop(['Normal/Attack', ' Timestamp'], axis=1).to_numpy()
        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset[valid_split_index:]
        trainset = trainset[:valid_split_index]
        testset = pd.read_pickle(args.test_path)
        test_label = testset['Normal/Attack']
        test_label[test_label == 'Normal'] = 0
        test_label[test_label != 0] = 1
        testset = testset.drop(['Normal/Attack', ' Timestamp'], axis=1).to_numpy()

    elif args.dataset == 'SMD':
        trainset = np.loadtxt(os.path.join(args.train_dir, f'{args.sub_data_name}.txt'),
                              delimiter=',', 
                              dtype=np.float32)
        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset[valid_split_index:]
        trainset = trainset[:valid_split_index]
        testset = np.loadtxt(os.path.join(args.test_dir, f'{args.sub_data_name}.txt'),
                             delimiter=',',
                             dtype=np.float32)
        test_label = np.loadtxt(os.path.join(args.test_label_dir, f'{args.sub_data_name}.txt'),
                                delimiter=',',
                                dtype=int)

    elif args.dataset == 'SMAP_MSL':
        trainset = np.load(os.path.join(args.train_dir, f'{args.sub_data_name}.npy'))
        testset = np.load(os.path.join(args.test_dir, f'{args.sub_data_name}.npy'))
        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset[valid_split_index:]
        trainset = trainset[:valid_split_index]
        test_label_info = pd.read_csv(args.test_label_dir, index_col=0).loc[args.sub_data_name]
        test_label = np.zeros([int(test_label_info.num_values)], dtype=int)

        for i in eval(test_label_info.anomaly_sequences):
            if type(i) == list:
                test_label[i[0]:i[1] + 1] = 1
            else:
                test_label[i] = 1

    elif args.dataset == 'COLLECTOR':
        trainset = pd.read_csv(args.train_path).values
        testset = pd.read_csv(args.test_path).values

        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset[valid_split_index:]
        trainset = trainset[:valid_split_index]
        test_label = pd.read_csv(args.test_label_path, header=None).values

    return trainset, validset, testset, test_label