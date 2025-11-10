from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import timedelta
import dateutil
from typing import Optional, Union, List 

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
        elif self.model_type in ['forecasting', 'mix']:
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
        y = None 
        z = None

        if self.model_type == 'reconstruction':
            # item["ts"] = self.ts[i:last]
            y = self.tag_values[i:last]
            z = np.zeros_like(x.shape[0], dtype=np.float32)
            if self.with_attack:
                z = self.attacks[i:last]

        elif self.model_type in ['prediction', 'mix']:
            # item["ts"] = self.ts[last]
            y = self.tag_values[last]
            z = 0.0
            if self.with_attack:
                z = self.attacks[last]

        return np.float32(x), np.float32(y), np.float32(z), 0


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
        test_label_info = pd.read_csv(args.test_label_path, index_col=0).loc[args.sub_data_name]
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


def construct_data(data_np: np.ndarray, labels: Union[int, list, np.ndarray] = 0):
    """
    [단순화된 버전]
    이미 스케일링된 (Time, Features) NumPy 배열을 GDN의 입력 형식인
    (Features, Time) 리스트로 변환하고 라벨을 추가합니다.
    """
    
    # (Time, Features) -> (Features, Time)로 전치(Transpose) 후 리스트로 변환
    res = data_np.T.tolist()
    
    # 행 수(샘플 수)
    sample_n = data_np.shape[0] # (Time)
    
    if sample_n == 0:
        print("Warning: construct_data received empty data.")
        return []

    # res 마지막 항에 labels 추가
    if isinstance(labels, int):
        res.append([labels] * sample_n) # train일 경우 [0,0,...,0]
    elif len(labels) == sample_n:
        res.append(list(labels)) # test일 경우 labels 추가
    else:
        raise ValueError(f"Invalid label format or length mismatch. Data len: {sample_n}, Label len: {len(labels)}")

    return res
    # (scaler_list를 더 이상 반환하지 않음)


class SlidingWindowDataset(Dataset):
    """
    GDN을 위한 SlidingWindowDataset
    (사용자가 제공한 원본 코드)
    """
    def __init__(self, 
                 raw_data, 
                 edge_index, 
                 mode='train', 
                 config = None, 
                 boundary_index: Optional[Union[int, List[int]]]=None
                 ):
        
        self.raw_data = raw_data
        self.edge_index = edge_index      
        self.mode = mode        
        self.config = config
        self.slide_window = self.config['slide_window']
        self.slide_stride = self.config['slide_stride']

        data = raw_data[:-1] # 마지막 항(label list) 제외
        labels = raw_data[-1] # 마지막 항

        self.node_num = len(data)
        self.total_time_len = len(data[0]) if self.node_num > 0 else 0

        # to tensor
        data = torch.tensor(data).float() # torch.float64
        labels = torch.tensor(labels).float() # torch.float64

        self.x, self.y, self.labels = self.process(data, labels)

        # (boundary_index 관련 로직은 원본 코드를 따름)
        if isinstance(boundary_index, int):
            self.boundary_index = [boundary_index]
        else:
            self.boundary_index = boundary_index

        if boundary_index is not None:
             print("Warning: boundary_index logic not fully implemented. Using simpler range.")
             # process 함수에서 이미 샘플을 다 만들었으므로, self.x의 길이를 사용
             self.valid_indices = list(range(len(self.x)))
        else:
             # process 함수에서 이미 샘플을 다 만들었으므로, self.x의 길이를 사용
            self.valid_indices = list(range(len(self.x)))

        print(f"{self.mode} : # of valid windows: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)


    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []

        is_train = self.mode == 'train'

        rang = range(self.slide_window, self.total_time_len, self.slide_stride) if is_train else range(self.slide_window, self.total_time_len)
        
        for i in rang:
            window = data[:, i-self.slide_window:i] 
            target = data[:, i] 
            label = labels[i] 

            x_arr.append(window)
            y_arr.append(target)
            labels_arr.append(label)

        if not x_arr:
            return torch.empty(0, self.node_num, self.slide_window).float(), \
                   torch.empty(0, self.node_num).float(), \
                   torch.empty(0).float()

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()
        labels = torch.Tensor(labels_arr).contiguous()

        return x, y, labels

    def __getitem__(self, idx):
        index = self.valid_indices[idx]
        
        feature = self.x[index].float()
        y = self.y[index].float()
        label = self.labels[index].float()

        edge_index = self.edge_index.long()

        return feature, y, label, edge_index
