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


def convertNumpy(df):
	x = df[df.columns[3:]].values[::10, :]
	return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_dataset(args):

    valid_split_rate = args.valid_split_rate

    try:
        assert args.dataset in ['SWaT', 'SMD', 'SMAP_MSL', 'COLLECTOR', 'synthetic', 'WADI']
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

    elif args.dataset == 'synthetic':
        train_raw = pd.read_csv(args.train_dir, header=None).to_numpy()
        split = 10000

        trainset = train_raw[:, :split].reshape(split, -1)
        testset = train_raw[:, split:].reshape(split, -1)      

        valid_split_index = int(len(trainset) * valid_split_rate)    
        validset = trainset[valid_split_index:]
        trainset = trainset[:valid_split_index]  

        label_raw = pd.read_csv(args.test_label_dir, header=None)
        label_raw[0] -= split

        test_label = np.zeros(testset.shape)
        for i in range(label_raw.shape[0]):
            point = int(label_raw.values[i][0])
            # 컬럼 인덱스를 정수형으로 변환하여 할당
            col_indices = label_raw.values[i][1:].astype(int)
            test_label[max(0, point-30):min(len(test_label), point+30), col_indices] = 1        
        for i in range(label_raw.shape[0]):
            point = label_raw.values[i][0]
            test_label[point-30:point+30, label_raw.values[i][1:]] = 1
        testset += test_label * np.random.normal(0.75, 0.1, testset.shape)

        test_label = test_label.max(axis=1)

    elif args.dataset == 'WADI':

        train_raw = pd.read_csv(args.train_path, skiprows=1000, nrows=2e5)
        test_raw = pd.read_csv(args.test_path)
        ls = pd.read_csv(args.test_label_path)

        train_raw.dropna(how='all', inplace=True); test_raw.dropna(how='all', inplace=True)
        train_raw.fillna(0, inplace=True); test_raw.fillna(0, inplace=True)

        valid_split_index = int(len(train_raw) * valid_split_rate)    
        validset = train_raw[valid_split_index:]
        trainset = train_raw[:valid_split_index]  

        test_raw['Time'] = test_raw['Time'].astype(str)
        test_raw['Time'] = pd.to_datetime(test_raw['Date'] + ' ' + test_raw['Time'], dayfirst=True)
        labels = test_raw.copy(deep = True)

        for i in test_raw.columns.tolist()[3:]: labels[i] = 0
        for i in ['Start Time', 'End Time']: 
            ls[i] = ls[i].astype(str)
            ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i], dayfirst=True)
        for index, row in ls.iterrows():
            to_match = row['Affected'].split(', ')
            matched = []
            for i in test_raw.columns.tolist()[3:]:
                for tm in to_match:
                    if tm in i: 
                        matched.append(i); break			
            st, et = str(row['Start Time']), str(row['End Time'])
            labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1

        trainset, validset, testset, test_label = convertNumpy(trainset), convertNumpy(validset), convertNumpy(test_raw), convertNumpy(labels)
        test_label = (test_label > 0).astype(np.float32) 

        # 혹은 라벨이 여러 컬럼일 경우 하나라도 1이면 1이 되도록 통합 (Anomaly Detection 기준)
        if test_label.ndim > 1:
            test_label = test_label.max(axis=1)

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
        self.model_type = self.config['model_type']

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

        if is_train:
            # Train mode : 설정된 stride 사용
            current_stride = self.slide_stride
        else:
            # Test/Val mode : model_type에 따라 분기
            if self.model_type == 'reconstruction':
                current_stride = self.slide_window
            else:
                # Forecasting (or others): 1칸씩 이동 (Sliding Window)
                current_stride = 1

        # 결정된 stride 적용
        rang = range(self.slide_window, self.total_time_len, current_stride)
       
        for i in rang:
            window = data[:, i-self.slide_window:i] 
            if self.model_type == 'reconstruction':
                # Reconstruction: Target은 Input과 동일, Label도 Window 전체 구간
                target = window
                label = labels[i-self.slide_window:i]
            else:    
                # Forecasting: Target은 Window 다음 시점(i), Label도 해당 시점(i)    
                target = data[:, i] 
                label = labels[i] 

            x_arr.append(window)
            y_arr.append(target)
            labels_arr.append(label)

        if not x_arr:
            return torch.empty(0, self.node_num, self.slide_window).float(), \
                   torch.empty(0, self.node_num).float(), \
                   torch.empty(0).float()
        
        # List -> Tensor 변환
        # x = torch.stack(x_arr).contiguous()
        # y = torch.stack(y_arr).contiguous()
        # labels = torch.stack(labels_arr).contiguous()

        # 1. 데이터가 Tensor인 경우 (stack 사용)
        if isinstance(x_arr[0], torch.Tensor):
            x = torch.stack(x_arr).contiguous()
            y = torch.stack(y_arr).contiguous()
            labels = torch.stack(labels_arr).contiguous()
            
        # 2. 데이터가 Numpy Array인 경우 (np.array로 합친 후 변환)
        else:
            x = torch.from_numpy(np.array(x_arr)).float().contiguous()
            y = torch.from_numpy(np.array(y_arr)).float().contiguous()
            labels = torch.from_numpy(np.array(labels_arr)).float().contiguous()

        return x, y, labels

    def __getitem__(self, idx):
        index = self.valid_indices[idx]
        
        feature = self.x[index].float()
        y = self.y[index].float()
        label = self.labels[index].float()

        edge_index = self.edge_index.long()
        
        return feature, y, label, edge_index
