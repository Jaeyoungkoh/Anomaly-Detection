import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from typing import List, Optional, Union


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class CollectorSegLoader(object):
    def __init__(self, args, dataset_folder, win_size, step, mode="train"):
        self.args = args
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.boundary_index_train = self.args.boundary_index_train
        self.boundary_index_test = self.args.boundary_index_test
        self.scaler_type = self.args.scaler_type
        MinMaxScaler()

        # 1. 데이터 로드 및 스케일링
        train_data_raw = pd.read_csv(os.path.join(dataset_folder, self.args.train_csv)).values
        test_data_raw = pd.read_csv(os.path.join(dataset_folder, self.args.test_csv)).values
        self.test_labels = pd.read_csv(os.path.join(dataset_folder, self.args.label_csv), header=None).values

        # 2. 데이터를 먼저 훈련/검증으로 분리
        val_start_index = int(len(train_data_raw) * (1 - self.args.val_split))
        train_subset = train_data_raw[:val_start_index]
        val_subset = train_data_raw[val_start_index:]

        # 3. args.scaler_type에 따라 스케일러 초기화
        if self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
            if self.mode == 'train': print("Scaler: MinMaxScaler")

        elif self.scaler_type == 'standard':
            self.scaler = StandardScaler()
            if self.mode == 'train': print("Scaler: StandardScaler")

        elif self.scaler_type is None:
            self.scaler = None
            if self.mode == 'train': print("Scaler: None")       

        # 4. scaler 적용 (train 기준)
        if self.scaler is not None:
            self.scaler.fit(train_subset)
            self.train = self.scaler.transform(train_subset)
            self.val = self.scaler.transform(val_subset)
            self.test = self.scaler.transform(test_data_raw)
        else:
            self.train = train_subset
            self.val = val_subset
            self.test = test_data_raw

        if self.mode == "train" :
            print("train set shape: ", self.train.shape)
            print("val set shape: ", self.val.shape)
            print("test set shape: ", self.test.shape)
            print("test set label shape: ", self.test_labels.shape)

        # 5. boundary 기반 유효 인덱스 생성
        train_step = 1
        val_step = 1
        test_step = 1
        if self.mode == 'test':
            test_step = self.win_size

        self.valid_indices_train = self._generate_valid_indices(self.train, self.boundary_index_train, train_step)
        self.valid_indices_val = self._generate_valid_indices(self.val, self.boundary_index_train, val_step)
        self.valid_indices_test = self._generate_valid_indices(self.test, self.boundary_index_test, test_step)

    # 유효 인덱스 추출 함수
    def _generate_valid_indices(self, data, boundary_index, step) :
        max_index = len(data) - self.win_size + 1
        valid_indices = []

        for i in range(0, max_index, step):
            output_end = i + self.win_size - 1 

            if output_end >= len(data):
                 continue 
            
            # boundary crossing 여부 확인
            if boundary_index is not None:
                is_crossing = any((i < b <= output_end) for b in boundary_index)
                if is_crossing:
                    continue

            valid_indices.append(i)

        return valid_indices

    def __getitem__(self, index):
        if self.mode == "train":
            idx = self.valid_indices_train[index]
            x = self.train[idx:idx + self.win_size]
            y = self.test_labels[0:self.win_size]

        elif self.mode == 'val':
            idx = self.valid_indices_val[index]
            x = self.val[idx:idx + self.win_size]
            y = self.test_labels[0:self.win_size]

        elif self.mode == 'test':
            idx = self.valid_indices_test[index]           
            x = self.test[idx:idx + self.win_size]
            y = self.test_labels[idx:idx + self.win_size]

        return np.float32(x), np.float32(y)

    def __len__(self):
        if self.mode == "train":
            return len(self.valid_indices_train)
        elif self.mode == 'val':
            return len(self.valid_indices_val)
        elif self.mode == 'test':
            return len(self.valid_indices_test)

def get_loader_segment(args, mode='train'):

    dataset = args.dataset
    dataset_folder = args.dataset_folder
    win_size = args.win_size
    batch_size = args.batch_size
    step=1

    if (dataset == 'SMD'):
        dataset = SMDSegLoader(dataset_folder, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(dataset_folder, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(dataset_folder, win_size, 1, mode)
    elif (dataset == 'SWaT'):
        dataset = PSMSegLoader(dataset_folder, win_size, 1, mode)
    elif (dataset == 'COLLECTOR'):
        dataset = CollectorSegLoader(args, dataset_folder, win_size, 1, mode)        

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
