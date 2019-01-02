#coding=utf-8
from __future__ import print_function
import os.path as osp
import numpy as np
import torch
import h5py

from utils import makedir_exist_ok

class CloudGermanDataset(torch.utils.data.Dataset):
    """
    cloud german dataset
    Args:
        root(string): root directory to save h5 file

    """
    training_file   = 'training.h5'
    valid_file      = 'validation.h5'
    test_file       = 'round1_test_a_20181109.h5'

    classes         =
    class_num       = len(classes)
    A_channel_num   = 10
    B_channel_num   = 8


    def __init__(self, root, datatype='train', inputtype='AB', transform=None, target_transform=None):
        assert datatype in ('train', 'valid', 'test'), "datatype: {} must in ('train', 'valid', 'test')".format(datatype)
        assert inputtype in ('A', 'B', 'AB'), "inputtype:{} must in ('A', 'B', 'AB')"
        self.root   = osp.expanduser(root)
        self.transform  = transform
        self.target_transform   = target_transform
        self.datatype   = datatype
        self.inputtype  = inputtype

        self.training_file_path = osp.join(self.root, self.training_file)
        self.test_file_path     = osp.join(self.root, self.test_file)
        self.valid_file_path    = osp.join(self.root, self.valid_file)    

        self._check_exists()
        self.dataA, self.dataB, self.targets = self._load_data()

    def _check_exists(self):
        assert osp.exists(self.training_file_path), "training file: {} not exist".format(self.training_file_path)
        assert osp.exists(self.valid_file_path), "valid file: {} not exist".format(self.valid_file_path)
        assert osp.exists(self.test_file_path), "test file: {} not exist".format(self.test_file_path)
        return True
    
    def _load_data(self):
        """
        load h5 data
        """
        if self.datatype == 'train':
            data_file_path  = self.training_file_path
            have_label  = True
        elif self.datatype == 'valid':
            data_file_path  = self.valid_file_path
            have_label  = True
        elif self.datatype == 'test':
            data_file_path  = self.test_file_path
            have_label  = False
        else:
            raise NotImplementedError
        content     = h5py.File(data_file_path, 'r')
        sen1_x, sen2_x  = content['sen1'], content['sen2']
        if have_label:
            label_y = content['label']
        else:
            label_y = [None] * len(sen1_x)
        return sen1_x, sen2_x, label_y

    def __getitem__(self, index):
        """
        get i-th item
        """
        if self.inputtype == 'A':
            input_x     = np.asarray(self.dataA[index, ...])
        elif self.inputtype == 'B':
            input_x     = np.asarray(self.dataB[index, ...])
        elif self.inputtype == 'AB':
            input_x_A   = np.asarray(self.dataA[index, ...])
            input_x_B   = np.asarray(self.dataB[index, ...])
            # channel维度在第0维
            input_x     = np.concatenate([input_x_A, input_x_B], axis=0)
        target_y        = self.targets[index]

        if self.transform is not None:
            input_x     = self.transform(input_x)
        if self.target_transform is not None and target_y is not None:
            target_y    = self.target_transform(target_y)
        return input_x, target_y

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in self.classes.items()}
    