#coding=utf-8
from __future__ import print_function
import os.path as osp
import numpy as np
import torch
import h5py, pdb

from data_loader.utils import makedir_exist_ok

def horizontal_flip_np(input_x):
#    return np.flip(input_x, axis=2)
#    return input_x[:, :, ::-1]
    return np.flip(input_x, axis=2).copy()
def vertical_flip_np(input_x):
#    return np.flip(input_x, axis=1)
#    return input_x[:, ::-1, :]
    return np.flip(input_x, axis=1).copy()

class CloudGermanDataset(torch.utils.data.Dataset):
    """
    cloud german dataset
    Args:
        root(string): root directory to save h5 file
    """
    training_file   = 'training.h5'
    valid_file      = 'validation.h5'
    test_file       = 'round1_test_a_20181109.h5'

    classes         = ['LCZ-1', 'LCZ-2','LCZ-3','LCZ-4','LCZ-5','LCZ-6','LCZ-7','LCZ-8','LCZ-9','LCZ-10',
                        'LCZ-A','LCZ-B','LCZ-C','LCZ-D','LCZ-E','LCZ-F','LCZ-G']
    class_num       = len(classes)
    A_channel_num   = 8
    B_channel_num   = 10
    mean_A  = [-3.59122410e-05, -7.65856054e-06, 5.93738526e-05, 2.51662314e-05, \
                4.42011066e-02, 2.57610271e-01, 7.55674349e-04, 1.35034668e-03]
    mean_B  = [1.23756961e-01, 1.09277464e-01, 1.01085520e-01, 1.14239862e-01, 1.59265669e-01, 1.81472360e-01, \
                1.74574031e-01, 1.95016074e-01, 1.54284689e-01, 1.09050507e-01]
    std_A   = [0.14552695, 0.14552102, 0.35118144, 0.35083747, 0.10879789, \
                0.94647736, 0.14515599, 0.09256628]
    std_B   = [0.01514473, 0.01787005, 0.0237855, 0.02031125, 0.02770328, 0.03284229, \
                0.03837105, 0.03622275, 0.02952162, 0.02598372]
    USE_INPUT_NORMALIZE = True

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
        self.dataProb   = np.random.random((len(self), ))

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
            label_y = np.argmax(np.asarray(content['label']), axis=1)
        else:
            label_y = [None] * len(sen1_x)

#        USE_NUM = 2048
#        return sen1_x[:USE_NUM, ...], sen2_x[:USE_NUM, ...], label_y[:USE_NUM, ...]
        return sen1_x, sen2_x, label_y

    def _stat_mean_std(self):
        """
        statistics on training data for mean and std
        """
        total_num   = len(self)
        channel_num = 18
        mean_arr    = np.zeros((total_num, channel_num))
        std_arr     = np.zeros((total_num, channel_num))
        for i in range(total_num):
            if i % 5000 == 0:
                print("{}/{} stated".format(i, total_num))
            input_x, _  = self.__getitem__(i)
            for j in range(channel_num):
                ij_data = input_x[j, ...]
                mean_arr[i, j]  = np.mean(ij_data)
                std_arr[i, j]   = np.std(ij_data)
        print("mean of mean_arr:(shape is {}) is {}".format(np.mean(mean_arr, axis=0), mean_arr.shape))
        print("mean of std_arr:(shape is {}) is {}".format(np.mean(std_arr, axis=0), std_arr.shape))

    def _stat_cls(self):
        """
        statistics on training data for class num
        """
        total_num   = len(self)
        cls_num_arr = np.zeros((self.class_num, ))
        for i in range(total_num):
            if i % 5000 == 0:
                print("{}/{} stated".format(i, total_num))
            _, y_label  = self.__getitem__(i)
            cls_num_arr[y_label] += 1.0
        print("class num of training set is: {}".format(cls_num_arr))

    def __getitem__(self, index):
        """
        get i-th item
        """
#        if self.inputtype == 'A':
#            # 原始的dataA数据是 [sample_num, 32, 32, 10]
#            input_x     = np.asarray(self.dataA[index, ...], dtype=np.float32).transpose((2,0,1))
#        elif self.inputtype == 'B':
#            # 原始的dataA数据是 [sample_num, 32, 32, 8]
#            input_x     = np.asarray(self.dataB[index, ...], dtype=np.float32).transpose((2,0,1))
#        elif self.inputtype == 'AB':
            
        input_x_A   = np.asarray(self.dataA[index, ...], dtype=np.float32).transpose((2,0,1))
        input_x_B   = np.asarray(self.dataB[index, ...], dtype=np.float32).transpose((2,0,1))
        
        if self.USE_INPUT_NORMALIZE:
            for j in range(self.A_channel_num):
                j_val   = input_x_A[j, ...]
                j_val   = (j_val - self.mean_A[j]) / self.std_A[j]
                input_x_A[j, ...]   = j_val
            for j in range(self.B_channel_num):
                j_val   = input_x_B[j, ...]
                j_val   = (j_val - self.mean_B[j]) / self.std_B[j]
                input_x_B[j, ...]   = j_val

        # channel维度在第0维
        input_x     = np.concatenate([input_x_A, input_x_B], axis=0)
        target_y        = self.targets[index]

        USE_transform   = True and self.datatype == 'train'
        if USE_transform:
            if self.dataProb[index] > 0.5:
                input_x = horizontal_flip_np(input_x)
                if self.dataProb[index] > 0.9:
                    input_x = vertical_flip_np(input_x)

#        if self.transform is not None:
#            input_x     = self.transform(input_x)
#        if self.target_transform is not None and target_y is not None:
#            target_y    = self.target_transform(target_y)
        if self.datatype == 'test':
            return input_x
        return input_x, target_y

    def __len__(self):
        return len(self.dataA)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}
