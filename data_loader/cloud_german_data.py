#coding=utf-8
# refer to http://tianchi-tum.oss-eu-central-1.aliyuncs.com/analysis.ipynb for data reading

import os.path as osp
import h5py
import init_paths

def read_h5_data(h5file_path):
    """
    读取 h5 文件
    """
    assert osp.exists(h5file_path), "{} not exist".format(h5file_path)
    content     = h5py.File(h5file_path, 'r')
    return content

def explore_h5_data(h5_data, data_name=''):
    print("-"*40)
    print("explore data {}".format(data_name))
    print("\t"+str(h5_data.keys()))
    # h5_data is a dict
    for name, val in h5_data.items():
        print("\t\t key: {}, val.shape:{}".format(name, val.shape))
    print("-"*60)

def main():
    DATA_DIR            = osp.join(init_paths.ROOT_DIR, 'data/tianchi/cloud_german')
    train_h5_file_path  = osp.join(DATA_DIR, 'training.h5')
    val_h5_file_path    = osp.join(DATA_DIR, 'validation.h5')
    train_h5_data       = read_h5_data(train_h5_file_path)
    val_h5_data         = read_h5_data(val_h5_file_path)
    explore_h5_data(val_h5_data, 'valid')
    explore_h5_data(train_h5_data, 'train.full')

if __name__ == "__main__":
    main()
