from torchvision import datasets, transforms
from base import BaseDataLoader

from data_loader.cloud_german_dataset import CloudGermanDataset, CloudGermanDataset2, CloudGermanDataset3, CloudGermanDataset4

import pdb

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class CloudGermanLoader(BaseDataLoader):
    """
    cloud german data loading
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, datatype='train', inputtype='AB'):
        self.data_dir   = data_dir
        # 暂时
        trsfm           = None
        self.dataset    = CloudGermanDataset(root=self.data_dir, \
            datatype=datatype, inputtype=inputtype, transform=trsfm)
        if datatype == 'valid' or datatype == 'test':
            assert shuffle is False, "shuffle:{} must be False when datatype is {}".format(shuffle, datatype)
        # validation_split is None because train and valid have been splitted ahead
        super(CloudGermanLoader, self).__init__(self.dataset, batch_size, shuffle, 0.0, num_workers)


class CloudGermanLoader2(BaseDataLoader):
    """
    cloud german data loading
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, datatype='train', inputtype='AB'):
        self.data_dir   = data_dir
        # 暂时
        trsfm           = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomResizedCrop(32, scale=(0.9,1.1), ratio=(0.8,1.25)),
                    transforms.ToTensor()
            ])
        self.dataset    = CloudGermanDataset2(root=self.data_dir, \
            datatype=datatype, inputtype=inputtype, transform=trsfm)
        if datatype == 'valid' or datatype == 'test':
            assert shuffle is False, "shuffle:{} must be False when datatype is {}".format(shuffle, datatype)
        # validation_split is None because train and valid have been splitted ahead
        super(CloudGermanLoader2, self).__init__(self.dataset, batch_size, shuffle, 0.0, num_workers)
        
class CloudGermanLoader3(BaseDataLoader):
    """
    cloud german data loading
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, datatype='train', inputtype='AB'):
        self.data_dir   = data_dir
        # 暂时
        trsfm           = None
        self.dataset    = CloudGermanDataset3(root=self.data_dir, \
            datatype=datatype, inputtype=inputtype, transform=trsfm)
        if datatype == 'valid' or datatype == 'test':
            assert shuffle is False, "shuffle:{} must be False when datatype is {}".format(shuffle, datatype)
        # validation_split is None because train and valid have been splitted ahead
        super(CloudGermanLoader3, self).__init__(self.dataset, batch_size, shuffle, 0.0, num_workers)
        
class CloudGermanLoader4(BaseDataLoader):
    """
    cloud german data loading
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, datatype='train', inputtype='AB'):
        self.data_dir   = data_dir
        # 暂时
        trsfm           = None
        self.dataset    = CloudGermanDataset4(root=self.data_dir, \
            datatype=datatype, inputtype=inputtype, transform=trsfm)
        if datatype == 'valid' or datatype == 'test':
            assert shuffle is False, "shuffle:{} must be False when datatype is {}".format(shuffle, datatype)
        # validation_split is None because train and valid have been splitted ahead
        super(CloudGermanLoader4, self).__init__(self.dataset, batch_size, shuffle, 0.0, num_workers)
