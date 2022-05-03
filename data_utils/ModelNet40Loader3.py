import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

def load_data_part(root,partition,normal):
    """
    加载数据部分；
    
    """

    all_data = []
    all_label = []
    all_target = []
    str = '*_normal_resampled_1024_normal_%s.h5' if normal else '*_resampled_1024_%s.h5'
    #E:\code\pycharm\pointcloud\data_utils\data\../../data/modelnet40*hdf5_2048\*train*.h5
    for h5_name in glob.glob(os.path.join(root, str%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label



class ModelNet40(Dataset):
    def __init__(self, root, partition='train',normal=True):
        self.data, self.label = load_data_part(root,partition,normal)
        self.partition = partition


    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        if self.partition == 'train':
            np.random.shuffle(pointcloud)
        
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]