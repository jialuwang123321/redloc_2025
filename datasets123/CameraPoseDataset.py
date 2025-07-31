from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
from util123 import utils
import ipdb as pdb

class CameraPoseDataset(Dataset):
    """
        A class representing a dataset of images and their poses
    """

    def __init__(self, dataset_path, labels_file, data_transform=None,skipvim=1, tab=None, return_idx =False):
        """
        :param dataset_path: (str) the path to the dataset
        :param labels_file: (str) a file with images and their path labels
        :param data_transform: (Transform object) a torchvision transform object
        :return: an instance of the class
        """
        super(CameraPoseDataset, self).__init__()
        self.img_paths, self.poses = read_labels_file(labels_file, dataset_path)
        # self.dataset_size = self.poses.shape[0]
        self.transform = data_transform
        # skipvim =100
        self.trainskip = skipvim
        self.tab = tab
        self.return_idx = return_idx
        print('self.return_idx = ',self.return_idx)
        self.transform_show = utils.show_transforms.get('baseline')
        
        # trainskip 
        
        frame_idx = np.arange(len(self.img_paths))
        frame_idx = frame_idx[::self.trainskip]
        self.dataset_size = len(frame_idx)

        print('after. frame_idx = ',frame_idx)
        print('len(frame_idx) = ',len(frame_idx))  
        print('self.tab = ',self.tab) 


        # train and test 部分数据集
        if self.tab is not None:
            assert self.vab is None, "tab有值的时候，必须保证vab是None"
            self.ta = self.tab[0]                
            self.tb = self.tab[1] 
            if self.ta!=0:
                if self.tb>self.ta: #tab0.5-0.7，那么取0.5-0.7部分的训练集
                    frame_idx_tmp = frame_idx[int(len(frame_idx)*self.ta):int(len(frame_idx)*self.tb)]
                    frame_idx = frame_idx_tmp
                else:    
                    frame_idx_tmp = frame_idx[:int(len(frame_idx)*self.ta)]
                    frame_idx = frame_idx_tmp
            else:
                assert self.tb !=0, "当tab不是None,必须保证ta和tb至少有一个不是0"
                frame_idx_tmp = frame_idx[-int(len(frame_idx)*self.tb):]
                frame_idx = frame_idx_tmp

                
        self.img_paths = [self.img_paths[i] for i in frame_idx]
        self.poses = [self.poses[i] for i in frame_idx]
        print('len(self.poses) = ',len(self.poses))
        print('len(self.img_paths) = ',len(self.img_paths))
        if len(self.img_paths) != len(self.poses):
            raise Exception('RGB file count does not match pose file count!')
    

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img = imread(self.img_paths[idx])
        # 复制图像到 img_show
        img_show = np.copy(img)
        img_show = self.transform_show(img_show)
        pose = self.poses[idx]
        if self.transform:
            img = self.transform(img)

        # print('123self.return_idx = ',self.return_idx)
        if self.return_idx:
            sample = {'img': img, 'pose': pose, 'idx':idx}
        else:
            sample = {'img': img, 'pose': pose, 'img_show':img_show}
        return sample
    
    def get_img_path(self, idx):
        return self.img_paths[idx]


def read_labels_file(labels_file, dataset_path):
    # labels_file=  /home/transposenet/datasets123/7Scenes/abs_7scenes_pose.csv_heads_train.csv
    df = pd.read_csv(labels_file)
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    return imgs_paths, poses