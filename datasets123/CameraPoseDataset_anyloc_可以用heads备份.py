from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
from util123 import utils
import ipdb as pdb
import torch

import os
def contains_test(labels_file):
    # 获取文件名部分
    filename = os.path.basename(labels_file)
    return "test" in filename

class CameraPoseDataset_anyloc(Dataset):
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
        super(CameraPoseDataset_anyloc, self).__init__()
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
        # print('self.img_paths = ',self.img_paths)
        pdb.set_trace()
        if len(self.img_paths) != len(self.poses):
            raise Exception('RGB file count does not match pose file count!')
    
        # ------------------- 取anyloc特征图 ----------------------------------------



        # ------------------取数组  allVectors_vlad.shape =  (20, 1, 49152)-----------------------------------------
        
        # 读取压缩文件
        print('labels_file = ',labels_file, 'contains_test(labels_file) = ',contains_test(labels_file))
        if contains_test(labels_file):
            new_filename_vlad= '/root/autodl-tmp/anyloc_vectors/allVectors_data_compressed_seq-01-aug.npz'
        else:
            new_filename_vlad= '/root/autodl-tmp/anyloc_vectors/allVectors_data_compressed_seq-02.npz'
        data = np.load(new_filename_vlad)

        # 获取保存的数组和文件名
        self.allVectors_vlad_loaded = data['ret_all'] #allVectors_vlad.shape =  (20, 1, 49152)
        img_fnames_loaded = data['img_fnames']

        # 打印加载的数据形状
        print(f'load from {new_filename_vlad} ')
        print('Loaded allVectors_vlad.shape = ', self.allVectors_vlad_loaded.shape) #Loaded allVectors_vlad.shape =  (1000, 1, 49152)
        # print('Loaded img_fnames:', img_fnames_loaded)
        pdb.set_trace()
    
        # ------------------取数组  allVectors_vlad.shape =  (20, 1, 49152)-------


        # 从 .npz 文件中加载数据
        print('labels_file = ',labels_file, 'contains_test(labels_file) = ',contains_test(labels_file))
        if contains_test(labels_file):
            print('load from test set data_compressed.npz ')
            data = np.load('/home/transposenet/data/data_compressed.npz')
            # data = np.load('/home/transposenet/data/allVectors_data_compressed_seq-01-aug.npz')
            
        else:
            print('load from train set data.npz ')
            data = np.load('/home/transposenet/data/data.npz')
        # pdb.set_trace()
        
        
        

        # 提取 ret_all 和 img_fnames
        ret_all = data['ret_all']
        img_fnames = data['img_fnames']

        # 如果 img_fnames 是字符串列表，它在保存为 .npz 文件时会被存储为一个 numpy 数组
        # 我们可以将其转换回 Python 列表
        img_fnames = img_fnames.tolist()

        # print("Data loaded successfully!")
        # print("加载出来后特征图 ret_all shape:", ret_all.shape)
        # print("img_fnames:", img_fnames)
        self.anyloc_dino_feature = ret_all.reshape(-1, 1536, 16, 16)  # squeeze后, ret_all.shape=(1000, 256, 1536)
        # print("加载出来后特征图 self.anyloc_dino_feature shape:", self.anyloc_dino_feature.shape) # (1000, 256, 1536)
        # pdb.set_trace()
  

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_rgb = imread(self.img_paths[idx]) #原本img。shape=  (480, 640, 3)
        # 复制图像到 img_show
        # print('原本img。shape= ', img.shape)#, 'img.device = ',img.device)
        img = self.anyloc_dino_feature[idx] #加载出来后特征图 img。shape=  (256, 1536)
        
        # print('加载出来后特征图 img。shape= ', img.shape) #, 'img.device = ',img.device)
        img_show = np.copy(img)
        # img_show = self.transform_show(img_show)
        pose = self.poses[idx]
        
        if self.transform:
            img_rgb = self.transform(img_rgb)
        img = torch.from_numpy(img)

        vlad = self.allVectors_vlad_loaded[idx]

        # print('123self.return_idx = ',self.return_idx)
        if self.return_idx:
            sample = {'img_dino': img, 'pose': pose, 'idx':idx, 'img':img_rgb, 'vlad':vlad}
        else:
            sample = {'img_dino': img, 'pose': pose, 'img_show':img_show, 'img':img_rgb, 'vlad':vlad}
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