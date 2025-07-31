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

class CameraPoseDataset_anyloc_cambridge(Dataset):
    """
        A class representing a dataset of images and their poses
    """

    def __init__(self, dataset_path, labels_file, data_transform=None,skipvim=1, tab=None, return_idx =False, scene=None, results_type=None):
        """
        :param dataset_path: (str) the path to the dataset
        :param labels_file: (str) a file with images and their path labels
        :param data_transform: (Transform object) a torchvision transform object
        :return: an instance of the class
        """
        super(CameraPoseDataset_anyloc_cambridge, self).__init__()
        self.img_paths, self.poses = read_labels_file(labels_file, dataset_path)
        # self.dataset_size = self.poses.shape[0]
        self.transform = data_transform
        # skipvim =100
        self.trainskip = skipvim
        self.tab = tab
        self.return_idx = return_idx
        # print('self.return_idx = ',self.return_idx)
        self.transform_show = utils.show_transforms.get('baseline')
        self.scene = scene
        self.results_type = results_type
        print('self.scene = ',scene)
        print('self.results_type = ',self.results_type )
        #pdb.set_trace()

        
        # trainskip 
        
        frame_idx = np.arange(len(self.img_paths))
        frame_idx = frame_idx[::self.trainskip]
        self.dataset_size = len(frame_idx)

        print('after. frame_idx = ',frame_idx)
        print('len(frame_idx) = ',len(frame_idx))  
        print('self.tab = ',self.tab) 
        #pdb.set_trace()


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
        #pdb.set_trace()
        if len(self.img_paths) != len(self.poses):
            raise Exception('RGB file count does not match pose file count!')


        # ------- -------取数组  allVectors_vlad.shape =  (20, 1, 49152)-----------------------------
        print('labels_file = ',labels_file, 'contains_test(labels_file) = ',contains_test(labels_file))
        if contains_test(labels_file):
            # dark      
            if self.results_type==1:          
                new_filename_vlad_1= f'/home/transposenet/data/anyloc_cambridge/Cambridge_{self.scene}/descriptors/allVectors_data_compressed_test_rgb-aug.npz'

            # good
            elif self.results_type==2:
                new_filename_vlad_1= f'/home/transposenet/data/anyloc_cambridge/Cambridge_{self.scene}/descriptors/allVectors_data_compressed_test_rgb.npz'
 
            else:
                raise ValueError("Invalid results_type! ")   


            data = np.load(new_filename_vlad_1)['ret_all']
            print('data.shape = ',data.shape)
            print('new_filename_vlad_1 = ',new_filename_vlad_1)
           

        else:          
            new_filename_vlad_1= f'/home/transposenet/data/anyloc_cambridge/Cambridge_{self.scene}/descriptors/allVectors_data_compressed_train_rgb.npz'

            data = np.load(new_filename_vlad_1)['ret_all']
            print('data.shape = ',data.shape)
            print('new_filename_vlad_1 = ',new_filename_vlad_1)
        

        # 获取保存的数组和文件名
        self.allVectors_vlad_loaded = data #allVectors_vlad.shape =  (20, 1, 49152)
        print('Loaded allVectors_vlad.shape = ', self.allVectors_vlad_loaded.shape) #Loaded allVectors_vlad.shape =  (1000, 1, 49152)
        # pdb.set_trace()
        #------- --------取数组  allVectors_vlad.shape =  (20, 1, 49152)----------




        if len(self.img_paths) != self.allVectors_vlad_loaded.shape[0]:
            raise Exception('RGB file count does not match self.allVectors_vlad_loaded file count!')
  

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_rgb = imread(self.img_paths[idx]) #原本img。shape=  (480, 640, 3)
        # print('self.img_paths[idx] = ',self.img_paths[idx])
        # 复制图像到 img_show
        # print('原本img。shape= ', img.shape)#, 'img.device = ',img.device)
        # img = self.anyloc_dino_feature[idx] #加载出来后特征图 img。shape=  (256, 1536)
        img = img_rgb
        
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
    df = pd.read_csv(labels_file) #dataset_path =  /home/transposenet/data/Cambridges
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values] #'/home/transposenet/data/Cambridge/KingsCollege/seq8/frame00079.png'
    # print('imgs_paths = ',imgs_paths) #'/home/transposenet/data/Cambridge/StMarysChurch/seq13/frame00350.png
    # pdb.set_trace()
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