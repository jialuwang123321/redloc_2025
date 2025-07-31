import os
import torch
# from torch import nn
import scipy.linalg as slin
import math
import transforms3d.quaternions as txq
import transforms3d.euler as txe
import numpy as np
import sys

from torch.nn import Module
from torch.autograd import Variable
from torch.nn.functional import pad
from torchvision.datasets.folder import default_loader
from collections import OrderedDict

import torch.nn.functional as F
import torch.nn as nn


class CameraPoseLoss_SoftKL(nn.Module):
    """
    A class to represent camera pose loss
    """

    def __init__(self, config, T=10.0, learnable=True):
        """
        :param config: (dict) configuration to determine behavior
        """
        super(CameraPoseLoss_SoftKL, self).__init__()
        self.learnable = learnable
        self.s_x = torch.nn.Parameter(torch.Tensor([config.get("s_x")]), requires_grad=self.learnable)
        self.s_q = torch.nn.Parameter(torch.Tensor([config.get("s_q")]), requires_grad=self.learnable)
        self.norm = config.get("norm")
        self.T = T
        # 定义 KL 散度损失函数
        self.softKL_criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, targ):
            """
            Forward pass
            :param est_pose: (torch.Tensor) batch of estimated poses, a Nx7 tensor
            :param gt_pose: (torch.Tensor) batch of ground_truth poses, a Nx7 tensor
            :return: camera pose loss
            """
            # 分别获取平移和旋转部分
            # print("pred.shape = ",pred.shape)
            # print("targ.shape = ",targ.shape)
            translation_student = pred[:, :3]
            translation_teacher = targ[:, :3]
            # print('translation_student.shape = ',translation_student.shape, 'translation_teacher.shape = ',translation_teacher.shape)

            rotation_student = pred[:, 3:]
            rotation_teacher = targ[:, 3:]
            # print('rotation_student.shape = ',rotation_student.shape, 'rotation_teacher.shape = ',rotation_teacher.shape)

            # 将平移部分转换为概率分布
            prob_translation_student = F.log_softmax(translation_student / self.T, dim=-1)
            prob_translation_teacher = F.softmax(translation_teacher / self.T, dim=-1)
            # print('prob_translation_student.shape = ',prob_translation_student.shape, 'prob_translation_teacher.shape = ',prob_translation_teacher.shape)

            # 将旋转部分转换为概率分布
            prob_rotation_student = F.log_softmax(rotation_student / self.T, dim=-1)
            prob_rotation_teacher = F.softmax(rotation_teacher / self.T, dim=-1)
            # print('prob_rotation_student.shape = ',prob_rotation_student.shape, 'prob_rotation_teacher.shape = ',prob_rotation_teacher.shape)

            # 计算平移部分的 KL 散度损失
            l_x = self.softKL_criterion(prob_translation_student, prob_translation_teacher) * (self.T ** 2)
            # print('计算平移部分的 KL 散度损失 l_x = ',l_x)
            # 计算旋转部分的 KL 散度损失
            l_q = self.softKL_criterion(prob_rotation_student, prob_rotation_teacher) * (self.T ** 2)
            # print('计算旋转部分的 KL 散度损失 l_x = ',l_q)

            if self.learnable:
                # print('平移torch.exp(-self.s_x) = ',torch.exp(-self.s_x), 'self.s_x = ',self.s_x)
                # print('旋转torch.exp(-self.s_q) = ',torch.exp(-self.s_q), 'self.s_q = ',self.s_q)
                loss = l_x * torch.exp(-self.s_x) + self.s_x + l_q * torch.exp(-self.s_q) + self.s_q
            else:
                # print('平移self.s_x = ',self.s_x)
                # print('旋转self.s_q = ',self.s_q)
                loss =  self.s_x*l_x + self.s_q*l_q
            # print('KL 散度损失 = ',l_q)
            return loss


class FeatureLoss(nn.Module):
    def __init__(self, t_loss_fn=nn.MSELoss(),img_in=False, per_channel=False,simple_L2=False,cos_vs1=False): #, sax=0.0, learn_beta=False):
        super(FeatureLoss, self).__init__()
        self.t_loss_fn = t_loss_fn
        # self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        def normalize_feature_map(feature_map):
            mean = feature_map.mean(dim=[2], keepdim=True)
            std = feature_map.std(dim=[2], keepdim=True)
            return (feature_map - mean) / (std + 1e-5)
        self.normalize_feature_map = normalize_feature_map
        
        # 定义余弦相似度损失函数
        self.criterion_cosine = nn.CosineEmbeddingLoss()
        self.img_in = img_in
        self.per_channel = per_channel
        self.simple_L2 = simple_L2
        self.cos_vs1 = cos_vs1

    def forward(self, pred, targ):
        ''' Compute Feature MSE Loss 
        :param: feature_rgb, [C,H,W] or [C, N_rand]
        :param: feature_target, [C,H,W] or [C, N_rand]
        :param: img_in, True: input is feature maps, False: input is rays
        :param: random, True: randomly using per pixel or per channel cossimilarity loss
        '''
        # print('before 截取, pred.shape = ', pred.shape)
        # print('targ.shape = ', targ.shape)
        if (len(pred.shape) < 4 ):
            feature_rgb = pred #[ B, 730, 384])
        else:
            feature_rgb = pred[-1, :, :, :] #[12, B, 730, 384])-->[ B, 730, 384])
        if (len(targ.shape) < 4):
            feature_target = targ #[ B, 730, 384])
        else:
            feature_target = targ[-1, :, :, :] #[12, B, 730, 384])-->[ B, 730, 384])
        # feature_target = targ[-1, :, :, :]
        if feature_target.shape[1]!= feature_rgb.shape[1]: #([32, 197, 384]) v.s. #([32, 196, 384])
            print("没有取出token的 feature_target shape:", feature_target.shape) #([32, 197, 384])
            M= feature_target.shape[1]  #M= 197 
            token_position =  int((M-1)/2) #token_position =  98
            # print('M = ',M, 'token_position =',token_position)
            mask = torch.ones(M, dtype=torch.bool) 
            mask[token_position] = False 
            feature_target = feature_target[:, mask, :]
        #     print("取出token的 feature_target shape:", feature_target.shape) #([32, 196, 384])
        # print('after 截取, feature_rgb.shape = ', feature_rgb.shape)
        # print('feature_target.shape = ', feature_target.shape)

        if self.simple_L2:
            # 对学生和老师的特征图进行归一化
            feature_rgb = self.normalize_feature_map(feature_rgb)
            feature_target = self.normalize_feature_map(feature_target)
            print('simple_L2 = ',self.simple_L2)
            loss = self.t_loss_fn(feature_target,feature_rgb)
            return loss
        elif self.cos_vs1:
            print('cos_vs1')
            # 计算余弦相似度损失 ,需要将特征图展平  
            C,H,W = feature_rgb.size()
            fr = feature_rgb.reshape(C, H*W)
            ft = feature_target.reshape(C, H*W)

            # 目标标签，1 表示相似
            cos_vs1_target = torch.ones(C).to(feature_rgb.device)
            # print('fr.shape = ',fr.shape, 'ft.shape = ',ft.shape, 'cos_vs1_target.shape = ',cos_vs1_target.shape)

            loss = self.criterion_cosine(fr, ft, cos_vs1_target)
            return loss

        if self.img_in:
            C,H,W = feature_rgb.size()
            # print('feature_rgb.size()=C,H,W = ', feature_rgb.size())
            fr = feature_rgb.reshape(C, H*W)
            ft = feature_target.reshape(C, H*W)
        else:
            fr = feature_rgb
            ft = feature_target

        # cosine loss
        if self.per_channel:
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        else:
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        loss = 1 - cos(fr, ft).mean()
        # print('FeatureLoss')
        # print('loss = ',loss)
        # print('self.sax = ',self.sax,' torch.exp(-self.sax) = ', torch.exp(-self.sax) )

        return loss