import os
import torch
from torch import nn
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

class AtLocCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, learn_beta=False):
        super(AtLocCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, pred, targ):
        loss = -self.sax* self.t_loss_fn(pred[:, :3], targ[:, :3]) + \
               torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:])
        
        # loss = torch.exp(-self.sax) * self.t_loss_fn(pred[:, :3], targ[:, :3]) + self.sax + \
        #        torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:]) + self.saq
       
       
        # print('in AtLocCriterion: self.sax = ', self.sax.item(), 'self.saq= ',self.saq.item())
        # print('loss_gard = ',loss.item(),'=', torch.exp(-self.sax).item(),'*',self.t_loss_fn(pred[:, :3], targ[:, :3]).item(),'+',self.sax.item(), "*", torch.exp(-self.sax).item(),'*',self.q_loss_fn(pred[:, 3:], targ[:, 3:]).item(),"+", self.saq.item())
        # print('\n\n in AtLocCriterion\npred[:, :3].shape = ',pred[:, :3].shape, 'targ[:, :3].shape = ',targ[:, :3].shape)
        # print('pred[:, 3:].shape = ',pred[:, 3:].shape, 'targ[:, 3:].shape = ',targ[:, 3:].shape, '\n\n')
        return loss

class AtLocCriterion_all(nn.Module):
    def __init__(self,  sa=1.0, sb=1.0, sc=1.0):
        super(AtLocCriterion_all, self).__init__()
        self.sa = nn.Parameter(torch.Tensor([sa]), requires_grad=True)
        self.sb = nn.Parameter(torch.Tensor([sb]), requires_grad=True)
        self.sc = nn.Parameter(torch.Tensor([sc]), requires_grad=True)

    def forward(self, L1, L2, L3):
        loss = self.sa* L1 + self.sb* L2 +self.sc* L3 
        # print(' self.sa = ', self.sa.item(), ' self.sb = ', self.sb.item(), ' self.sc =',  self.sc.item())
        # print('\n\n in AtLocCriterion\npred[:, :3].shape = ',pred[:, :3].shape, 'targ[:, :3].shape = ',targ[:, :3].shape)
        # print('pred[:, 3:].shape = ',pred[:, 3:].shape, 'targ[:, 3:].shape = ',targ[:, 3:].shape, '\n\n')
        return loss

# class AtLocCriterion_feature(nn.Module):
#     def __init__(self, t_loss_fn=nn.MSELoss(), sax=0.0, learn_beta=False):
#         super(AtLocCriterion_feature, self).__init__()
#         self.t_loss_fn = t_loss_fn
#         self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)

#     def forward(self, pred, targ, img_in=True, per_channel=False):
#         ''' Compute Feature MSE Loss 
#         :param: feature_rgb, [C,H,W] or [C, N_rand]
#         :param: feature_target, [C,H,W] or [C, N_rand]
#         :param: img_in, True: input is feature maps, False: input is rays
#         :param: random, True: randomly using per pixel or per channel cossimilarity loss
#         '''
      
#         # print('before transpose, pred.shape = ',pred.shape)
#         feature_rgb = torch.transpose(pred, dim0=1, dim1=0)
#         feature_target = torch.transpose(targ, dim0=1, dim1=0)
#         # print('after transpose, feature_rgb.shape = ',feature_rgb.shape)
#         batch_size = feature_rgb.shape[0]
#         # print('batch_size =', batch_size)
#         loss_all = 0

#         for i in range(0,batch_size):
#             feature_rgb_i = feature_rgb[i, :, :, :]
#             feature_target_i = feature_target[i, :, :, :]
#             # print('i = ', i, 'feature_rgb_i.shape = ',feature_rgb_i.shape)

#             fi_use = [0,6,-1] #2,4,6,8,10,12,14,16,18,20,22,24
#             feature_rgb_use = []
#             feature_target_use = []
#             for fi in range(0,12,1):#fi_use:

#                 feature_rgb_use.append(feature_rgb_i[fi, :, :])
#                 feature_target_use.append(feature_target_i[fi, :, :])
#                 # print('len(feature_rgb_use) = ',len(feature_rgb_use))

#             feature_rgb_i_ = torch.stack(feature_rgb_use)
#             feature_target_i_ = torch.stack(feature_target_use)
#             # print('feature_rgb_i_.shape = ',feature_rgb_i_.shape)

#             if img_in:
#                 C,H,W = feature_rgb_i_.size()
#                 fr = feature_rgb_i_.reshape(C, H*W)
#                 ft = feature_target_i_.reshape(C, H*W)
#             else:
#                 fr = feature_rgb_i_
#                 ft = feature_target_i_

#             # cosine loss
#             if per_channel:
#                 cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
#             else:
#                 cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#             loss_i = 1 - cos(fr, ft).mean()
#             # print('at batch ', i, 'loss_all = ',loss_all, 'loss_i = ', loss_i)
#             loss_all = loss_all + loss_i
#             # print( 'loss_all = ',loss_all)
            
#         # print('before mean over batch , final loss_all = ',loss_all)
#         loss = loss_all/batch_size
#         # print('after mean over batch ,final loss = ',loss)
#             # print('AtLocCriterion_feature')
#             # print('loss = ',loss)
#             # print('self.sax = ',self.sax,' torch.exp(-self.sax) = ', torch.exp(-self.sax) )

#         return loss

# class AtLocCriterion_feature(nn.Module):
#     def __init__(self, t_loss_fn=nn.MSELoss(), sax=0.0, learn_beta=False):
#         super(AtLocCriterion_feature, self).__init__()
#         self.t_loss_fn = t_loss_fn
#         self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)

#     def forward(self, pred, targ, img_in=True, per_channel=False):
#         ''' Compute Feature MSE Loss 
#         :param: feature_rgb, [C,H,W] or [C, N_rand]
#         :param: feature_target, [C,H,W] or [C, N_rand]
#         :param: img_in, True: input is feature maps, False: input is rays
#         :param: random, True: randomly using per pixel or per channel cossimilarity loss
#         '''
        
#         feature_rgb = pred[ :,0, :, :] #[12, B, 730, 384])-->[ B, 730, 384])
#         feature_target = targ[ :,0, :, :]
#         # print('feature_rgb.shape = ',feature_rgb.shape)

#         if img_in:
#             C,H,W = feature_rgb.size()
#             fr = feature_rgb.reshape(C, H*W)
#             ft = feature_target.reshape(C, H*W)
#         else:
#             fr = feature_rgb
#             ft = feature_target

#         # cosine loss
#         if per_channel:
#             cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
#         else:
#             cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#         loss = 1 - cos(fr, ft).mean()
#         # print('AtLocCriterion_feature')
#         # print('loss = ',loss)
#         # print('self.sax = ',self.sax,' torch.exp(-self.sax) = ', torch.exp(-self.sax) )

#         return loss

class AtLocCriterion_feature(nn.Module):
    def __init__(self, t_loss_fn=nn.MSELoss()): #, sax=0.0, learn_beta=False):
        super(AtLocCriterion_feature, self).__init__()
        self.t_loss_fn = t_loss_fn
        # self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        def normalize_feature_map(feature_map):
            mean = feature_map.mean(dim=[2], keepdim=True)
            std = feature_map.std(dim=[2], keepdim=True)
            return (feature_map - mean) / (std + 1e-5)
        self.normalize_feature_map = normalize_feature_map

        # 定义余弦相似度损失函数
        self.criterion_cosine = nn.CosineEmbeddingLoss()
    def forward(self, pred, targ, img_in=True, per_channel=False,simple_L2=False,cos_vs1=False):
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
            # print("没有取出token的 feature_target shape:", feature_target.shape) #([32, 197, 384])
            M= feature_target.shape[1]  #M= 197 
            token_position =  int((M-1)/2) #token_position =  98
            # print('M = ',M, 'token_position =',token_position)
            mask = torch.ones(M, dtype=torch.bool) 
            mask[token_position] = False 
            feature_target = feature_target[:, mask, :]
        #     print("取出token的 feature_target shape:", feature_target.shape) #([32, 196, 384])
        # print('after 截取, feature_rgb.shape = ', feature_rgb.shape)
        # print('feature_target.shape = ', feature_target.shape)

        if simple_L2:
            # 对学生和老师的特征图进行归一化
            feature_rgb = self.normalize_feature_map(feature_rgb)
            feature_target = self.normalize_feature_map(feature_target)
            # print('simple_L2 = ',simple_L2)
            loss = self.t_loss_fn(feature_target,feature_rgb)
            return loss
        elif cos_vs1:
            # print('cos_vs1')
            # 计算余弦相似度损失 ,需要将特征图展平  
            C,H,W = feature_rgb.size()
            fr = feature_rgb.reshape(C, H*W)
            ft = feature_target.reshape(C, H*W)

            # 目标标签，1 表示相似
            cos_vs1_target = torch.ones(C).to(feature_rgb.device)
            # print('fr.shape = ',fr.shape, 'ft.shape = ',ft.shape, 'cos_vs1_target.shape = ',cos_vs1_target.shape)

            loss = self.criterion_cosine(fr, ft, cos_vs1_target)
            return loss

        if img_in:
            C,H,W = feature_rgb.size()
            # print('feature_rgb.size()=C,H,W = ', feature_rgb.size())
            fr = feature_rgb.reshape(C, H*W)
            ft = feature_target.reshape(C, H*W)
        else:
            fr = feature_rgb
            ft = feature_target

        # cosine loss
        if per_channel:
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        else:
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        loss = 1 - cos(fr, ft).mean()
        # print('AtLocCriterion_feature')
        # print('loss = ',loss)
        # print('self.sax = ',self.sax,' torch.exp(-self.sax) = ', torch.exp(-self.sax) )

        return loss


class AtLocCriterion_SoftKL(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, learn_beta=False, T=2.0):
        super(AtLocCriterion_SoftKL, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

        self.T = T

        # 定义 KL 散度损失函数
        self.softKL_criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, targ):

        # 设置温度参数 T
        T = 10.0

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
        prob_translation_student = F.log_softmax(translation_student / T, dim=-1)
        prob_translation_teacher = F.softmax(translation_teacher / T, dim=-1)
        # print('prob_translation_student.shape = ',prob_translation_student.shape, 'prob_translation_teacher.shape = ',prob_translation_teacher.shape)

        # 将旋转部分转换为概率分布
        prob_rotation_student = F.log_softmax(rotation_student / T, dim=-1)
        prob_rotation_teacher = F.softmax(rotation_teacher / T, dim=-1)
        # print('prob_rotation_student.shape = ',prob_rotation_student.shape, 'prob_rotation_teacher.shape = ',prob_rotation_teacher.shape)

        # 计算平移部分的 KL 散度损失
        translation_loss = self.softKL_criterion(prob_translation_student, prob_translation_teacher) * (T ** 2)

        # 计算旋转部分的 KL 散度损失
        rotation_loss = self.softKL_criterion(prob_rotation_student, prob_rotation_teacher) * (T ** 2)

        # 总损失为平移部分和旋转部分的加权和
        # loss = translation_loss + rotation_loss

        loss = -self.sax * translation_loss + \
               -self.saq * rotation_loss 

        # loss = torch.exp(-self.sax) * translation_loss + self.sax + \
        #        torch.exp(-self.saq) * rotation_loss + self.saq
        # print('in AtLocCriterion_SoftKL: self.sax = ', self.sax.item(), 'self.saq= ',self.saq.item())
        # print('torch.exp(-self.sax) = ', torch.exp(-self.sax).item(), 'torch.exp(-self.saq) = ',torch.exp(-self.saq).item())
       
        # print(f'Translation Soft Loss (with T={T}): {translation_loss.item()}')
        # print(f'Rotation Soft Loss (with T={T}): {rotation_loss.item()}')
        # print(f'Total Soft Loss (with T={T}): {loss.item()}')

        return loss

class AtLocCriterion_aux(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, learn_beta=False):
        super(AtLocCriterion_aux, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, pred, targ):
        loss_all = []
        for i in range(pred.shape[0]):
            loss = torch.exp(-self.sax) * self.t_loss_fn(pred[i, :, :3], targ[:, :3]) + self.sax + \
                torch.exp(-self.saq) * self.q_loss_fn(pred[i, :, 3:], targ[:, 3:]) + self.saq
            loss_all.append(loss)
      
        loss_all = torch.stack(loss_all) #torch.Size([3, 1])
        loss_all = torch.mean(loss_all)
        return loss_all
    
class AtLocPlusCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, srx=0.0, srq=0.0, learn_beta=False, learn_gamma=False):
        super(AtLocPlusCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)

    def forward(self, pred, targ):
        # absolute pose loss
        s = pred.size()
        abs_loss = torch.exp(-self.sax) * self.t_loss_fn(pred.view(-1, *s[2:])[:, :3], targ.view(-1, *s[2:])[:, :3]) + self.sax + \
                   torch.exp(-self.saq) * self.q_loss_fn(pred.view(-1, *s[2:])[:, 3:], targ.view(-1, *s[2:])[:, 3:]) + self.saq

        # get the VOs
        pred_vos = calc_vos_simple(pred)
        targ_vos = calc_vos_simple(targ)

        # VO loss
        s = pred_vos.size()
        vo_loss = torch.exp(-self.srx) * self.t_loss_fn(pred_vos.view(-1, *s[2:])[:, :3], targ_vos.view(-1, *s[2:])[:, :3]) + self.srx + \
                  torch.exp(-self.srq) * self.q_loss_fn(pred_vos.view(-1, *s[2:])[:, 3:], targ_vos.view(-1, *s[2:])[:, 3:]) + self.srq

        # total loss
        loss = abs_loss + vo_loss
        return loss

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def delink(self):
        self.log.close()

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except:
        print('Could not load image {:s}, unexpected error'.format(filename))
        return None
    return img

def qlog(q):
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q

def qexp(q):
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
    return q

def calc_vos_simple(poses):
    vos = []
    for p in poses:
        pvos = [p[i+1].unsqueeze(0) - p[i].unsqueeze(0) for i in range(len(p)-1)]
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)
    return vos

def quaternion_angular_error(q1, q2):
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta

def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

  # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out

def load_state_dict(model, state_dict):
    model_names = [n for n,_ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]

  # find prefix for the model and state dicts from the first param name
    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        model_prefix = model_names[0].split('.')[0]
        state_prefix = state_names[0].split('.')[0]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if state_prefix is None:
            k = model_prefix + k
        else:
            k = k.replace(state_prefix, model_prefix)
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)



