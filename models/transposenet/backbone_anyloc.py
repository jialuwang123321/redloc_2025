"""
Code for the backbone of TransPoseNet
Backbone code is based on https://github.com/facebookresearch/detr/tree/master/models with the following modifications:
- use efficient-net as backbone and extract different activation maps from different reduction maps
- change learned encoding to have a learned token for the pose
"""
import torch.nn.functional as F
from torch import nn
from .pencoder import build_position_encoding, NestedTensor
from typing import Dict, List
import torch
import ipdb as pdb


import matplotlib.pyplot as plt
import os
from PIL import Image


class DownsampleNetwork(nn.Module):
    def __init__(self):
        super(DownsampleNetwork, self).__init__()
        self.downsample = nn.Sequential(
            # 第一次下采样
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # (Batchsize, 64, 224, 224)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (Batchsize, 64, 112, 112)

            # 第二次下采样
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (Batchsize, 128, 112, 112)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (Batchsize, 128, 56, 56)

            # 第三次下采样
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (Batchsize, 256, 56, 56)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (Batchsize, 256, 28, 28)

            # 第四次下采样
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # (Batchsize, 512, 28, 28)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (Batchsize, 512, 14, 14)

            # 最终卷积层将通道数增加到1536
            nn.Conv2d(512, 1536, kernel_size=3, stride=1, padding=1),  # (Batchsize, 1536, 14, 14)
            nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False)  # (Batchsize, 1536, 16, 16)
        )

    def forward(self, x):
        return self.downsample(x)


class UpsampleNetwork(nn.Module):
    def __init__(self):
        super(UpsampleNetwork, self).__init__()
        self.upsample = nn.Sequential(
            # 第一次上采样
            nn.ConvTranspose2d(1536, 512, kernel_size=4, stride=2, padding=1), # (Batchsize, 512, 32, 32)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 第二次上采样
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # (Batchsize, 256, 64, 64)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 第三次上采样
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # (Batchsize, 128, 128, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 第四次上采样
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (Batchsize, 64, 256, 256)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 最终卷积层将通道数减少到3，并且调整大小到224x224
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # (Batchsize, 3, 256, 256)
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)  # (Batchsize, 3, 224, 224)
        )

    def forward(self, x):
        return self.upsample(x)



class BackboneBase(nn.Module):

    def __init__(self,backbone_down: nn.Module,backbone_up: nn.Module, backbone: nn.Module, reduction):
        super().__init__()
        self.body_down = backbone_down
        self.body_up = backbone_up
        self.body = backbone
        self.reductions = reduction
        self.reduction_map = {"reduction_3": 40, "reduction_4": 112, "reduction_6":1280}
        self.num_channels = [self.reduction_map[reduction] for reduction in self.reductions]

        # 初始化归一化层，假设 xs 是具有 3 个通道的图像张量
        self.norm_layer = nn.BatchNorm2d(num_features=3)

    def forward(self, tensor_list: NestedTensor, rgb=None):
        # print('rgb.shape = ',rgb.shape) #train [8,3,224,224], test [1,3,224,224]
        # pdb.set_trace()
        # print('tensor_list.tensors.shape = ',tensor_list.tensors.shape) #tensor_list.tensors.shape =  torch.Size([8, 1536, 16, 16])
        # pdb.set_trace()
        input_tensor = tensor_list.tensors
        
        # 计算 rgb 的大小 (以 MB 为单位)
        # rgb_size_in_bytes = rgb.numel() * rgb.element_size()
        # rgb_size_in_mb = rgb_size_in_bytes / (1024 * 1024)
        # print(f'rgb size: {rgb_size_in_mb:.2f} MB')
        # pdb.set_trace()
        #---------------- 下加上采样  -----------------------------
        xs_down = self.body_down(rgb)
        # print('xs_down.shape = ',xs_down.shape) # torch.Size([8, 3, 224, 224])
        # pdb.set_trace()
        xs_up = self.body_up(xs_down)
        # print('xs_up.shape = ',xs_up.shape) # torch.Size([8, 3, 224, 224])
        # pdb.set_trace()
        xs_plot = rgb- xs_up
        xs = self.norm_layer(xs_plot)
        '''
        二维的批归一化层（Batch Normalization Layer）。
        目的：对输入的张量进行归一化操作，使得不同 mini-batch 之间的数据分布保持一致
        作用：，减少梯度消失或爆炸问题，加速模型的收敛速度，并增强模型的泛化能力
        '''
        # print('xs_plot.shape = ',xs_plot.shape) # torch.Size([8, 3, 224, 224])
        # pdb.set_trace()
        #---------------- 下加上采样  -----------------------------

        # #---------------- anyloc 加上采样  -----------------------------
        # xs = self.body_up(input_tensor)  # after body up, xs.shape =  torch.Size([8, 3, 224, 224])
        # # print('after body up, xs.shape = ',xs.shape) 
        # # print('input_tensor.shape = ',input_tensor.shape)

        # xs = rgb- xs
        # # 使用归一化层对 xs 进行归一化
        # xs = self.norm_layer(xs)
        # #---------------- anyloc 加上采样  -----------------------------s

        # print('after xs-rgb, xs.shape = ',xs.shape) 
        # pdb.set_trace()

        # #------------  可视化---------------------------------------
        # #定义保存路径
        # save_dir = "/home/transposenet/vis_aug"
        # os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建

        # # 可视化并保存前 3 个样本
        # for i in range(3):
        #     xs_plot = rgb.clone().detach()#.cpu().numpy()
        #     # 将张量从 [3, 224, 224] 转换为 [224, 224, 3]，并且从 Tensor 转换为 NumPy 数组
        #     img = xs_plot[i].permute(1, 2, 0).cpu().numpy()

        #     # 将张量值归一化到 [0, 1] 范围并转换为 uint8
        #     img = (img - img.min()) / (img.max() - img.min())  # 归一化到 [0, 1]
        #     img = (img * 255).astype("uint8")

        #     # 将 NumPy 数组转换为 PIL 图像
        #     img_pil = Image.fromarray(img)
        #     # 保存图像到指定目录
        #     img_pil.save(os.path.join(save_dir, f"image_{i}.png"))

        # print(f"前 3 个样本已保存到 {save_dir} 目录中。")
        # #------------  可视化-------------------------------

        xs = self.body.extract_endpoints(xs) 
        # print('xs=',xs)


        # xs = self.body.extract_endpoints(tensor_list.tensors) 
        out: Dict[str, NestedTensor] = {}
        for name in self.reductions:
            # #------------方法一：不用上采样不用effcientnet----------------
            # x = tensor_list.tensors
            # # print('x.shape = ',x.shape) #x.shape =  torch.Size([8, 1536, 16, 16])
            # # print('x.device = ',x.device)
            # # pdb.set_trace()
            # m = tensor_list.mask
    
            # # is_all_false = torch.all(m == False) # 判断 m 是否全为 False
            # # print("Is m all False?:", is_all_false.item())

            # assert m is not None
            # mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            # # is_all_false = torch.all(mask == False)
            # # print("after interpolate Is mask all False?:", is_all_false.item())
            # # pdb.set_trace()
            # out[name] = NestedTensor(x, mask)
            # #------------方法一：不用上采样不用effcientnet-------------

            #-------------方法二：用上采样+effcientnet---------------
            x = xs[name]
            # print('after extract_endpoints, x = ',x)
            # print('after extract_endpoints, x.shape = ',x.shape)
            # pdb.set_trace()
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
            #-------------方法二：用上采样+effcientnet------------


        return out


class Backbone(BackboneBase):
    def __init__(self, backbone_model_path: str, reduction):
        backbone_down = DownsampleNetwork()
        backbone_up = UpsampleNetwork() #torch.load(backbone_model_path)
        backbone = torch.load(backbone_model_path)
        super().__init__(backbone_down,backbone_up,backbone, reduction)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor, rgb=None):
        xs = self[0](tensor_list,rgb)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items(): # allVectors_vlad.shape =  (20, 1, 49152)
            # print('name = ',name)#name =  reduction_4
            # print('x.tensors.shape=',x.tensors.shape, 'x.tensors.device = ',x.tensors.device)
            
            # x.tensors.shape= torch.Size([8, 112, 14, 14]) x.tensors.device =  cuda:0, name =  reduction_4
            # x.tensors.shape= torch.Size([8, 40, 28, 28]) x.tensors.device =  cuda:0 ,name =  reduction_3
            # x.tensors.shape= torch.Size([8, 1280, 7, 7]) x.tensors.device =  cuda:0, name =  reduction_6
            # pdb.set_trace()
            out.append(x)
            # position encoding
            ret = self[1](x)
            # print('ret=',ret)
            # pdb.set_trace()
            if isinstance(ret, tuple):
                p_emb, m_emb = ret
                # print('p_emb.to(x.tensors.dtype).shape = ',p_emb.to(x.tensors.dtype).shape)#p_emb.to(x.tensors.dtype).shape =  torch.Size([8, 256])
                # print('m_emb.to(x.tensors.dtype).shape = ',m_emb.to(x.tensors.dtype).shape)
                #m_emb.to(x.tensors.dtype).shape =  torch.Size([8, 256, 14, 14]), name =  reduction_4
                #m_emb.to(x.tensors.dtype).shape =  torch.Size([8, 256, 28, 28]), name =  reduction_3
                #m_emb.to(x.tensors.dtype).shape =  torch.Size([8, 256, 7, 7]), name =  reduction_6
                # pdb.set_trace()
                pos.append([p_emb.to(x.tensors.dtype), m_emb.to(x.tensors.dtype)])
            else:
                pos.append(ret.to(x.tensors.dtype))

        return out, pos

def build_backbone(config):
    position_embedding = build_position_encoding(config)
    backbone = Backbone(config.get("backbone"), config.get("reduction")) # efficientnet, [reduction_4", "reduction_3"]
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
