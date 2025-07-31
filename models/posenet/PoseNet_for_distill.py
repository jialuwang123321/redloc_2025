import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseNet(nn.Module):
    """
    A class to represent a classic pose regressor (PoseNet) with an efficient-net backbone
    PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization,
    Kendall et al., 2015
    """
    def __init__(self, backbone_path,return_feature=False):
        """
        Constructor
        :param backbone_path: backbone path to a resnet backbone
        """
        super(PoseNet, self).__init__()
        self.return_feature =  return_feature
        # Efficient net
        self.backbone = torch.load(backbone_path)
        backbone_dim = 1280
        latent_dim = 2048#1024

        # Regressor layers
        self.fc1 = nn.Linear(backbone_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 3)
        self.fc3 = nn.Linear(latent_dim, 4)

        self.dropout = nn.Dropout(p=0.1)
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, data):
        """
        Forward pass
        :param data: (torch.Tensor) dictionary with key-value 'img' -- input image (N X C X H X W)
        :return: (torch.Tensor) dictionary with key-value 'pose' -- 7-dimensional absolute pose for (N X 7)
        """
        x = self.backbone.extract_features(data.get('img'))
        # print('x=',x)

        # print('after extract, feature_map_posenet.shape = ',feature_map_posenet.shape) #torch.Size([8, 1280, 7, 7])
        x = self.avg_pooling_2d(x)
        # print('after avg_pooling_2d, x.shape = ',x.shape) #torch.Size([8, 1280, 1, 1])
        x = x.flatten(start_dim=1)
        # print('after flatten, x.shape = ',x.shape) #torch.Size([8, 1280])
        x = self.dropout(F.relu(self.fc1(x)))
        # print('after dropout, x.shape = ',x.shape) #torch.Size([8, 1024])
        if self.return_feature:
            feature_map_posenet = x.clone()  # 保存特征图的深拷贝
        p_x = self.fc2(x)
        # print('after fc2, x.shape = ',x.shape) #torch.Size([8, 1024])
        p_q = self.fc3(x)
        # print('after fc3, x.shape = ',x.shape) #torch.Size([8, 1024])
        if self.return_feature:
          return {'pose': torch.cat((p_x, p_q), dim=1),'feature_map_posenet':feature_map_posenet}
        else:
          return {'pose': torch.cat((p_x, p_q), dim=1)}
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import numpy as np

# import os
# os.environ['TORCH_MODEL_ZOO'] = os.path.join('..', 'data', 'models')

# import sys
# sys.path.insert(0, '../')

#def trace_hook(m, g_in, g_out):
#  for idx,g in enumerate(g_in):
#    g = g.cpu().data.numpy()
#    if np.isnan(g).any():
#      set_trace()
#  return None

def filter_hook(m, g_in, g_out):
  g_filtered = []
  for g in g_in:
    g = g.clone()
    g[g != g] = 0
    g_filtered.append(g)
  return tuple(g_filtered)




def forward_hook(module, input, output):
    global activations
    activations = output
    activations = activations.detach().cpu().numpy().flatten()
    # print('grad_np=',grad_np)
    # print('activations.shape = ',activations.shape)
    activations = activations.flatten()
    # 将梯度保存为文本文件，追加模式
    with open('/home/transposenet/out/activations_posenet.txt', 'w') as f:
        np.savetxt(f, activations, fmt='%f')

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]
    gradients = gradients.cpu().numpy().flatten()
    # print('grad_np=',grad_np)
    # print('gradients.shape = ',gradients.shape)
    gradients =gradients.flatten()
    with open('/home/transposenet/out/gradient_posenet.txt', 'w') as f:
        np.savetxt(f, gradients, fmt='%f')

class PoseNet_origional(nn.Module):
  def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
      feat_dim=2048, filter_nans=False,return_feature=False):
    super(PoseNet_origional, self).__init__()
    self.droprate = droprate

    # replace the last FC layer in feature extractor
    self.feature_extractor = feature_extractor
    self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
    fe_out_planes = self.feature_extractor.fc.in_features
    self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

    self.return_feature = return_feature

    # final_conv_layer = self.feature_extractor.layer4[-1]

    # final_conv_layer.register_forward_hook(forward_hook)
    # final_conv_layer.register_full_backward_hook(backward_hook)

    self.fc_xyz  = nn.Linear(feat_dim, 3)
    self.fc_wpqr = nn.Linear(feat_dim, 4)
    if filter_nans:
      self.fc_wpqr.register_backward_hook(hook=filter_hook)

    # initialize
    if pretrained:
      init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
    else:
      init_modules = self.modules()

    for m in init_modules:
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)

  def forward(self, x):
    x = self.feature_extractor(x)
    x = F.relu(x)
    if self.droprate > 0:
      x = F.dropout(x, p=self.droprate)
    # print('after dropout, x.shape = ',x.shape) #torch.Size([8, 2048])
    if self.return_feature:
        feature_map_posenet = x.clone()  # 保存特征图的深拷贝

    xyz  = self.fc_xyz(x)
    wpqr = self.fc_wpqr(x)
        
    if self.return_feature:
      return {'pose': torch.cat((xyz, wpqr), 1),'feature_map_posenet':feature_map_posenet}
    else:
      return torch.cat((xyz, wpqr), 1)


from mamba_ssm import Mamba

class PoseNet_origional_mamba(nn.Module):
  def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
      feat_dim=2048, filter_nans=False,use_classical_mamba = False):
    super(PoseNet_origional_mamba, self).__init__()
    self.droprate = droprate

    # replace the last FC layer in feature extractor
    self.feature_extractor = feature_extractor
    self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
    fe_out_planes = self.feature_extractor.fc.in_features
    self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
    self.use_classical_mamba = use_classical_mamba
    ##### mamba #########
    self.mamba1_model_ori_B1D = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=2048, # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
        )

    self.fc_xyz  = nn.Linear(feat_dim, 3)
    self.fc_wpqr = nn.Linear(feat_dim, 4)
    if filter_nans:
      self.fc_wpqr.register_backward_hook(hook=filter_hook)

    # initialize
    if pretrained:
      init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
    else:
      init_modules = self.modules()

    for m in init_modules:
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)

  def forward(self, x):
    x = self.feature_extractor(x) # torch.Size([32, 2048])
    # print('after feature_extractor x.shape = ',x.shape)

    
    #F1
    x = F.relu(x)
    #F2
    # x = x.mean(dim=1)  # 全局平均池化获取特征表示
    # print('after x.mea, x.shape = ',x.shape) #([B, 1024])

    if self.use_classical_mamba:
      # print('before unsqueeze, x shape =', x.shape)
      x = x.unsqueeze(1)
      # print('before use_mamba1_B1D, x shape =', x.shape) #([B,1, 1024])
      x = self.mamba1_model_ori_B1D(x)
      # print('after use_mamba1_B1D, x shape =', x.shape) #([B,1, 1024])
      x = x.squeeze(1)
      # print('after squeeze x shape =', x.shape) 
    else:
      # print('before unsqueeze, x shape =', x.shape)
      x = x.unsqueeze(1)
      # 将4的顺序反过来
      # print('before flip, x shape =', x.shape)
      x1 = x.flip(dims=[2])
      # 在第一个维度拼接 x1 和 x
      # print('before cat, x shape =', x.shape)
      x = torch.cat((x1, x), dim=1)

      # print('before use_mamba1_B1D, x shape =', x.shape) #([B,1, 1024])
      x = self.mamba1_model_ori_B1D(x)
      # print('after use_mamba1_B1D, x shape =', x.shape) #([B,1, 1024])
      x = x[:,1,:]
      # print('after x[:,1,:], x shape =', x.shape) 

    if self.droprate > 0:
      x = F.dropout(x, p=self.droprate)

    xyz  = self.fc_xyz(x)
    wpqr = self.fc_wpqr(x)
    return torch.cat((xyz, wpqr), 1)