import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
# from network.att import AttentionBlock
from mamba_ssm import Mamba

class AtLoc_dinov2_mamba(nn.Module):
    def __init__(self, feature_extractor, droprate=0.5, pretrained=True, feat_dim=2048, lstm=False, return_B1D_feature=False, return_dinov2_feature=False ):
        super(AtLoc_dinov2_mamba, self).__init__()
        self.droprate = droprate
        self.lstm = lstm
        self.dinov2=True

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        self.use_mamba1_BLD = False
        self.use_mamba1_B1D = True
        self.use_mamba1_1BD = False

        self.return_B1D_feature =return_B1D_feature
        self.return_dinov2_feature = return_dinov2_feature

        print('\n####################################')
        print('self.use_mamba1_BLD = ',self.use_mamba1_BLD, 'self.use_mamba1_B1D = ', self.use_mamba1_B1D, 'self.use_mamba1_1BD = ', self.use_mamba1_1BD)
        print('self.return_B1D_feature = ',self.return_B1D_feature, 'self.return_dinov2_feature = ', self.return_dinov2_feature)
        print('####################################\n')
        if self.dinov2: 

            # dinov2 特征提取层 定义
            v2=0 # 加载预训练的 DINOv2 模型  0:Large;  1:Base;    2:small
            if v2==0: 
                feat_dim_dino = 1024
                self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to('cuda')
            elif v2==1:
                feat_dim_dino = 384#s:384
                self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to('cuda')
            elif v2==2:
                feat_dim_dino = 384#s:384
                self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to('cuda')
            
            # 6-DoF输出层 定义
            self.fc_xyz = nn.Linear(feat_dim_dino, 3)
            self.fc_wpqr = nn.Linear(feat_dim_dino, 4)


        if self.use_mamba1_BLD:
             # batch, length, dim = 2, 64, 16 # x = torch.randn(batch, length, dim).to("cuda")
            self.mamba1_model_ori_BLD = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=feat_dim_dino, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )

        if self.use_mamba1_B1D:
            self.mamba1_model_ori_B1D = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=feat_dim_dino, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )
            # self.conv1d = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)

            # self.max_pool = nn.MaxPool1d(kernel_size=2,stride=1)
            # self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=1)
        if self.use_mamba1_1BD:
            self.mamba1_model_ori_1BD = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=feat_dim_dino, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )


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
        if self.dinov2: 
            # print('before self.dinov2, x shape =', x.shape) #([B=64, 3, 224, 224])
            features_dict = self.dinov2.forward_features(x)
            x = features_dict['x_norm_patchtokens']
            # print('after x_norm_patchtokens, x.shape = ',x.shape) #([B, 256, 1024])

            if self.use_mamba1_BLD:
                #BLD
                # print('before use_mamba1_BLD, x shape =', x.shape) #([B,256, 1024])
                x = self.mamba1_model_ori_BLD(x)
                # print('after use_mamba1_BLD, x shape =', x.shape) #([B,256, 1024])

            x = x.mean(dim=1)  # 全局平均池化获取特征表示
            # print('after x.mea, x.shape = ',x.shape) #([B, 1024])



        if self.use_mamba1_B1D:
            # #B1D
            # x = x.unsqueeze(1)
            # # print('before repeat, x shape =', x.shape)
            # x = x.repeat(1, 2, 1)

            x = x.unsqueeze(1)
            # 将4的顺序反过来
            x1 = x.flip(dims=[2])
            # 在第一个维度拼接 x1 和 x
            x = torch.cat((x1, x), dim=1)

            # print('before use_mamba1_B1D, x shape =', x.shape) #([B,1, 1024])
            x = self.mamba1_model_ori_B1D(x)
            # print('after use_mamba1_B1D, x shape =', x.shape) #([B,1, 1024])
            # print('x = ',x)
            x = x[:,1,:]

            # x = x.squeeze(1)
            if self.return_B1D_feature:
                feature_map_B1D = x.clone()  # 保存特征图的深拷贝

        if self.use_mamba1_1BD:
            #1BD
            x = x.unsqueeze(0)
            # print('before use_mamba1_1BD, x shape =', x.shape) #([1,B, 1024])
            x = self.mamba1_model_ori_1BD(x)
            # print('after use_mamba1_1BD, x shape =', x.shape) #([1,B, 1024])
            x = x.squeeze(0)



        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)
        # print('before fc_xyz, x shape =', x.shape)
        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        # print('xyz shape =', xyz.shape)
        if self.return_B1D_feature:
            return torch.cat((xyz, wpqr), 1),feature_map_B1D
        return torch.cat((xyz, wpqr), 1)


