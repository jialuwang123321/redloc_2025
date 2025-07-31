import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from network.att import AttentionBlock

class AtLoc(nn.Module):
    def __init__(self, feature_extractor, droprate=0.5, pretrained=True, feat_dim=2048, lstm=False):
        super(AtLoc, self).__init__()
        self.droprate = droprate
        self.lstm = lstm
        self.dinov2=True

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

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
            self.fc_wpqr = nn.Linear(feat_dim_dino, 3)

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
            features_dict = self.dinov2.forward_features(x)
            x = features_dict['x_norm_patchtokens']
            # print('after x_norm_patchtokens, x.shape = ',x.shape) #([64, 256, 384])
            x = x.mean(dim=1)  # 全局平均池化获取特征表示
            # print('after x.mea, x.shape = ',x.shape) #([64, 384])

        else: 
            x = self.feature_extractor(x)
            x = F.relu(x)

            if self.lstm:
                x = self.lstm4dir(x)
            else:
                x = self.att(x.view(x.size(0), -1))

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)


