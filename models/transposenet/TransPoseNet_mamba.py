"""
The TransPoseNet_mamba model
"""

import torch
import torch.nn.functional as F
from torch import nn
from .transformer_encoder import Transformer
from .pencoder import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from mamba_ssm import Mamba
import numpy as np
import scipy.ndimage

import matplotlib.pyplot as plt
import numpy as np
import os
# from models.pose_loss_mamba import CameraPoseLoss_SoftKL, FeatureLoss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 定义钩子函数
def hook_fn(grad):
    # 将梯度转换为 NumPy 数组
    grad_np = grad.cpu().numpy()
    print('t grad_np.shape = ',grad_np.shape)
    grad_np = grad_np.flatten()
    print('grad_np.shape = ',grad_np.shape)
    # 将梯度保存为文本文件，追加模式
    with open('/home/transposenet/out/gradient_t.txt', 'w') as f:
        np.savetxt(f, grad_np, fmt='%f')
        # np.savetxt(f, grad_np.flatten().reshape(-1, grad_np.shape[-1]), fmt='%f')

def hook_fn_2(grad):
    grad_np = grad.cpu().numpy()
    print('rot grad_np.shape = ',grad_np.shape)
    grad_np = grad_np.flatten()
    print('grad_np.shape = ',grad_np.shape)
    with open('/home/transposenet/out/gradient_rot.txt', 'w') as f:
        np.savetxt(f, grad_np, fmt='%f')
        # np.savetxt(f, grad_np.flatten().reshape(-1, grad_np.shape[-1]), fmt='%f')



def save_pca_heatmap(features, output_path):
    print('before reshape, features.shape = ', features.shape)
    batch_size, patch_h, patch_w, feat_dim = features.shape

    # 将 tensor 转换为 numpy 数组
    tensor_np = features.cpu().numpy()

    # 创建一个新的数组来存放插值后的结果
    tensor_resized_np = np.zeros((batch_size, 256, 256, feat_dim))

    # 对每个 batch 进行插值
    for i in range(tensor_np.shape[0]):
        for j in range(tensor_np.shape[-1]):
            s = scipy.ndimage.zoom(tensor_np[i, :, :, j], (256 / patch_h, 256 / patch_w), order=3)  # 使用三次样条插值
            tensor_resized_np[i, :, :, j] = s

    print('tensor_resized_np.shape = ', tensor_resized_np.shape)

    # 将插值后的 numpy 数组转换回 PyTorch 张量
    features = torch.tensor(tensor_resized_np)
    
    # 进行 PCA 处理
    features = features.reshape(-1, feat_dim).cpu().numpy()

    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features)
    pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / (pca_features[:, 0].max() - pca_features[:, 0].min())

    pca_features_fg = pca_features[:, 0] > 0.3
    pca_features_bg = ~pca_features_fg

    b = np.where(pca_features_bg)

    pca.fit(features[pca_features_fg])
    pca_features_rem = pca.transform(features[pca_features_fg])
    for i in range(3):
        pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())

    pca_features_rgb = pca_features.copy()
    pca_features_rgb[pca_features_fg] = pca_features_rem
    pca_features_rgb[b] = 0

    print('before pca_features_rgb.reshape, pca_features_rgb.shape = ', pca_features_rgb.shape)
    pca_features_rgb = pca_features_rgb.reshape(batch_size, 256, 256, 3)
    print('after pca_features_rgb.reshape, pca_features_rgb.shape = ', pca_features_rgb.shape)

    image = pca_features_rgb[0]
    print('after pca_features_rgb[0] image.shape = ', image.shape)

    # # 创建自定义 colormap
    # colors = [(1, 1, 1), (0.9, 0.9, 0.9), (0.8, 0.8, 0.8), (0.7, 0.7, 0.7), 
    #           (0.6, 0.6, 0.6), (0.5, 0.5, 0.5), (0.4, 0.4, 0.4), (0.3, 0.3, 0.3), (0, 0, 0)]
    # n_bins = 100  # Discretizes the interpolation into bins
    # cmap_name = 'custom_cmap'
    # custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    plt.imshow(image)#, cmap=custom_cmap)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
    plt.close()

class TransPoseNet_mamba(nn.Module):

    def __init__(self, config, pretrained_path, return_feature=False, return_feature_plot=False,use_classical_mamba=False):
        """
        config: (dict) configuration of the model
        pretrained_path: (str) path to the pretrained backbone
        """
        super().__init__()

        config["backbone"] = pretrained_path
        config["learn_embedding_with_pose_token"] = True

        # CNN backbone
        self.backbone = build_backbone(config)

        # Position (t) and orientation (rot) encoders
        self.transformer_t = Transformer(config)
        self.transformer_rot = Transformer(config)

        self.use_global_mamba = False
        self.return_B1D_feature = return_feature
        self.return_feature_plot = return_feature_plot
        self.use_classical_mamba = use_classical_mamba
        

        ##### mamba #########
        self.mamba1_model_ori_B1D = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=256, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
            )
        
    
        if self.use_global_mamba:
            self.mamba1_model_196 = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=196, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
                )
            self.mamba1_model_784 = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=784, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
                )

        decoder_dim = self.transformer_t.d_model

        # The learned pose token for position (t) and orientation (rot)
        self.pose_token_embed_t = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)
        self.pose_token_embed_rot = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)

        self.num_scenes = config.get("num_scenes")
        self.multiscene = False
        
        self.classify_scene = config.get("classify_scene")

        if self.num_scenes is not None and self.num_scenes > 1:
            self.scene_embed = nn.Linear(1, decoder_dim)
            self.multiscene = True
            if self.classify_scene:
                self.avg_pooling = nn.AdaptiveAvgPool2d(1)
                self.scene_cls = nn.Sequential(nn.Linear(1280 , self.num_scenes),
                                               nn.LogSoftmax(dim=1))

        # The projection of the activation map before going into the Transformer's encoder
        self.input_proj_t = nn.Conv2d(self.backbone.num_channels[0], decoder_dim, kernel_size=1)
        self.input_proj_rot = nn.Conv2d(self.backbone.num_channels[1], decoder_dim, kernel_size=1)

        # Whether to use prior from the position for the orientation
        self.use_prior = config.get("use_prior_t_for_rot")

        # Regressors for position (t) and orientation (rot)
        self.regressor_head_t = PoseRegressor(decoder_dim, 3)
        self.regressor_head_rot = PoseRegressor(decoder_dim, 4, self.use_prior)

    def forward_transformers(self, data):
        """
        The forward pass expects a dictionary with key-value 'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
        return a dictionary with the following keys--values:
            global_desc_t: latent representation from the position encoder
            global_dec_rot: latent representation from the orientation encoder
        """
        samples = data.get('img')

        # Handle data structures
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # Extract the features and the position embedding from the visual backbone
        features, pos = self.backbone(samples)

        src_t, mask_t = features[0].decompose()
        src_rot, mask_rot = features[1].decompose()
        if self.return_feature_plot:
            src_t_plot = src_t.clone()
            src_rot_plot = src_rot.clone()

        ################  bi-mamba start t #####################
        
        
        
        if self.use_global_mamba:
            B, C, H, W = src_t.shape 
            # print('src_t.shape = ',src_t.shape) #([8, 40, 28, 28])
            src_t = src_t.flatten(2) 
            # print('after flatten, src_t.shape = ',src_t.shape) #([8, 112, 196])
            # 将4的顺序反过来
            x1t = src_t.flip(dims=[2])
            # print('after flip, src_t.shape = ',src_t.shape) #(([8, 112, 196])
            # 在第一个维度拼接 x1 和 src_t
            src_t = torch.cat((x1t, src_t), dim=1) 

            # print('before mamba1_model_ori_B1D, src_t shape =', src_t.shape)#[8, 224, 196])
            src_t = self.mamba1_model_196(src_t)
            # print('after ma./mba1_model_ori_B1D, src_t shape =', src_t.shape) #[8, 224, 196])
            src_t = src_t[:,C:,:]
            # print('after src_t[:,C,:], src_t.shape = ',src_t.shape)
            src_t = src_t.view(B, C, H, W)
            # print('after view, src_t.shape = ',src_t.shape)

            B, C, H, W = src_rot.shape 
            # print('src_rot.shape = ',src_rot.shape) #e([8, 40, 28, 28])
            src_rot = src_rot.flatten(2) 
            # print('after flatten, src_rot.shape = ',src_rot.shape) #([8, 40, 784])
            # 将4的顺序反过来
            x1t = src_rot.flip(dims=[2])
            # print('after flip, src_rot.shape = ',src_rot.shape) #(e([8, 40, 784])
            # 在第一个维度拼接 x1 和 src_rot
            src_rot = torch.cat((x1t, src_rot), dim=1) 

            # print('before mamba1_model_ori_B1D, src_rot shape =', src_rot.shape)#([8, 80, 784])
            src_rot = self.mamba1_model_784(src_rot)
            # print('after ma./mba1_model_o.i_B1D, src_rot shape =', src_rot.shape) #[8, 224, 196])
            src_rot = src_rot[:,C:,:]
            # print('after src_rot[:,C,:], src_rot.shape = ',src_rot.shape)
            src_rot = src_rot.view(B, C, H, W)
            # print('after view, src_rot.shape = ',src_rot.shape)
        ################  mamba end #####################
        

        # Run through the transformer to translate to "camera-pose" language
        assert mask_t is not None
        assert mask_rot is not None

        bs = src_t.shape[0]
        pose_token_embed_rot = self.pose_token_embed_rot.unsqueeze(1).repeat(1, bs, 1)
        pose_token_embed_t = self.pose_token_embed_t.unsqueeze(1).repeat(1, bs, 1)

        scene_dist = None
        if self.multiscene:
            selected_scene = data.get("scene")
            if self.classify_scene:
                src_scene, _ = features[2].decompose()
                src_scene = self.avg_pooling(src_scene).flatten(1)
                scene_dist = self.scene_cls(src_scene)
            if selected_scene is None: # test time
                assert(self.classify_scene)
                selected_scene = torch.argmax(scene_dist, dim=1).to(dtype=torch.float32)
            else:
                selected_scene = selected_scene.unsqueeze(1)

            scene_embed = self.scene_embed(selected_scene)
            pose_token_embed_rot = scene_embed + pose_token_embed_rot
            pose_token_embed_t = scene_embed + pose_token_embed_t


        local_descs_t = self.transformer_t(self.input_proj_t(src_t), mask_t, pos[0], pose_token_embed_t)
        local_descs_rot = self.transformer_rot(self.input_proj_rot(src_rot), mask_rot, pos[1],
                                               pose_token_embed_rot)
      
        # # 创建张量
        # local_descs_t.requires_grad_(True)
        # # 注册钩子
        # local_descs_t.register_hook(hook_fn)
        # # 创建张量
        # local_descs_rot.requires_grad_(True)
        # # 注册钩子
        # local_descs_rot.register_hook(hook_fn_2)

        # Take the global desc from the pose token
        global_desc_t = local_descs_t[:, 0, :]
        global_desc_rot = local_descs_rot[:, 0, :]

        return {'global_desc_t':global_desc_t, 'global_desc_rot':global_desc_rot, "scene_dist":scene_dist, "local_descs_t":local_descs_t, 'local_descs_rot':local_descs_rot}

        # if self.return_feature_plot:
        #     return {'global_desc_t':global_desc_t, 'global_desc_rot':global_desc_rot, "scene_dist":scene_dist, 'src_t_plot':src_t_plot, 'src_rot_plot':src_rot_plot}
        # else:
        #     return {'global_desc_t':global_desc_t, 'global_desc_rot':global_desc_rot, "scene_dist":scene_dist}

    def forward_heads(self, transformers_res):
        """
        The forward pass execpts a dictionary with two keys-values:
        global_desc_t: latent representation from the position encoder
        global_dec_rot: latent representation from the orientation encoder
        returns: dictionary with key-value 'pose'--expected pose (NX7)
        """
        global_desc_t = transformers_res.get('global_desc_t')
        global_desc_rot = transformers_res.get('global_desc_rot')

        local_descs_t = transformers_res.get('local_descs_t')
        local_descs_rot = transformers_res.get('local_descs_rot')



        ################  mamba start t #####################
        global_desc_t = global_desc_t.unsqueeze(1)

        # print('self.use_classical_mamba = ',self.use_classical_mamba)
        if self.use_classical_mamba:
            # print('self.use_classical_mamba = ',self.use_classical_mamba)
            #----------非 non local------------------
            # print('before mamba1_model_ori_B1D, global_desc_t shape =', global_desc_t.shape)#([8, 2, 256])
            global_desc_t = self.mamba1_model_ori_B1D(global_desc_t)
            # print('after mamba1_model_ori_B1D, global_desc_t shape =', global_desc_t.shape) #([B,1, 1024])
            global_desc_t = global_desc_t.squeeze(1)
            # print('after global_desc_t[:,1,:], global_desc_t.shape = ',global_desc_t.shape)
            #----------非 non local------------------
        else:
            # #---------- non local 专属----------------------------------------
            # print('after unsqueeze, global_desc_t.shape = ',global_desc_t.shape) #([8, 1, 256]
            # 将4的顺序反过来
            x1t = global_desc_t.flip(dims=[2])
            # print('after flip, global_desc_t.shape = ',global_desc_t.shape) #([8, 1, 256])
            # 在第一个维度拼接 x1 和 global_desc_t
            global_desc_t = torch.cat((x1t, global_desc_t), dim=1) 

            # print('before mamba1_model_ori_B1D, global_desc_t shape =', global_desc_t.shape)#([8, 2, 256])
            global_desc_t = self.mamba1_model_ori_B1D(global_desc_t)
            # print('after mamba1_model_ori_B1D, global_desc_t shape =', global_desc_t.shape) #([B,1, 1024])
            global_desc_t = global_desc_t[:,1,:]
            # print('after global_desc_t[:,1,:], global_desc_t.shape = ',global_desc_t.shape)
            # #---------- non local 专属--------------------------------

        
        
        if self.return_B1D_feature:
            feature_map_B1D_t = global_desc_t.clone()  # 保存特征图的深拷贝
        ################  mamba end #####################


        ################  mamba start r #####################
        global_desc_rot = global_desc_rot.unsqueeze(1)
        # print('after unsqueeze, global_desc_rot.shape = ',global_desc_rot.shape) #([8, 1, 256]
        
        if self.use_classical_mamba:
            #----------非 non local------------------
            # print('before mamba1_model_ori_B1D, global_desc_t shape =', global_desc_t.shape)#([8, 2, 256])
            global_desc_rot = self.mamba1_model_ori_B1D(global_desc_rot)
            # print('after mamba1_model_ori_B1D, global_desc_t shape =', global_desc_t.shape) #([B,1, 1024])
            global_desc_rot = global_desc_rot.squeeze(1)
            # print('after global_desc_t[:,1,:], global_desc_t.shape = ',global_desc_t.shape)
            #----------非 non local------------------

        else:
            #---------- non local 专属--------------------------------
            # 将4的顺序反过来
            x1r = global_desc_rot.flip(dims=[2])
            # print('after flip, global_desc_rot.shape = ',global_desc_rot.shape) #([8, 1, 256])
            # 在第一个维度拼接 x1 和 global_desc_rot
            global_desc_rot = torch.cat((x1r, global_desc_rot), dim=1) 

            # print('before mamba1_model_ori_B1D, global_desc_rot shape =', global_desc_rot.shape)#([8, 2, 256])
            global_desc_rot = self.mamba1_model_ori_B1D(global_desc_rot)
            # print('after mamba1_model_ori_B1D, global_desc_rot shape =', global_desc_rot.shape) #([B,1, 1024])
            global_desc_rot = global_desc_rot[:,1,:] #torch.Size([8, 256])
            # print('after global_desc_rot[:,1,:], global_desc_rot.shape = ',global_desc_rot.shape)
            #---------- non local 专属--------------------------------



        # global_desc_rot = global_desc_rot.squeeze(1)
        if self.return_B1D_feature:
            feature_map_B1D_r = global_desc_rot.clone()  # 保存特征图的深拷贝
        ################  mamba end #####################
        # print('before self.regressor_head_t, global_desc_t.shape = ',global_desc_t.shape) #torch.Size([8, 256])
        # print('global_desc_rot.shape = ',global_desc_rot.shape) #torch.Size([8, 256])


        x_t = self.regressor_head_t(global_desc_t)
        if self.use_prior: #False
            global_desc_rot = torch.cat((global_desc_t, global_desc_rot), dim=1)

        x_rot = self.regressor_head_rot(global_desc_rot)
        expected_pose = torch.cat((x_t, x_rot), dim=1)
        # print('self.return_B1D_feature = ',self.return_B1D_feature, 'return_feature_plot =',self.return_feature_plot)

        
        if self.return_B1D_feature:
            return {'pose': expected_pose, "scene_dist":transformers_res.get("scene_dist"), 'feature_map_t': feature_map_B1D_t, 'feature_map_r': feature_map_B1D_r ,'global_desc_t':global_desc_t,'global_desc_r':global_desc_rot}
        else:
            # return {'pose': expected_pose, "scene_dist":transformers_res.get("scene_dist")}
            return {'pose': expected_pose, "scene_dist":transformers_res.get("scene_dist"),'local_descs_t':local_descs_t, 'local_descs_rot':local_descs_rot,'global_desc_t':global_desc_t,'global_desc_r':global_desc_rot}
    
    def forward(self, data):
        """ The forward pass expects a dictionary with key-value 'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED

            returns dictionary with key-value 'pose'--expected pose (NX7)
        """
        transformers_encoders_res = self.forward_transformers(data)
        # Regress the pose from the image descriptors
        heads_res = self.forward_heads(transformers_encoders_res)
        return heads_res


class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim, use_prior=False):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the outpur dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(decoder_dim * 2, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))

        return self.fc_o(x)
