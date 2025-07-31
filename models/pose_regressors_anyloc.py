from .posenet.PoseNet import PoseNet
from .transposenet.TransPoseNet_anyloc import TransPoseNet_anyloc
# from .transposenet.TransPoseNet import TransPoseNet
from .transposenet.TransPoseNet_mamba_anyloc import TransPoseNet_mamba_anyloc

def get_model(model_name, backbone_path,use_mamba:False,config,return_feature=False, return_feature_plot =False,use_classical_mamba=False):
    """
    Get the instance of the request model
    :param model_name: (str) model name
    :param backbone_path: (str) path to a .pth backbone
    :param config: (dict) config file
    :return: instance of the model (nn.Module)
    """
    

    if model_name == 'posenet':
        return PoseNet(backbone_path)
    elif model_name == 'transposenet':
        print('\n\n ====================== use_mamba = ',use_mamba)
        if use_mamba:
            return TransPoseNet_mamba_anyloc(config, backbone_path,return_feature=return_feature, return_feature_plot = return_feature_plot,use_classical_mamba=use_classical_mamba)
        else:
            return TransPoseNet_anyloc(config, backbone_path,return_feature=return_feature)
            
      
    else:
        raise "{} not supported".format(model_name)