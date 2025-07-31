"""
Entry point training and testing TransPoseNet
"""
import argparse
import torch
import numpy as np
import json
import logging
from util123 import utils
import time
from datasets123.CameraPoseDataset_anyloc_cambridge import CameraPoseDataset_anyloc_cambridge
# from datasets123.CameraPoseDataset import CameraPoseDataset
# from datasets123.MSCameraPoseDataset import MSCameraPoseDataset
from models.pose_losses import CameraPoseLoss
from models.pose_regressors_anyloc import get_model
from os.path import join
import os
import ipdb as pdb


# Function to count the number of learnable parameters and convert to MB
def count_learnable_parameters_in_mb(model):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = (param_count * 4) / (1024 * 1024)  # Convert to MB
    return param_count, param_size_mb

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name",
                            help="name of model to create (e.g. posenet, transposenet")
    arg_parser.add_argument("mode", help="train or eval")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")

    #mambaloc
    arg_parser.add_argument("--scene", help="scene name")
    arg_parser.add_argument('--skipvim', type=int, default=1)
    arg_parser.add_argument('--tab', type=str, help="输入格式为 'float1-float2'", default=None)
    arg_parser.add_argument('--return_vim_features', type=bool, default=False, help="vim是否返回特征图")
    arg_parser.add_argument('--use_mamba', type=bool, default=False, help="是否用mamba")
    arg_parser.add_argument('--use_classical_mamba', type=bool, default=False, help="是否用mamba")
    arg_parser.add_argument('--current_epoch', type=int, default=1)
    arg_parser.add_argument('--pick_saved_epoch', type=bool, default=False, help="是否pick_saved_epoch")
    arg_parser.add_argument('--results_type', type=int, default=0)
    # arg_parser.add_argument('--no_shuffle', type=bool, default=True)

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {} with {}".format(args.model_name, args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open('config_anyloc_cambridge.json', "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
   
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Create the model
    model = get_model(args.model_name, args.backbone_path, args.use_mamba, config,use_classical_mamba=args.use_classical_mamba).to(device)

    # Count and log the number of learnable parameters for each model
    teacher_params_count = count_learnable_parameters_in_mb(model)
  

    logging.info("Number of learnable parameters in the teacher model: {}".format(teacher_params_count))

    print('Number of learnable parameters in the teacher model: ', teacher_params_count)
   
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    print('args.checkpoint_path = ',args.checkpoint_path)
    
    multiscene = model.classify_scene
    classify_scene = model.classify_scene

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Freeze parts of the model if indi cated
        freeze = config.get("freeze") # freeze =  False
        freeze_exclude_phrase = config.get("freeze_exclude_phrase") # freeze_exclude_phrase =  regressor_head_rot
        if isinstance(freeze_exclude_phrase, str):
            freeze_exclude_phrase = [freeze_exclude_phrase]
        if freeze:
            for name, parameter in model.named_parameters():
                freeze_param = True
                for phrase in freeze_exclude_phrase:
                    if phrase in name:
                        freeze_param = False
                        break
                if freeze_param:
                        parameter.requires_grad_(False)

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)
        nll_loss = torch.nn.NLLLoss()

        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=1.1)#config.get('lr_scheduler_gamma'))

        # Set the dataset and data loader
        no_augment = config.get("no_augment")
        if no_augment:
            transform = utils.test_transforms.get('baseline')
        else:
            transform = utils.train_transforms.get('baseline')

        if multiscene:
            equalize_scenes = config.get("equalize_scenes")
            dataset = MSCameraPoseDataset(args.dataset_path, args.labels_file, transform, equalize_scenes=equalize_scenes)
        else:
            dataset = CameraPoseDataset_anyloc_cambridge(args.dataset_path, args.labels_file, transform, args.skipvim, args.tab, scene=args.scene, results_type=args.results_type)

        if args.use_mamba:
            loader_params = {'batch_size': config.get('batch_size'),
                                    'shuffle': True,
                                    'num_workers': 16}
        else:
            loader_params = {'batch_size': config.get('batch_size'),
                                    'shuffle': True,
                                    'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device).to(dtype=torch.float32)
                gt_pose = minibatch.get('pose')
                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                if freeze: # For TransPoseNet
                    model.eval()
                    with torch.no_grad():
                        transformers_res = model.forward_transformers(minibatch)
                    model.train()

                # Zero the gradients
                optim.zero_grad()

                # Forward pass to estimate the pose
                if freeze:
                    res = model.forward_heads(transformers_res)
                else:
                    res = model(minibatch)

                est_pose = res.get('pose')
                # Pose loss

                if multiscene and classify_scene:
                    est_scene = res.get('scene_dist')
                    gt_scene = minibatch.get('scene').to(dtype=torch.int64)
                    criterion = pose_loss(est_pose, gt_pose) + nll_loss(est_scene, gt_scene)
                else:
                    criterion = pose_loss(est_pose, gt_pose)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_pose.detach(), gt_pose.detach())
                    logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                                 "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item()))
            # Save checkpoint
            n_freq_checkpoint = 1
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
            
                if args.use_mamba:
                    weights_filename = f'{checkpoint_prefix}_{args.scene}_mamba_skip_{args.skipvim}_checkpoint-{epoch}.pth' 
                else:
                    weights_filename = f'{checkpoint_prefix}_{args.scene}_skip_{args.skipvim}_checkpoint-{epoch}.pth' 
                
                torch.save(model.state_dict(), weights_filename)



                test_at_hard01 = True
                test_at_good = True


                if test_at_good:
                    print('\n\ ================ test_at_good start ================\n')
                    #直接调用验证集验证然后删掉
                    # weights_filename = f'{checkpoint_prefix}_{args.scene}_checkpoint_{epoch}.pth' 
                    log_filename = f'{weights_filename}.log' 
                    
                    if args.use_mamba:
                    
                        if args.use_classical_mamba:
                            if args.scene in ['OldHospital', 'KingsCollege', 'ShopFacade', 'StMarysChurch']:
                                command = f"python main_anyloc_cambridge.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/Cambridge /home/transposenet/datasets123/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --use_mamba True --scene {args.scene} --skipvim {args.skipvim} --use_classical_mamba True --current_epoch {epoch} --results_type 2" #/home/transposenet/out/{checkpoint_prefix}_checkpoint-{epoch}.pth"
                            else: 
                                command = f"python main_anyloc_cambridge.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/7Scenes /home/transposenet/datasets123/7Scenes/abs_7scenes_pose.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --use_mamba True --scene {args.scene} --skipvim {args.skipvim} --use_classical_mamba True --current_epoch {epoch} --results_type 2" 
                        else:
                            if args.scene in ['OldHospital', 'KingsCollege', 'ShopFacade', 'StMarysChurch']:
                                command = f"python main_anyloc_cambridge.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/Cambridge /home/transposenet/datasets123/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --use_mamba True --scene {args.scene} --skipvim {args.skipvim} --current_epoch {epoch} --results_type 2" #/home/transposenet/out/{checkpoint_prefix}_checkpoint-{epoch}.pth"
                            else: 
                                command = f"python main_anyloc_cambridge.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/7Scenes /home/transposenet/datasets123/7Scenes/abs_7scenes_pose.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --use_mamba True --scene {args.scene} --skipvim {args.skipvim} --current_epoch {epoch} --results_type 2" 

                    
                    else:
                        if args.scene in ['OldHospital', 'KingsCollege', 'ShopFacade', 'StMarysChurch']:
                            command = f"python main_anyloc_cambridge.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/Cambridge /home/transposenet/datasets123/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --scene {args.scene} --skipvim {args.skipvim} --current_epoch {epoch} --results_type 2" #/home/transposenet/out/{checkpoint_prefix}_checkpoint-{epoch}.pth"
                        else: 
                            command = f"python main_anyloc_cambridge.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/7Scenes /home/transposenet/datasets123/7Scenes/abs_7scenes_pose.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --scene {args.scene} --skipvim {args.skipvim} --current_epoch {epoch} --results_type 2" 

                    os.system(command)
                    # if 1: #epoch <330:
                    #     print("Deleting weights file:", weights_filename)
                    #     os.remove(weights_filename)
                    #     command = f"rm -rf /home/transposenet/out/*.log"
                    #     os.system(command)


                # ------------------------------ test_at_hard01 -------------------------------------------------------
                if test_at_hard01:
                    print('\n\ ================ test_at_hard01 start ================\n')
                    #直接调用验证集验证然后删掉
                    # weights_filename = f'{checkpoint_prefix}_{args.scene}_checkpoint_{epoch}.pth' 
                    log_filename = f'{weights_filename}.log' 
                    
                    if args.use_mamba:
                    
                        if args.use_classical_mamba:
                            if args.scene in ['OldHospital', 'KingsCollege', 'ShopFacade', 'StMarysChurch']:
                                command = f"python main_anyloc_cambridge.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/Cambridge /home/transposenet/datasets123/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{args.scene}_test_hard01.csv --checkpoint_path {weights_filename} --use_mamba True --scene {args.scene} --skipvim {args.skipvim} --use_classical_mamba True --current_epoch {epoch} --pick_saved_epoch True --results_type 1" #/home/transposenet/out/{checkpoint_prefix}_checkpoint-{epoch}.pth"
                            else: 
                                command = f"python main_anyloc_cambridge.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/7Scenes /home/transposenet/datasets123/7Scenes/abs_7scenes_pose.csv_{args.scene}_test_hard01.csv --checkpoint_path {weights_filename} --use_mamba True --scene {args.scene} --skipvim {args.skipvim} --use_classical_mamba True --current_epoch {epoch} --pick_saved_epoch True --results_type 1" 
                        else:
                            if args.scene in ['OldHospital', 'KingsCollege', 'ShopFacade', 'StMarysChurch']:
                                command = f"python main_anyloc_cambridge.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/Cambridge /home/transposenet/datasets123/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{args.scene}_test_hard01.csv --checkpoint_path {weights_filename} --use_mamba True --scene {args.scene} --skipvim {args.skipvim} --current_epoch {epoch} --pick_saved_epoch True --results_type 1" #/home/transposenet/out/{checkpoint_prefix}_checkpoint-{epoch}.pth"
                            else: 
                                command = f"python main_anyloc_cambridge.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/7Scenes /home/transposenet/datasets123/7Scenes/abs_7scenes_pose.csv_{args.scene}_test_hard01.csv --checkpoint_path {weights_filename} --use_mamba True --scene {args.scene} --skipvim {args.skipvim} --current_epoch {epoch} --pick_saved_epoch True --results_type 1" 

                    
                    else:
                        if args.scene in ['OldHospital', 'KingsCollege', 'ShopFacade', 'StMarysChurch']:
                            command = f"python main_anyloc_cambridge.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/Cambridge /home/transposenet/datasets123/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{args.scene}_test_hard01.csv --checkpoint_path {weights_filename} --scene {args.scene} --skipvim {args.skipvim} --current_epoch {epoch} --pick_saved_epoch True --results_type 1" #/home/transposenet/out/{checkpoint_prefix}_checkpoint-{epoch}.pth"
                        else: 
                            command = f"python main_anyloc_cambridge.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/7Scenes /home/transposenet/datasets123/7Scenes/abs_7scenes_pose.csv_{args.scene}_test_hard01.csv --checkpoint_path {weights_filename} --scene {args.scene} --skipvim {args.skipvim} --current_epoch {epoch} --pick_saved_epoch True --results_type 1" 

                    os.system(command)
                    # if 1: #epoch <330:
                    #     print("Deleting weights file:", weights_filename)
                    #     os.remove(weights_filename)
                        # command = f"rm -rf /home/transposenet/out/*.log"
                        # os.system(command)

            # # Plot the loss function
            # loss_fig_path = checkpoint_prefix + "_loss_fig.png"
            # utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

        # # Plot the loss function
        # loss_fig_path = checkpoint_prefix + "_loss_fig.png"
        # utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)

    else: # Test
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        if multiscene:
            dataset = MSCameraPoseDataset(args.dataset_path, args.labels_file, transform)
        else:
            dataset = CameraPoseDataset_anyloc_cambridge(args.dataset_path, args.labels_file, transform, scene=args.scene, results_type = args.results_type)
        if args.use_mamba:
            loader_params = {'batch_size': 1,
                            'shuffle': False,
                            'num_workers': 16}
        else:
            loader_params = {'batch_size': 1,
                            'shuffle': False,
                            'num_workers': config.get('n_workers')}

        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats = np.zeros((len(dataloader.dataset), 3))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device).to(dtype=torch.float32)

                if multiscene and classify_scene:
                    # at Test time the classifier will determine the scene at a multiscene scenario
                    minibatch['scene'] = None


                gt_pose = minibatch.get('pose')

                # Forward pass to predict the pose
                tic = time.time()
                est_pose = model(minibatch).get('pose')
                toc = time.time()

                # Evaluate error
                posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

                # Collect statistics
                stats[i, 0] = posit_err.item()
                stats[i, 1] = orient_err.item()
                stats[i, 2] = (toc - tic)*1000

                logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    stats[i, 0],  stats[i, 1],  stats[i, 2]))
 
        
        if args.results_type ==1:
            results_type = "hard01"
        if args.results_type ==2:
            results_type = "good"
        print("args.results_type = ",args.results_type,'results_type = ',results_type )
        
        
        if args.use_mamba:
            if args.use_classical_mamba:
                results_filename = f'results_{results_type}_anyloc_use_classical_mamba_{args.scene}_skip_{args.skipvim}.txt'
            else:
                results_filename = f'results_{results_type}_anyloc_mamba_{args.scene}_skip_{args.skipvim}.txt'
            
        else:
            results_filename = f'results_{results_type}_{args.scene}_skip_{args.skipvim}.txt'

        with open(results_filename, 'a') as file:
            file.write(f"Performance of {args.checkpoint_path} on {args.labels_file}\n")
            file.write(f"Median pose error: {np.nanmedian(stats[:, 0])}[m], {np.nanmedian(stats[:, 1])}[deg]\n")
            file.write(f"Mean inference time: {np.mean(stats[:, 2])}\n")
        # Record overall statistics
        logging.info("\nPerformance of {} on {}".format(args.checkpoint_path, args.labels_file))
        logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]), np.nanmedian(stats[:, 1])))
        logging.info(
            "Var pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanstd(stats[:, 0])**2, np.nanstd(stats[:, 1])**2))
        logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))

        if args.pick_saved_epoch:
            print('\n pick_saved_epoch\n')
            

            # 计算当前验证集的 Median pose error
            current_median_pose_err_m = np.nanmedian(stats[:, 0])
            current_median_pose_err_deg = np.nanmedian(stats[:, 1])

            print("Current Median pose error: {:.3f}[m], {:.3f}[deg]".format(
                current_median_pose_err_m, current_median_pose_err_deg))

            # 定义存储最小值的文件名
            if args.use_mamba:
                if args.use_classical_mamba:
                    min_pose_error_filename = f'min_results_anyloc_use_classical_mamba_{args.scene}_skip_{args.skipvim}.txt'
                else:
                    min_pose_error_filename = f'min_results_anyloc_mamba_{args.scene}_skip_{args.skipvim}.txt'
                
            else:
                min_pose_error_filename = f'min_results_{args.scene}_skip_{args.skipvim}.txt'

            print('args.current_epoch =',args.current_epoch)
            if args.current_epoch==1:
                # 初始化最小的 Median pose error 为无穷大
                min_median_pose_err_m = float('inf')
                min_median_pose_err_deg = float('inf')

                # 创建文件并写入初始最小值
                with open(min_pose_error_filename, 'w') as f:
                    f.write(f"{min_median_pose_err_m:.3f},{min_median_pose_err_deg:.3f}\n")


                # 测试： 从文件中读取最后一行，提取并解析为原来的数据格式
                if os.path.exists(min_pose_error_filename):
                    with open(min_pose_error_filename, 'r') as f:
                        last_line = f.readlines()[-1].strip()  # 获取最后一行并去除换行符
                        initial_last_min_median_pose_err_m, initial_last_min_median_pose_err_deg = map(float, last_line.split(','))

                        print("Last Min Median pose error from file: {:.3f}[m], {:.3f}[deg]".format(
                            initial_last_min_median_pose_err_m, initial_last_min_median_pose_err_deg))
            else: 
                # 从文件中读取最后一行，提取并解析为原来的数据格式
                if os.path.exists(min_pose_error_filename):
                    with open(min_pose_error_filename, 'r') as f:
                        last_line = f.readlines()[-1].strip()  # 获取最后一行并去除换行符
                        min_median_pose_err_m, min_median_pose_err_deg = map(float, last_line.split(','))

                        print("Last Min Median pose error from file: {:.3f}[m], {:.3f}[deg]".format(
                            min_median_pose_err_m, min_median_pose_err_deg))
            


            # 根据当前 Median pose error 和最小值进行判断
            weights_filename = args.checkpoint_path
            current_scene = args.scene

            if args.current_epoch > 1:
                command = f"rm -rf /home/transposenet/out/*.log"
                os.system(command)
                # 检查 weights_filename 是否包含 current_scene
                if current_scene in weights_filename:
                    # 进行当前 Median pose error 和最小值的比较判断
                    # if (current_median_pose_err_m > min_median_pose_err_m and 
                    #     current_median_pose_err_deg > min_median_pose_err_deg):
                    #     logging.info("Deleting weights file: {}".format(weights_filename))
                    #       # 你可以在调试时使用 
                    #     os.remove(weights_filename)

                    # elif (current_median_pose_err_m > min_median_pose_err_m and 
                    #     current_median_pose_err_deg <= min_median_pose_err_deg):
                    #     logging.info("Keeping weights file: {}".format(weights_filename))
                    #     # 不执行删除操作

                    if (current_median_pose_err_m <= min_median_pose_err_m and 
                        current_median_pose_err_deg <= min_median_pose_err_deg):
                        # 如果当前误差小于等于最小误差，删除以前的所有权重文件，保留当前文件
                        logging.info("Deleting all old weights files except: {}".format(weights_filename))                 
                        keep_file_name = os.path.basename(weights_filename) # 使用 os.path.basename() 提取文件名部分
                        for file in os.listdir(os.path.dirname(weights_filename)):
                            # 检查文件名中是否包含 current_scene
                            if file.endswith(".pth") and file != keep_file_name and current_scene in file:
                                print('keep_file_name = ',keep_file_name, 'file = ',file)
                                logging.info("Deleting old weights file: {}".format(file))
                                  # 你可以在调试时使用 
                                os.remove(os.path.join(os.path.dirname(weights_filename), file))
                            else:
                                logging.info("Skipping old file as it does not match the current scene: {}".format(file))
                else:
                    logging.info("Skipping file as it does not match the current scene: {}".format(weights_filename))
            
            # 判断是否更新最小的 Median pose error
            if current_median_pose_err_m < min_median_pose_err_m:
                min_median_pose_err_m = current_median_pose_err_m
            if current_median_pose_err_deg < min_median_pose_err_deg:
                min_median_pose_err_deg = current_median_pose_err_deg


            # 将最小的 min_median_pose_err_m 和 min_median_pose_err_deg 覆盖写入文件
            with open(min_pose_error_filename, 'w') as f:
                f.write(f"{min_median_pose_err_m:.3f},{min_median_pose_err_deg:.3f}\n")


            print("Now, Min Median pose error: {:.3f}[m], {:.3f}[deg]".format(
                min_median_pose_err_m, min_median_pose_err_deg))
            