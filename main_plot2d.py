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
from datasets123.CameraPoseDataset import CameraPoseDataset
from datasets123.MSCameraPoseDataset import MSCameraPoseDataset
from models.pose_losses import CameraPoseLoss
from models.pose_regressors import get_model
from os.path import join
import os
import os.path as osp

import matplotlib.pyplot as plt  # Import for plotting


import matplotlib.pyplot as plt
import numpy as np
import pdb

import matplotlib.pyplot as plt
import numpy as np
import sys

def vis_pose(vis_info, image_filename):
    '''
    Visualize predicted pose result vs. ground truth pose
    '''
    pose = vis_info['pose']
    pose_gt = vis_info['pose_gt']
    theta = vis_info['theta']
    ang_threshold = 10  # Angular error threshold
    seq_num = theta.shape[0]

    # Calculate the min and max values for the axes
    x_min = min(pose[:, 0].min(), pose_gt[:, 0].min())
    x_max = max(pose[:, 0].max(), pose_gt[:, 0].max())
    y_min = min(pose[:, 1].min(), pose_gt[:, 1].min())
    y_max = max(pose[:, 1].max(), pose_gt[:, 1].max())
    z_min = min(pose[:, 2].min(), pose_gt[:, 2].min())
    z_max = max(pose[:, 2].max(), pose_gt[:, 2].max())

    # Create figure object
    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    # Plot translation trajectory
    ax1 = fig.add_axes([0, 0.2, 0.9, 0.85], projection='3d')
    ax1.scatter(pose[10:, 0], pose[10:, 1], zs=pose[10:, 2], c='r', s=3**2, depthshade=0)  # Predicted
    ax1.scatter(pose_gt[:, 0], pose_gt[:, 1], zs=pose_gt[:, 2], c='g', s=3**2, depthshade=0)  # GT
    ax1.scatter(pose[0:10, 0], pose[0:10, 1], zs=pose[0:10, 2], c='k', s=3**2, depthshade=0)  # Predicted

    ax1.view_init(30, 120)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)

    # Plot angular error
    ax2 = fig.add_axes([0.1, 0.05, 0.75, 0.2])
    err = theta.reshape(1, seq_num)
    err = np.tile(err, (20, 1))
    ax2.imshow(err, vmin=0, vmax=ang_threshold, aspect=3)
    ax2.set_yticks([])
    ax2.set_xticks([0, seq_num*1/5, seq_num*2/5, seq_num*3/5, seq_num*4/5, seq_num])

    # Save the figure
    plt.savefig(image_filename, dpi=300) #50
    plt.close(fig)



# def vis_pose(vis_info, image_filename):
#     '''
#     Visualize predicted pose result vs. ground truth pose
#     '''
#     # pdb.set_trace()
#     pose = vis_info['pose']
#     pose_gt = vis_info['pose_gt']
#     theta = vis_info['theta']
#     ang_threshold = 10  # Angular error threshold
#     seq_num = theta.shape[0]

#     # Create figure object
#     fig = plt.figure(figsize=(8, 6))
#     plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

#     # Plot translation trajectory
#     ax1 = fig.add_axes([0, 0.2, 0.9, 0.85], projection='3d')
#     ax1.scatter(pose[10:, 0], pose[10:, 1], zs=pose[10:, 2], c='r', s=3**2, depthshade=0)  # Predicted
#     ax1.scatter(pose_gt[:, 0], pose_gt[:, 1], zs=pose_gt[:, 2], c='g', s=3**2, depthshade=0)  # GT
#     ax1.scatter(pose[0:10, 0], pose[0:10, 1], zs=pose[0:10, 2], c='k', s=3**2, depthshade=0)  # Predicted

#     ax1.view_init(30, 120)
#     ax1.set_xlabel('x (m)')
#     ax1.set_ylabel('y (m)')
#     ax1.set_zlabel('z (m)')
    
#     ax1.set_xlim(-1, 1)
#     ax1.set_ylim(-1, 1)
#     ax1.set_zlim(-1, 1)

#     # Plot angular error
#     ax2 = fig.add_axes([0.1, 0.05, 0.75, 0.2])
#     err = theta.reshape(1, seq_num)
#     err = np.tile(err, (20, 1))
#     ax2.imshow(err, vmin=0, vmax=ang_threshold, aspect=3)
#     ax2.set_yticks([])
#     ax2.set_xticks([0, seq_num*1/5, seq_num*2/5, seq_num*3/5, seq_num*4/5, seq_num])

#     # Save the figure
#     plt.savefig(image_filename, dpi=300) #50
#     plt.close(fig)

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
    arg_parser.add_argument('--results_type', type=int, default=1)
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
    with open('config.json', "r") as read_file:
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
    model = get_model(args.model_name, args.backbone_path, args.use_mamba, config).to(device)

    # Count and log the number of learnable parameters for each model
    teacher_params_count = count_learnable_parameters_in_mb(model)
  

    logging.info("Number of learnable parameters in the teacher model: {}".format(teacher_params_count))

    print('Number of learnable parameters in the teacher model: ', teacher_params_count)
   
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

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
            dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform, args.skipvim, args.tab)

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
            if (epoch % n_freq_checkpoint) == 0 and epoch > 10:
                
                
                if args.use_mamba:
                    weights_filename = f'{checkpoint_prefix}_{args.scene}_mamba_skip_{args.skipvim}_checkpoint-{epoch}.pth' 
                else:
                    weights_filename = f'{checkpoint_prefix}_{args.scene}_skip_{args.skipvim}_checkpoint-{epoch}.pth' 
                
                torch.save(model.state_dict(), weights_filename)

                #直接调用验证集验证然后删掉
                # weights_filename = f'{checkpoint_prefix}_{args.scene}_checkpoint_{epoch}.pth' 
                log_filename = f'{checkpoint_prefix}.log' 
                
                if args.use_mamba:
                    if args.scene in ['OldHospital', 'KingsCollege', 'ShopFacade', 'StMarysChurch']:
                        command = f"python main_plot2d.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/Cambridge /home/transposenet/datasets123/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --use_mamba True --scene {args.scene} --skipvim {args.skipvim}" #/home/transposenet/out/{checkpoint_prefix}_checkpoint-{epoch}.pth"
                    else: 
                        command = f"python main_plot2d.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/7Scenes /home/transposenet/datasets123/7Scenes/abs_7scenes_pose.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --use_mamba True --scene {args.scene} --skipvim {args.skipvim}" 
                else:
                    if args.scene in ['OldHospital', 'KingsCollege', 'ShopFacade', 'StMarysChurch']:
                        command = f"python main_plot2d.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/Cambridge /home/transposenet/datasets123/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --scene {args.scene} --skipvim {args.skipvim}" #/home/transposenet/out/{checkpoint_prefix}_checkpoint-{epoch}.pth"
                    else: 
                        command = f"python main_plot2d.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/7Scenes /home/transposenet/datasets123/7Scenes/abs_7scenes_pose.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --scene {args.scene} --skipvim {args.skipvim}" 

                os.system(command)
                if epoch<10 or epoch>30:
                    print("Deleting weights file:", weights_filename)
                    os.remove(weights_filename)
                    command = f"rm -rf /home/transposenet/out/*.log"
                    os.system(command)

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

        if args.results_type == 1: #通过口令里面设置cvs=test_hard01.csv文件实现 
            print('test in hard01')
            test_condition = 'hard01'
        elif args.results_type == 2: #通过口令里面设置cvs=test.csv文件实现
            print('test in good')
            test_condition = 'good'
        elif args.results_type == 3: #通过口令里面设置cvs=test.csv文件然后修改transform实现（TODO）
            print('自定义，没有写!')
            # test_condition = 
            sys.exit()
        # pdb.set_trace()
       
        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        if multiscene:
            dataset = MSCameraPoseDataset(args.dataset_path, args.labels_file, transform)
        else:
            dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)
        if args.use_mamba:
            loader_params = {'batch_size': 1,
                            'shuffle': False,
                            'num_workers': 16}
        else:
            loader_params = {'batch_size': 1,
                            'shuffle': False,
                            'num_workers': 16}

        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
        pred_poses = []
        gt_poses = []
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

                # Collect poses for plotting
                pred_poses.append(est_pose.cpu().numpy())
                gt_poses.append(gt_pose.cpu().numpy())

                logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    stats[i, 0],  stats[i, 1],  stats[i, 2]))
 

        # Convert lists to numpy arrays
        pred_poses = np.vstack(pred_poses)
        gt_poses = np.vstack(gt_poses)

        # # Un-normalize the poses if necessary
        pose_stats_file = '/home/transposenet/data/pose_stats.txt'
        pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

        pred_poses[:, :3] = (pred_poses[:, :3] * pose_s) + pose_m
        gt_poses[:, :3] = (gt_poses[:, :3] * pose_s) + pose_m


        #########################  画2d/3d图开始 #############################################
        
        # 使用os.path.basename获取文件名
        filename = os.path.basename(args.checkpoint_path)
        filename_without_extension = os.path.splitext(filename)[0] # 去掉文件扩展名

        print('filename_without_extension = ',filename_without_extension)
        if args.use_mamba:
            image_filename = f'/home/transposenet/Visualizetrajectories/Transposenet_2D/{filename_without_extension}_2dmap_{test_condition}.png' 
        else:
            image_filename = f'/home/transposenet/Visualizetrajectories/Transposenet_2D/{filename_without_extension}_2dmap_{test_condition}.png' 

        # CAM画2D地图       
        if args.scene in ['OldHospital', 'KingsCollege', 'ShopFacade', 'StMarysChurch']:
            # Plot the poses
            fig = plt.figure()
            real_pose = (pred_poses[:, :3] - pose_m) / pose_s
            gt_pose = (gt_poses[:, :3] - pose_m) / pose_s
            plt.plot(gt_pose[:, 1], gt_pose[:, 0], color='black')
            plt.plot(real_pose[:, 1], real_pose[:, 0], color='red')
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=15)
            plt.title('Predicted vs Ground Truth Poses')
            plt.show(block=True)        

            fig.savefig(image_filename)

            print('saved image_filename = ', image_filename)   

        else: #用3D地图
            import math
            pred_poses = torch.from_numpy(pred_poses)
            gt_poses = torch.from_numpy(gt_poses)
            L = int(pred_poses.shape[0])
            results = np.zeros((L, 2))
            print('pred_poses.shape[0] = ',pred_poses.shape[0]) #{B,7}
            predict_pose_list=[]
            gt_pose_list=[]
            ang_error_list=[]
            pose_result_raw=[]
            pose_GT=[]
            for i in range(0,L,1):
                # pose_q = transforms.matrix_to_quaternion(torch.Tensor(pose[:,:3,:3]))#.cpu().numpy() # gnd truth in quaternion
                # pose_x = pose[:, :3, 3] # gnd truth position
                # predicted_q = transforms.matrix_to_quaternion(torch.Tensor(predict_pose[:,:3,:3]))#.cpu().numpy() # predict in quaternion
                # predicted_x = predict_pose[:, :3, 3] # predict position
                # pose_q = pose_q.squeeze() 
                # pose_x = pose_x.squeeze() 
                # predicted_q = predicted_q.squeeze() 
                # predicted_x = predicted_x.squeeze()

                predicted_q = pred_poses[i, 3:]
                predicted_x = pred_poses[i, :3]
                pose_q = gt_poses[i, 3:]
                pose_x = gt_poses[i, :3]
                #Compute Individual Sample Error 
                print('pose_q.shape = ',pose_q.shape)
                q1 = pose_q / torch.linalg.norm(pose_q)
                print('after torch.linalg.norm, q1.shape = ',q1.shape)
                q2 = predicted_q / torch.linalg.norm(predicted_q)
                print('after torch.linalg.norm, q2.shape = ',q2.shape)
                d = torch.abs(torch.sum(torch.matmul(q1,q2))) 
                d = torch.clamp(d, -1., 1.) # acos can only input [-1~1]
                print('d.shape = ',d.shape)
                theta = (2 * torch.acos(d) * 180/math.pi).numpy()
                print('theta.shape = ',theta.shape, 'theta = ',theta)
                error_x = torch.linalg.norm(torch.Tensor(pose_x-predicted_x)).numpy()
                print('error_x.shape = ',error_x.shape, 'error_x = ',error_x)
                print('i = ',i)
                results[i,:] = [error_x, theta]
                print('results = ',results)
                #print ('Iteration: {} Error XYZ (m): {} Error Q (degrees): {}'.format(i, error_x, theta)) 

                # save results for visualization
                predict_pose_list.append(predicted_x)
                gt_pose_list.append(pose_x)
                ang_error_list.append(theta)
                pose_result_raw.append(pred_poses)
                pose_GT.append(gt_poses)
                i += 1
            # pdb.set_trace()
            predict_pose_list = np.array(predict_pose_list)
            gt_pose_list = np.array(gt_pose_list)
            ang_error_list = np.array(ang_error_list)
            pose_result_raw = np.asarray(pose_result_raw)
            pose_GT = np.asarray(pose_GT)
            vis_info = {"pose": predict_pose_list, "pose_gt": gt_pose_list, "theta": ang_error_list, "pose_result_raw": pose_result_raw, "pose_GT": pose_GT}
            print('ang_error_list.shape = ',ang_error_list.shape)
            print('predict_pose_list.shape = ',predict_pose_list.shape)
            print('predictgt_pose_list_pose_list.shape = ',gt_pose_list.shape)
            vis_pose(vis_info, image_filename)
        #########################  画2d/3d图end #############################################




        if args.use_mamba:
            results_filename = f'results_mamba_{args.scene}_skip_{args.skipvim}.txt'
        else:
            results_filename = f'results_{args.scene}_skip_{args.skipvim}.txt'

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