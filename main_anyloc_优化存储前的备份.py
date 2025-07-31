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
from datasets123.CameraPoseDataset_anyloc import CameraPoseDataset_anyloc
# from datasets123.CameraPoseDataset import CameraPoseDataset
# from datasets123.MSCameraPoseDataset import MSCameraPoseDataset
from models.pose_losses import CameraPoseLoss
from models.pose_regressors_anyloc import get_model
from os.path import join
import os


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
    with open('config_anyloc.json', "r") as read_file:
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
            dataset = CameraPoseDataset_anyloc(args.dataset_path, args.labels_file, transform, args.skipvim, args.tab)

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

                #直接调用验证集验证然后删掉
                # weights_filename = f'{checkpoint_prefix}_{args.scene}_checkpoint_{epoch}.pth' 
                log_filename = f'{weights_filename}.log' 
                
                if args.use_mamba:
                
                    if args.use_classical_mamba:
                        if args.scene in ['OldHospital', 'KingsCollege', 'ShopFacade', 'StMarysChurch']:
                            command = f"python main_anyloc.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/Cambridge /home/transposenet/datasets123/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --use_mamba True --scene {args.scene} --skipvim {args.skipvim} --use_classical_mamba True" #/home/transposenet/out/{checkpoint_prefix}_checkpoint-{epoch}.pth"
                        else: 
                            command = f"python main_anyloc.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/7Scenes /home/transposenet/datasets123/7Scenes/abs_7scenes_pose.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --use_mamba True --scene {args.scene} --skipvim {args.skipvim} --use_classical_mamba True" 
                    else:
                        if args.scene in ['OldHospital', 'KingsCollege', 'ShopFacade', 'StMarysChurch']:
                            command = f"python main_anyloc.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/Cambridge /home/transposenet/datasets123/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --use_mamba True --scene {args.scene} --skipvim {args.skipvim}" #/home/transposenet/out/{checkpoint_prefix}_checkpoint-{epoch}.pth"
                        else: 
                            command = f"python main_anyloc.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/7Scenes /home/transposenet/datasets123/7Scenes/abs_7scenes_pose.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --use_mamba True --scene {args.scene} --skipvim {args.skipvim}" 

                
                else:
                    if args.scene in ['OldHospital', 'KingsCollege', 'ShopFacade', 'StMarysChurch']:
                        command = f"python main_anyloc.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/Cambridge /home/transposenet/datasets123/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --scene {args.scene} --skipvim {args.skipvim}" #/home/transposenet/out/{checkpoint_prefix}_checkpoint-{epoch}.pth"
                    else: 
                        command = f"python main_anyloc.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/7Scenes /home/transposenet/datasets123/7Scenes/abs_7scenes_pose.csv_{args.scene}_test.csv --checkpoint_path {weights_filename} --scene {args.scene} --skipvim {args.skipvim}" 

                os.system(command)
                # if 1: #epoch <330:
                #     print("Deleting weights file:", weights_filename)
                #     os.remove(weights_filename)
                #     command = f"rm -rf /home/transposenet/out/*.log"
                #     os.system(command)

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
            dataset = CameraPoseDataset_anyloc(args.dataset_path, args.labels_file, transform)
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
 
     
        if args.use_mamba:
            if args.use_classical_mamba:
                results_filename = f'results_anyloc_use_classical_mamba_{args.scene}_skip_{args.skipvim}.txt'
            else:
                results_filename = f'results_anyloc_mamba_{args.scene}_skip_{args.skipvim}.txt'
            
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