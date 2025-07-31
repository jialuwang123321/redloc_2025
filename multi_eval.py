import os

# 加速
# source /etc/network_turbo

dataset = "7Scenes"
dataset_2 = "7scenes"
scene = 'stairs'
use_mamba = False


checkpoint = 'run_15_07_24_19_19'
if use_mamba:
    for N in range(50,650,50):
        if N == 600:
            command = f"python main.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/{dataset} /home/transposenet/datasets123/{dataset}/abs_{dataset_2}_pose.csv_{scene}_test.csv --checkpoint_path /home/transposenet/out/{checkpoint}_final.pth --use_mamba {use_mamba}"
        else:
            command = f"python main.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/{dataset} /home/transposenet/datasets123/{dataset}/abs_{dataset_2}_pose.csv_{scene}_test.csv --checkpoint_path /home/transposenet/out/{checkpoint}_checkpoint-{N}.pth --use_mamba {use_mamba}"
        os.system(command)
else:
     for N in range(50,650,50):
        if N == 600:
            command = f"python main.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/{dataset} /home/transposenet/datasets123/{dataset}/abs_{dataset_2}_pose.csv_{scene}_test.csv --checkpoint_path /home/transposenet/out/{checkpoint}_final.pth"
        else:
            command = f"python main.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/{dataset} /home/transposenet/datasets123/{dataset}/abs_{dataset_2}_pose.csv_{scene}_test.csv --checkpoint_path /home/transposenet/out/{checkpoint}_checkpoint-{N}.pth"
        os.system(command)   
