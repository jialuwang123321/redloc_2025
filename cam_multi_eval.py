import os

# 加速
# source /etc/network_turbo
dataset_0 = "Cambridge"
dataset = "CambridgeLandmarks"
dataset_2 = "cambridge"
scene = 'KingsCollege'
checkpoint = 'run_16_07_24_06_44'

for N in range(600,650,50):  
    if N == 600:                                                                                                                                                                                                                                                                                                                 
        command = f"python main.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/{dataset_0} /home/transposenet/datasets123/{dataset}/abs_{dataset_2}_pose_sorted.csv_{scene}_test.csv --checkpoint_path /home/transposenet/out/{checkpoint}_final.pth --use_mamba True"
        os.system(command)
    else:
        command = f"python main.py transposenet test ./models/backbones/efficient-net-b0.pth /home/transposenet/data/{dataset_0} /home/transposenet/datasets123/{dataset}/abs_{dataset_2}_pose_sorted.csv_{scene}_test.csv --checkpoint_path /home/transposenet/out/{checkpoint}_checkpoint-{N}.pth --use_mamba True"
        os.system(command)


