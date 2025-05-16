#!/bin/bash
#SBATCH --gpus=1
#SBATCH -x g[0014-0016,0023,0025-0032,0034-0035,0039,0042-0043,0045,0047,0053,0060,0065,0159,0162,0164,0166,0170,0171,0172]
module purge
module load anaconda/2020.11 gcc/9.3
module load cuda/12.1
module load cudnn/8.9.7_cuda12.x
source activate dev240430
# export PATH=/HOME/scz0brz/run/anaconda3/bin:$PATH
export PYTHONUNBUFFERED=1
which python
# ldd /data/apps/cudnn/cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib/libcudnn_cnn_train.so.8
# strings /data/apps/cudnn/cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib/libcudnn_cnn_infer.so |grep libcudnn_cnn_infer.so.8
env

python -u extract_esmc_1.py > extract_esmc_1.log &



wait
echo "所有脚本运行完成。"
