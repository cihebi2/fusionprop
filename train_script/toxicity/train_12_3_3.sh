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

python train_12_2.py \
    --name "ESMC+ESM2" \
    --train_pos_csv csm_toxin_0.7.csv \
    --train_neg_csv csm_notoxin_0.7.csv \
    --test_pos_csv filtered_toxin_0.7.csv \
    --test_neg_csv filtered_notoxin_0.7.csv \
    --save_dir "./train_12_3_3" \
    --batch_size 16 \
    --epochs 30 \
    --lr 1e-4 \
    --weight_decay 5e-05 \
    --hidden_dim 768 \
    --dropout 0.5 \
    --max_seq_len 1024 \
    --use_esmc \
    --esmc_path "esmc_600m" \
    --use_esm2 \
    --splm_config "./configs/representation_config.yaml" \
    --splm_checkpoint "/HOME/scz0brz/run/AA_solubility/model/checkpoint_0520000.pth" \
    --feature_device 0 \
    --training_device 1 \
    --feature_cache_size 1000 \
    --num_workers 0 \
    --negative_sampling_ratio 20 \
    > train_12_3_3.log &

wait
echo "所有脚本运行完成。"
