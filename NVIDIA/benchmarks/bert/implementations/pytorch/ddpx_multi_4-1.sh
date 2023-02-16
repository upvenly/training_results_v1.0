#!/bin/bash
#SBATCH -J ddp
#SBATCH -p ty_zhiyuan
#SBATCH -N 1
#SBATCH -n 32
##SBATCH --ntasks-per-node=4
##SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4 
#SBATCH -o ddpx4-1.out 
#SBATCH -e ddpx4-1.err
#SBATCH --exclusive
echo "ddpx4-1.out"
source ~/.bashrc

#export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_HCA=mlx5_0
#export HSA_USERPTR_FOR_PAGED_MEM=0
#export MIOPEN_DEBUG_CONV_WINOGRAD=0
#export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
#export MIOPEN_FIND_MODE=5
#export MIOPEN_SYSTEM_DB_PATH=./tmp/pytorch-miopen-2.8
#export HIP_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
#export OMP_NUM_THREADS=1

ip=`net lookup $SLURM_JOB_NODELIST`
export addr=$(cat ip.txt)
python -m torch.distributed.launch --nproc_per_node 4 --nnodes=4 --node_rank=1 --master_addr=${addr} --master_port=25000  run_pretraining.py  --input_dir=bert_data/2048_shards_uncompressed  --output_dir=./results --seed=42 --do_train --target_mlm_accuracy=0.714  --train_mlm_accuracy_window_size=0 \
    --bert_config_path=bert_data/bert_config.json \
    --skip_checkpoint \
    --output_dir=./results \
    --fp16 \
    --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
    --gradient_accumulation_steps=1 \
    --log_freq=1 \
    --train_batch_size=6 \
    --learning_rate=4e-5 \
    --warmup_proportion=1.0 \
    --input_dir=bert_data/2048_shards_uncompressed \
    --phase2 \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --max_steps=10 \
    --eval_iter_samples=2 \
    --init_checkpoint=bert_data/model.ckpt-28252.pt \
    --use_ddp \
    --use_env \
