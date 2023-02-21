#!/bin/bash
#SBATCH -J Pytorch-1.10
#SBATCH -p ty_zhiyuan
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=dcu:2
#SBATCH -o %j.out
#SBATCH -e %j.err
module switch compiler/dtk/22.10
conda activate torch-python37
python -m torch.distributed.launch --nproc_per_node=2 \
    run_pretraining.py  --input_dir=bert_data/2048_shards_uncompressed  --output_dir=./results --seed=42 --do_train --target_mlm_accuracy=0.714  --train_mlm_accuracy_window_size=0 \
    --bert_config_path=bert_data/bert_config.json \
    --skip_checkpoint \
    --output_dir=./results \
    --fp16 \
    --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
    --gradient_accumulation_steps=1 \
    --log_freq=1 \
    --train_batch_size=4 \
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
    


