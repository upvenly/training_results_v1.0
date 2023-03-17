#!/bin/bash
# NODES=2 RANK=0 NPROC_PER_NODE=8 MASTER_ADDR=192.168.5.2 MASTER_PORT=23456 sh run_test.sh

data_path=/data/bert_data

python -m torch.distributed.launch \
     --nnodes="${NODES}" \
     --node_rank="${RANK}" \
     --nproc_per_node="${NPROC_PER_NODE}" \
     --master_addr="${MASTER_ADDR}" \
     --master_port="${MASTER_PORT}" \
    run_pretraining.py \
    --target_mlm_accuracy=0.714 \
    --train_mlm_accuracy_window_size=0 \
    --seed=42 \
    --do_train \
    --bert_config_path=$data_path/bert_config.json \
    --skip_checkpoint \
    --output_dir=/results \
    --fp16 \
    --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
    --gradient_accumulation_steps=1 \
    --log_freq=1 \
    --train_batch_size=4 \
    --learning_rate=4e-5 \
    --warmup_proportion=1.0 \
    --input_dir=$data_path/2048_shards_uncompressed \
    --phase2 \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --max_steps=1500 \
    --eval_iter_samples=2 \
    --init_checkpoint=$data_path/model.ckpt-28252.pt \
    --use_ddp \


