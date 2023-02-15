#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=8 \
    run_pretraining.py \
    --target_mlm_accuracy=0.714 \
    --train_mlm_accuracy_window_size=0 \
    --seed=42 \
    --do_train \
    --bert_config_path=bert_data/bert_config.json \
    --skip_checkpoint \
    --output_dir=/results \
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


