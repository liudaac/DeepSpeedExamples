#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
export NCCL_SOCKET_IFNAME=enp5s0
export NCCL_DEBUG=INFO

OUTPUT=$1
ZERO_STAGE=$2
PORT=$3
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
if [ "$PORT" == "" ]; then
    PORT=12345
fi
mkdir -p $OUTPUT

deepspeed --num_nodes 2 --num_gpus 1 --master_port $PORT --hostfile hostfile --master_addr 10.234.128.136 main.py \
   --model_name_or_path facebook/opt-350m \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0.1 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 128 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
