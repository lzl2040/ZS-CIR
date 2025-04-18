#!/bin/bash

args=()

BASE_MODEL="/mnt/input_zuo/ZS-CIR/models/llava-llama-3-8b"
BIT=4
R=64
ALPHA=16
BATCH_SIZE=768
MICRO_BATCH_SIZE=96 # 尽量调大，直到显存占满
EPOCH=2
LR=4e-4

GPUS=4 # 8
NUM_NODES=1 # 4

# 定义不同的模板
# "After_thinking_step_by_step_this_sentence_*sent_0*\nSummary_above_sentence_in_one_word:"  # cot

TEMPLATES=(
    "*sent_0*\nSummary_above_sentence_in_one_word:"  # org
    "The_essence_of_a_sentence_is_often_captured_by_its_main_subjects_and_actions_while_descriptive_terms_provide_additional_but_less_central_details_With_this_in_mind_this_sentence_*sent_0*\nSummary_above_sentence_in_one_word:"  # ke
)

# 对应的 RUN 名称
# RUNS=("org" "cot" "ke")
RUNS=("org" "ke-2")

wandb online

# 遍历 TEMPLATES 和 RUNS
for i in "${!TEMPLATES[@]}"; do
    TEMPLATE="${TEMPLATES[$i]}"
    RUN="/mnt/output_zuo/ZS-CIR/e5v-8b-4bit-${RUNS[$i]}"  # 使用预定义的 RUN 名称

    echo "Running experiment with TEMPLATE index: $i"
    echo "RUN: $RUN"
    echo "TEMPLATE: $TEMPLATE"
    echo "MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE, BATCH_SIZE: $BATCH_SIZE"

    python ft_llm.py \
        --base_model $BASE_MODEL \
        --data_path 'data/nli_for_simcse.csv' \
        --batch_size $BATCH_SIZE \
        --micro_batch_size $MICRO_BATCH_SIZE  \
        --num_epochs $EPOCH \
        --learning_rate $LR \
        --cutoff_len 32 \
        --lora_r $R \
        --lora_alpha $ALPHA \
        --lora_dropout 0.05 \
        --output_dir $RUN --is_sentemb \
        --mask_embedding_sentence_template "$TEMPLATE" --use_neg_sentence --save_steps 50 \
        --deepspeed ds.config \
        --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
        --logging_steps 1 --grad_checkpoint \
        --load_kbit $BIT \
        "${args[@]}"

done
