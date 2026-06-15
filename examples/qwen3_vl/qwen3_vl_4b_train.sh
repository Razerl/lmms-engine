#!/bin/bash
# hf download luodian/rae_siglip2 --local-dir data/rae_siglip2 # to make sure you have the pre-trained model and processor for RAE
set -e

usage() {
    cat <<'EOF'
Usage: sh examples/qwen3_vl/qwen3_vl_4b_train.sh [RUN_NAME] [WANDB_API_KEY] [NPROC_PER_NODE] [WARMUP_STEPS] [MAX_STEPS] [PACKING_LENGTH]

You can also pass values through environment variables:
  RUN_NAME=exp-name WANDB_API_KEY=... NPROC_PER_NODE=8 WARMUP_STEPS=500 MAX_STEPS=5000 PACKING_LENGTH=40000 sh examples/qwen3_vl/qwen3_vl_4b_train.sh
EOF
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

DATASET_PATH="data/oms_sft_v1_3.yaml"
PROCESSOR_NAME="/home/rzli/data/data_ssd_smb/models/QwenVL/Qwen3-VL-4B-Instruct"
MODEL_PATH="/home/rzli/data/data_ssd_smb/models/QwenVL/Qwen3-VL-4B-Instruct"
ATTN_IMPLEMENTATION="flash_attention_2"
PER_DEVICE_TRAIN_BATCH_SIZE=1
LEARNING_RATE=1.0e-05
WEIGHT_DECAY=0.0
GRADIENT_ACCUMULATION_STEPS=1
GRADIENT_CHECKPOINTING=true
NUM_TRAIN_EPOCHS=1
RUN_NAME="${1:-${RUN_NAME:-Qwen3VL-4B-ALL-SFT-test}}"
WANDB_API_KEY="${2:-${WANDB_API_KEY:-}}"
NPROC_PER_NODE="${3:-${NPROC_PER_NODE:-8}}"
WARMUP_STEPS="${4:-${WARMUP_STEPS:-500}}"
MAX_STEPS="${5:-${MAX_STEPS:-5000}}"
PACKING_LENGTH="${6:-${PACKING_LENGTH:-40000}}"
OUTPUT_DIR="/home/rzli/data/data_ssd_smb/work_dirs/lmms/${RUN_NAME}"

if [ -z "${WANDB_API_KEY}" ]; then
    echo "Error: WANDB_API_KEY is required for non-interactive wandb logging." >&2
    echo "Usage: sh examples/qwen3_vl/qwen3_vl_4b_train.sh ${RUN_NAME} <WANDB_API_KEY> [NPROC_PER_NODE] [WARMUP_STEPS] [MAX_STEPS] [PACKING_LENGTH]" >&2
    echo "Or: RUN_NAME=${RUN_NAME} WANDB_API_KEY=<key> sh examples/qwen3_vl/qwen3_vl_4b_train.sh" >&2
    exit 1
fi

for var_name in NPROC_PER_NODE WARMUP_STEPS MAX_STEPS PACKING_LENGTH; do
    eval "var_value=\${${var_name}}"
    case "${var_value}" in
        ''|*[!0-9]*|0)
            echo "Error: ${var_name} must be a positive integer, got '${var_value}'." >&2
            exit 1
            ;;
    esac
done

# export CUDA_VISIBLE_DEVICES="1,2"
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY
export WANDB_MODE=online
export WANDB_PROJECT="Qwen3VL"
export WANDB_ENTITY="lrzlrz1995-personal"
export WANDB_SILENT=true

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="8005" \
    -m lmms_engine.launch.cli \
    trainer_type=fsdp2_trainer \
    dataset_config.dataset_path=${DATASET_PATH} \
    dataset_config.dataset_format=yaml \
    dataset_config.processor_config.processor_name=${PROCESSOR_NAME} \
    dataset_config.dataset_type=qwen3_vl_iterable \
    dataset_config.processor_config.processor_type=qwen3_vl \
    +dataset_config.processor_config.extra_kwargs.image_max_pixels=1572864 \
    +dataset_config.processor_config.extra_kwargs.image_min_pixels=65536 \
    dataset_config.packing=true \
    dataset_config.packing_strategy=balanced \
    dataset_config.packing_length=${PACKING_LENGTH} \
    +dataset_config.extra_kwargs.packing_kwargs.num_buckets=2 \
    dataset_config.filter_overlong=true \
    dataset_config.video_backend=qwen_vl_utils \
    dataset_config.video_sampling_strategy=fps \
    dataset_config.video_max_pixels=50176 \
    dataset_config.video_max_frames=512 \
    model_config.load_from_pretrained_path=${MODEL_PATH} \
    model_config.attn_implementation=${ATTN_IMPLEMENTATION} \
    trainer_args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    trainer_args.learning_rate=${LEARNING_RATE} \
    trainer_args.weight_decay=${WEIGHT_DECAY} \
    trainer_args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer_args.gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
    trainer_args.num_train_epochs=${NUM_TRAIN_EPOCHS} \
    trainer_args.warmup_steps=${WARMUP_STEPS} \
    trainer_args.run_name=${RUN_NAME} \
    trainer_args.output_dir=${OUTPUT_DIR} \
    trainer_args.fsdp2=true \
    trainer_args.max_steps=${MAX_STEPS} \
    trainer_args.save_steps=500 \
    trainer_args.save_total_limit=5 \
    trainer_args.fsdp_config.transformer_layer_cls_to_wrap=["Qwen3VLTextDecoderLayer"] \
    trainer_args.fsdp_config.reshard_after_forward=false \
    trainer_args.sp_ulysses_degree=1 \
    trainer_args.use_liger_kernel=true \
    trainer_args.use_rmpad=true \
    trainer_args.dataloader_num_workers=0 \
    trainer_args.dataloader_prefetch_factor=null \
    trainer_args.print_batch_input_steps=5 \
    trainer_args.bf16=true \
    trainer_args.lr_scheduler_type=cosine \
    trainer_args.logging_steps=1 \
    trainer_args.group_by_length=false \
    trainer_args.report_to=wandb 
