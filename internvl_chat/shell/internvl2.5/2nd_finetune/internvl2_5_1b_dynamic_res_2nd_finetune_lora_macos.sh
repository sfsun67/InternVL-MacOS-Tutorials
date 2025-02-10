set -x

BATCH_SIZE=${BATCH_SIZE:-16}


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/internvl_chat_v2_5/internvl2_5_1b_dynamic_res_2nd_finetune_lora_macos'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# batch size per gpu: 4
# gradient accumulation steps: 2
# epoch: 1
python \
  internvl/train/internvl_chat_finetune_macos.py \
  --model_name_or_path "/Users/sf/codeAlgorithms/VLLM/cache/modelscope/hub/OpenGVLab/InternVL2_5-1B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "shell/data/internvl_2_5_fineturn_test_ai2d_train_12k.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 1 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 2 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 False \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "wandb" \
  2>&1 | tee -a "${OUTPUT_DIR}/dbg_training_log.txt"
