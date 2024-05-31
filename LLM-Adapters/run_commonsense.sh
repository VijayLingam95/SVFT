

#SVFT_PLAIN
WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=3191 finetune.py \
  --base_model 'google/gemma-2b' \
  --data_path './ft-training_set/commonsense_15k.json' \
  --output_dir './Gemma_2B_svft_CR15K/' \
  --batch_size 64 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 5e-2 \
  --cutoff_len 512\
  --val_set_size 120 \
  --adapter_name svft \
  --off_diag 0 \
  --pattern "banded" \
  --lora_target_modules "q_proj","v_proj","k_proj","o_proj","up_proj","down_proj","gate_proj"

#SVFT_Random_d=16
WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=3191 finetune.py \
  --base_model 'google/gemma-2b' \
  --data_path './ft-training_set/commonsense_15k.json' \
  --output_dir './Gemma_2B_svft_16diag_random_CR15K/' \
  --batch_size 64 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 5e-3 \
  --cutoff_len 512\
  --val_set_size 120 \
  --adapter_name svft \
  --off_diag 16 \
  --pattern "random" \
  --lora_target_modules "q_proj","v_proj","k_proj","o_proj","up_proj","down_proj","gate_proj"