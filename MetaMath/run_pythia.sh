export MODEL_PATH='EleutherAI/pythia-2.8B'
export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"


#SVFT_PLAIN
export SAVE_PATH='./Pythia_2B_metamath40k_svft16diag_random_rev39000'
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=1 --use_env train_pythia.py \
    --model_name_or_path $MODEL_PATH \
    --data_path "../../MetaMath/data/train/MetaMathQA-40K.json" \
    --data_length 10000000 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 5e-3\
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --pattern "random" \
    --off_diag 16 \
    --target_modules dense_4h_to_h dense_h_to_4h query_key_value dense \
    --adapter_name "svft" \
    --revision step39000

export SAVE_PATH='./Pythia_2B_metamath40k_svft16diag_random_rev143000'
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=1 --use_env train_pythia.py \
    --model_name_or_path $MODEL_PATH \
    --data_path "./data/train/MetaMathQA-40K.json" \
    --data_length 10000000 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 5e-3\
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --pattern "random" \
    --off_diag 16 \
    --target_modules dense_4h_to_h dense_h_to_4h query_key_value dense \
    --adapter_name "svft" \
    --revision step143000



