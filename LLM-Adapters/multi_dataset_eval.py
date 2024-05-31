from concurrent.futures import ProcessPoolExecutor
import queue
import subprocess
import os

def evaluate(dataset, gpu):
    print('*******dataset:', dataset)
    model_name = "Pythia_2B_143000_SVFT_CR15K"
    save_dir= "results/" + model_name
    
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass

    save_path = os.path.join(save_dir, dataset + ".txt")
    command = f"CUDA_VISIBLE_DEVICES={gpu} python commonsense_evaluate_latest.py \
               --model LLaMA-7B \
               --adapter LoRA \
               --dataset {dataset} \
               --base_model './{model_name}' \
               --batch_size 1| tee -a {save_path}"

    result = subprocess.run(command, shell=True, text=True, capture_output=False)
    print(f"Evaluation results for dataset {dataset} on GPU {gpu}:\n{result.stdout}")
    return gpu


datasets = ["boolq", "social_i_qa", "piqa", "ARC-Easy", "ARC-Challenge", "winogrande", "openbookqa", "hellaswag"]

gpus = [0, 0, 0, 0]
tasks_queue = queue.Queue()
gpu_queue = queue.Queue()

for gpu in gpus:
    gpu_queue.put(gpu)
for task in datasets:
    tasks_queue.put(task)

num_processes = min(len(datasets), len(gpus))  # number of processes to run in parallel

with ProcessPoolExecutor(max_workers=num_processes) as executor:
    futures = [executor.submit(evaluate, tasks_queue.get(), gpu_queue.get()) for i in range(num_processes)]
    for future in futures:
        gpu_id = future.result()
        gpu_queue.put(gpu_id)
        if tasks_queue.qsize() > 0:
            futures.append(executor.submit(evaluate, tasks_queue.get(), gpu_queue.get()))
