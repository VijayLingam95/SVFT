import random
from typing import Literal, List
from collections import Counter
from dataclasses import dataclass, field
from functools import partial

import torch
import evaluate
import numpy as np
from torch import optim
from datasets import load_dataset
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoImageProcessor,
    AutoModelForImageClassification,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from peft import get_peft_model, VeraConfig, BOFTConfig, LoraConfig

import sys
sys.path.append("../")
from svft.svft_layers import *


##########################
# Metrics
##########################

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


##########################
# Utils
##########################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_seed(SEED=0):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    transformers.set_seed(SEED)


def get_trainable_params_dict(model):
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    clf_trainable_p = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and "classifier" in n
    )
    other_p = trainable_p - clf_trainable_p
    return {
        "total_p": total_p,
        "trainable_p": trainable_p,
        "clf_trainable_p": clf_trainable_p,
        "other_p": other_p,
    }


def print_trainable_parameters(model):
    params_dict = get_trainable_params_dict(model)
    total_p = params_dict["total_p"]
    trainable_p = params_dict["trainable_p"]
    clf_trainable_p = params_dict["clf_trainable_p"]
    other_p = params_dict["other_p"]
    print(
        f"Total params: {total_p}  | Trainable params: {trainable_p}  |  Trainable%: {trainable_p/total_p*100:.2f}%"
    )
    print(
        f"Clf Trainable params: {clf_trainable_p}  |  Clf Trainable%: {clf_trainable_p/total_p*100:.2f}%"
    )
    print(
        f"FT Trainable params: {other_p}  |  FT Trainable%: {other_p/total_p*100:.2f}%"
    )
    print()


##########################
# Dataset Utilities
##########################

label_key = "label"
image_path_key = "image"


def collate_fn(examples):
    pixel_values = torch.stack(
        [torch.Tensor(example["pixel_values"]) for example in examples]
    )
    labels = torch.tensor([example[label_key] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def preprocess(example_batch, transform_fn):
    example_batch["pixel_values"] = [
        transform_fn(image.convert("RGB")) for image in example_batch[image_path_key]
    ]
    return example_batch


def get_transforms(image_processor):
    if "height" in image_processor.size:
        return Compose(
            [
                Resize((image_processor.size["height"], image_processor.size["width"])),
                ToTensor(),
                Normalize(
                    mean=image_processor.image_mean, std=image_processor.image_std
                ),
            ]
        )
    elif "height" in image_processor.crop_size:
        return Compose(
            [
                Resize(
                    (
                        image_processor.crop_size["height"],
                        image_processor.crop_size["width"],
                    )
                ),
                ToTensor(),
                Normalize(
                    mean=image_processor.image_mean, std=image_processor.image_std
                ),
            ]
        )
    else:
        raise ValueError("Unknown image processor")


def get_ids_and_labels_from_dataset(dataset):
    labels = set(dataset[label_key])
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    return label2id, id2label


def sampled_balanced_train_val(dataset, num_train_per_label=10, num_val_per_label=2):
    label_counter = Counter(dataset[label_key])

    inst_train = num_train_per_label
    inst_val = num_val_per_label

    assert min(label_counter.values()) >= inst_train + inst_val
    label_counter = Counter()
    train_ids = []
    val_ids = []

    labels = list(enumerate(dataset[label_key]))
    random.shuffle(labels)

    for i, l in labels:
        if label_counter[l] < inst_train:
            train_ids.append(i)
        elif label_counter[l] < inst_train + inst_val:
            val_ids.append(i)
        else:
            continue
        label_counter[l] += 1

    return dataset.select(train_ids), dataset.select(val_ids)


DATASET_NAME_TO_URL = {
    "cifar100": "cifar100",
    "food101": "ethz/food101",
    "flowers102": "dpdl-benchmark/oxford_flowers102",
    "resisc45": "timm/resisc45",
}


def get_dataset(dataset_name):
    if dataset_name == "cifar100":
        dataset_url = DATASET_NAME_TO_URL[dataset_name]

        dataset = load_dataset(dataset_url, split="train")
        dataset = dataset.rename_column("fine_label", label_key)
        dataset = dataset.rename_column("img", image_path_key)

        dataset_test = load_dataset(dataset_url, split="test")
        dataset_test = dataset_test.rename_column("fine_label", label_key)
        dataset_test = dataset_test.rename_column("img", image_path_key)
        dataset_train, dataset_val = sampled_balanced_train_val(dataset)

        return dataset_train, dataset_val, dataset_test

    elif dataset_name == "food101":
        dataset_url = DATASET_NAME_TO_URL[dataset_name]
        dataset = load_dataset(dataset_url, split="train")
        dataset_test = load_dataset(dataset_url, split="validation")
        dataset_train, dataset_val = sampled_balanced_train_val(dataset)
        return dataset_train, dataset_val, dataset_test

    elif dataset_name in {"flowers102", "resisc45"}:
        dataset_url = DATASET_NAME_TO_URL[dataset_name]
        dataset_train = load_dataset(dataset_url, split="train")
        dataset_val = load_dataset(dataset_url, split="validation")
        dataset_test = load_dataset(dataset_url, split="test")
        dataset_train, _ = sampled_balanced_train_val(
            dataset_train, num_train_per_label=10, num_val_per_label=0
        )
        _, dataset_val = sampled_balanced_train_val(
            dataset_val, num_train_per_label=0, num_val_per_label=2
        )
        return dataset_train, dataset_val, dataset_test

    else:
        raise ValueError("Unknown dataset name")


##########################
# Finetuning Config
##########################


MODEL_NAME_TO_URL = {
    "dino-v2-large": "facebook/dinov2-large",
    "vit-base": "google/vit-base-patch16-224-in21k",
    "vit-large": "google/vit-large-patch16-224-in21k",
}


def get_target_modules(model_name, finetuning_method):
    if model_name == "dino-v2-large":
        if finetuning_method in {"vera", "svft"}:
            return [
                "query",
                "key",
            ]
        else:
            return "all-linear"
    elif model_name in {"vit-base", "vit-large"}:
        if finetuning_method == "head":
            return []
        else:
            return [
                "query",
                "value",
            ]
    else:
        raise ValueError("Unknown model name")


def get_classifier_modules(model_name):
    if model_name in {"dino-v2-large", "vit-base", "vit-large"}:
        return [
            "classifier",
        ]
    else:
        raise ValueError("Unknown model name")


@dataclass
class ScriptArguments:
    results_json: str = field(
        default="results.json", metadata={"help": "Results json file"}
    )
    model_name: Literal["dino-v2-large", "vit-base", "vit-large"] = field(
        default="vit-base", metadata={"help": "Model name"}
    )
    dataset_name: Literal[
        "cifar100",
        "food101",
        "flowers102",
        "resisc45",
    ] = field(default="cifar100", metadata={"help": "Dataset name"})
    finetuning_method: Literal[
        "vera", "boft", "lora", "dora", "svft", "head", "full"
    ] = field(default="head", metadata={"help": "Finetuning method"})
    clf_learning_rate: float = field(
        default=1e-3, metadata={"help": "Classifier learning rate"}
    )
    other_learning_rate: float = field(
        default=1e-4, metadata={"help": "Other learning rate"}
    )

    ## BOFT
    boft_block_size: int = field(default=0, metadata={"help": "BOFT block size (m)"})
    boft_n_butterfly_factor: int = field(
        default=0, metadata={"help": "BOFT n butterfly factor (b)"}
    )

    ## VeRA
    vera_rank: int = field(default=0, metadata={"help": "Vera rank"})

    ## LoRA and DoRA
    lora_rank: int = field(default=0, metadata={"help": "Lora rank"})

    ## SVFT rank
    svft_rank: int = field(default=0, metadata={"help": "SVFT rank"})

    ## Target Modules
    target_modules: List[str] = field(
        default_factory=list,
        metadata={"help": "Target modules for finetuning"},
    )


def main():
    import json
    import wandb
    from pprint import pprint

    wandb.init(mode="disabled")

    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    reset_seed(training_args.seed)

    ## Load dataset
    dataset_train, dataset_val, dataset_test = get_dataset(script_args.dataset_name)
    label2id, id2label = get_ids_and_labels_from_dataset(dataset_train)

    # Set image transforms
    model_name = script_args.model_name
    model_url = MODEL_NAME_TO_URL[model_name]
    image_processor = AutoImageProcessor.from_pretrained(model_url)
    transform_fn = get_transforms(image_processor)

    dataset_train.set_transform(lambda x: preprocess(x, transform_fn))
    dataset_val.set_transform(lambda x: preprocess(x, transform_fn))
    dataset_test.set_transform(lambda x: preprocess(x, transform_fn))

    # Load model
    model = AutoModelForImageClassification.from_pretrained(
        model_url,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    ).to(device)
    print_trainable_parameters(model)

    # Get Target Modules
    if not script_args.target_modules:
        script_args.target_modules = get_target_modules(
            model_name, script_args.finetuning_method
        )

    # Set fine-tuning config
    if script_args.finetuning_method == "vera":
        config = VeraConfig(
            r=script_args.vera_rank,
            target_modules=script_args.target_modules,
            modules_to_save=get_classifier_modules(model_name),
            vera_dropout=0.1,
            bias="none",
        )
    elif script_args.finetuning_method == "boft":
        config = BOFTConfig(
            boft_block_size=script_args.boft_block_size,
            boft_n_butterfly_factor=script_args.boft_n_butterfly_factor,
            target_modules=script_args.target_modules,
            modules_to_save=get_classifier_modules(model_name),
            boft_dropout=0.1,
            bias="boft_only",
        )
    elif script_args.finetuning_method == "lora":
        config = LoraConfig(
            r=script_args.lora_rank,
            target_modules=script_args.target_modules,
            modules_to_save=get_classifier_modules(model_name),
            bias="none",
            lora_dropout=0.1,
        )
    elif script_args.finetuning_method == "dora":
        config = LoraConfig(
            r=script_args.lora_rank,
            target_modules=script_args.target_modules,
            modules_to_save=get_classifier_modules(model_name),
            bias="none",
            lora_dropout=0.1,
            use_dora=True,
        )
    elif script_args.finetuning_method == "head":
        classifier_modules = get_classifier_modules(model_name)
        for n, p in model.named_parameters():
            if all(c not in n for c in classifier_modules):
                p.requires_grad = False
    elif script_args.finetuning_method in ["svft", "full"]:
        pass
    else:
        raise ValueError("Unknown finetuning method")

    if script_args.finetuning_method == "svft":
        peft_model = model
        modules_to_save_list = get_target_modules_list(
            peft_model, get_classifier_modules(model_name)
        )
        freeze_model(peft_model, modules_to_save_list)
        target_modules_list = get_target_modules_list(
            peft_model, script_args.target_modules
        )
        create_and_replace_modules(peft_model, target_modules_list, partial(LinearWithSVFT, off_diag=script_args.svft_rank))
    elif script_args.finetuning_method in ["head", "full"]:
        peft_model = model
    else:
        peft_model = get_peft_model(model, config)

    print_trainable_parameters(peft_model)
    params_dict = get_trainable_params_dict(peft_model)

    # Setup Trainer
    args = TrainingArguments(**training_args.to_dict())
    classifier_group = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad
        and any(cls_name in n for cls_name in get_classifier_modules(model_name))
    ]
    other_parameters_group = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad
        and all(cls_name not in n for cls_name in get_classifier_modules(model_name))
    ]
    optimizer = optim.AdamW(
        [
            {
                "params": classifier_group,
                "lr": script_args.clf_learning_rate,
            },
            {
                "params": other_parameters_group,
                "lr": script_args.other_learning_rate,
            },
        ],
        lr=script_args.other_learning_rate,
        weight_decay=training_args.weight_decay,
    )

    num_train_steps = (
        len(dataset_train)
        // training_args.per_device_train_batch_size
        * training_args.num_train_epochs
    )

    if training_args.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * training_args.warmup_ratio),
            num_training_steps=num_train_steps,
        )
    elif training_args.lr_scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * training_args.warmup_ratio),
            num_training_steps=num_train_steps,
        )

    trainer = Trainer(
        peft_model,
        args,
        optimizers=(optimizer, scheduler),
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train()
    with open(
        training_args.output_dir + f"final_train_results_{training_args.seed}.json", "w"
    ) as f:
        json.dump(train_results, f, indent=4)

    if script_args.finetuning_method == "svft":
        create_and_replace_modules(peft_model, target_modules_list, reset_from_svft)
    elif script_args.finetuning_method not in {"svft", "head", "full"}:
        peft_model = peft_model.merge_and_unload()

    eval_results = trainer.evaluate(dataset_test)
    print(eval_results)

    for key in script_args.__dataclass_fields__:
        value = getattr(script_args, key)
        eval_results[key] = value

    for key in training_args.__dataclass_fields__:
        if "accelerator" in key:
            continue
        value = getattr(training_args, key)
        eval_results[key] = value

    eval_results.update(params_dict)
    pprint(eval_results, indent=4)

    with open(
        training_args.output_dir + f"final_eval_results_{training_args.seed}.json", "w"
    ) as f:
        json.dump(eval_results, f, indent=4)

    # Save to results.json
    try:
        with open(script_args.results_json, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

    results.append(eval_results)
    with open(script_args.results_json, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
