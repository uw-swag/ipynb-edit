import argparse
from peft import LoraConfig, PeftModel, get_peft_model
from urf.config import TrainLoraConfig
from urf.common import build_tokenizer, build_datasets, build_trainer
import json
from transformers.trainer_utils import get_last_checkpoint
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    TorchAoConfig,
)
from torch.optim import AdamW
import random
import numpy as np
import seutil as su
from pathlib import Path
from typing import List, Tuple
from functools import partial
import dataclasses

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
model_dir = '1.3b_file'
cycle = '3'
print(f'{model_dir=}/{cycle=}')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# parser = argparse.ArgumentParser(prog="LoRA Model Trainer")
# parser.add_argument("base_dir")  
# base_dir = parser.parse_args().base_dir
# print(base_dir)
base_dir = '/home/b27jin/ipynb-edit-internal/'
with open(base_dir + "model/lora_config.json", mode="r") as f:
    config_dict = json.load(f)
config = TrainLoraConfig.model_validate(config_dict)

@dataclasses.dataclass
class RawData:
    repo: str
    commit: str
    file: str
    message: str
    old: List[str]
    new: List[str]
    cell_diff: List[Tuple[str, int, int, int, int]]
    line_diff: List[List[Tuple[str, int, int, int, int]]]

    def concat_old_cells(self) -> str:
        return "\n".join(self.old)

    def concat_new_cells(self) -> str:
        return "\n".join(self.new)

print("Start Loading")
data = su.io.load(Path.cwd() / "model/split/commits.jsonl", clz=RawData)
train_dataset, test_dataset, val_dataset = [], [], []
with open("model/split/train_index.txt", "r") as file:
    line = file.readline().strip()
    train = list(map(int, line.split(",")))
    train_dataset = [data[i] for i in train]

with open("model/split/test_index.txt", "r") as file:
    line = file.readline().strip()
    test = list(map(int, line.split(",")))
    test_dataset = [data[i] for i in test]

# with open("model/split/val_index.txt", "r") as file:
#     line = file.readline().strip()
#     val = list(map(int, line.split(",")))
#     val_dataset = [data[i] for i in val]
print("Complete Loading")

def build_model(model_config, tokenizer):
    print(model_config.path)
    # quant_config = TorchAoConfig(quant_type="int8_weight_only")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.path,
        torch_dtype=torch.bfloat16,
        # quantization_config=quant_config,
        attn_implementation="flash_attention_2",
        device_map="cuda"
    )

    if model.config.eos_token_id != tokenizer.eos_token_id or model.config.pad_token_id != tokenizer.pad_token_id:
        # Initialize pad token embed to be the same as eos token embed (as in pretraining)
        print(f"Now using EOS token {tokenizer.eos_token}: {tokenizer.eos_token_id}")
        print(f"Now using PAD token {tokenizer.pad_token}: {tokenizer.pad_token_id}")
        print("Pad token's embeddings are copied from EOS token's.")
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.model.embed_tokens.weight.data[tokenizer.pad_token_id] = model.model.embed_tokens.weight.data[tokenizer.eos_token_id]
    model.config.use_cache = False # Not useful for training

    if model_config.lora_path is not None:
        print("Loading existing LoRA")
        model = PeftModel.from_pretrained(model, model_config.lora_path, is_trainable=True)
    else:
        print("Creating new LoRA")
        lora_config = LoraConfig(
            r=model_config.rank, lora_alpha=model_config.alpha,
            bias="none", use_rslora=True, init_lora_weights="pissa",
            target_modules=["gate_proj", "up_proj", "down_proj"],
            # modules_to_save=model_config.modules_to_save,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    print("LoRA Created")
    optimizer = AdamW(model.parameters(), lr=model_config.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    return model, optimizer

def save_model(trainer):
    output_dir = trainer.args.output_dir
    base_model_save_dir = base_dir + f"model/{model_dir}/base_{cycle}"
    adapter_model_save_dir = base_dir + f"model/{model_dir}/adapter_{cycle}"
    final_save_dir = base_dir + f"model/{model_dir}/final_{cycle}"
    
    # Accelerator
    # if trainer.is_fsdp_enabled:
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    model = trainer.model
    model.config.use_cache = True
    for config in model.peft_config.values(): # Trick lora into not doing svd again
        config.init_lora_weights = False
    model.save_pretrained(adapter_model_save_dir)
    model = model.unload()
    model.save_pretrained(base_model_save_dir)

    # quant_config = TorchAoConfig(quant_type="int8_weight_only")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_save_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # quantization_config=quant_config,
        device_map="cuda"
    )
    model = PeftModel.from_pretrained(model, adapter_model_save_dir)
    model = model.merge_and_unload()
    model.save_pretrained(final_save_dir)

    trainer.tokenizer.save_pretrained(final_save_dir)


tokenizer = build_tokenizer(config.tokenizer)    
model, optimizer = build_model(config.model, tokenizer)

# from accelerate import Accelerator
# accelerator = Accelerator()
# model, tokenizer, optimizer = accelerator.prepare(model, tokenizer,optimizer)

# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
# if not dist.is_initialized():
#     dist.init_process_group(backend="nccl")

# device_id = int(os.environ["LOCAL_RANK"])
# torch.cuda.set_device(device_id)
# model.to(device_id)
# model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

def generate_code_prompt_whole_file(code_cells):

    result = ""
    for i in range (len(code_cells)):
        result += "<Cell_" + str(i) + ">\n"
        result += code_cells[i]
        result += "\n<\\Cell_" + str(i) + ">\n"

    return result

def generate_code_prompt_cell_only_old(old_cells, cell_diff):

    result = ""
    for cd in cell_diff:
        for i in range (cd[1], cd[2]):
            result += old_cells[i] + "\n"
    return result

def generate_code_prompt_cell_only_new(new_cells, cell_diff):

    result = ""
    for cd in cell_diff:
        for i in range (cd[3], cd[4]):
            result += new_cells[i] + "\n"
    return result

def prepare_file_examples(train_dataset):
        
    examples = []
    for i in range(len(train_dataset)):
        result = ""
        result += "### Instruction\n"
        result += "[\n"
        result += "Commit Message:" + "\" " + train_dataset[i].message + "\"\n\n"
        result += "Original Code Cells:\n" 
        result += "\'\'\'\n"
        result += generate_code_prompt_whole_file(train_dataset[i].old) + "\n"
        result += "\'\'\'\n"
        result += "]\n\n"
        result += "### Response:\n"
        result += generate_code_prompt_whole_file(train_dataset[i].new) + "\n"
        result += "<|EOT|>\n\n"
        examples.append({"text": result})
    return examples

def prepare_cell_examples(train_dataset):
        
    examples = []
    for i in range(len(train_dataset)):
        result = ""
        result += "### Instruction\n"
        result += "[\n"
        result += "Commit Message:" + "\" " + train_dataset[i].message + "\"\n\n"
        result += "Original Code Cells:\n" 
        result += "\'\'\'\n"
        result += generate_code_prompt_cell_only_old(train_dataset[i].old, train_dataset[i].cell_diff) + "\n"
        result += "\'\'\'\n"
        result += "]\n\n"
        result += "### Response:\n"
        result += generate_code_prompt_cell_only_new(train_dataset[i].new, train_dataset[i].cell_diff) + "\n"
        result += "<|EOT|>\n\n"
        examples.append({"text": result})
    return examples

from datasets import Dataset
def build_datasets(train_dataset, test_dataset, tokenizer, config):
    if 'file' in model_dir:
        print('prompts for files')
        train_dataset_hf = Dataset.from_list(prepare_file_examples(train_dataset)).shuffle(seed=config.datasets.shuffle_seed)
        eval_dataset_hf = Dataset.from_list(prepare_file_examples(test_dataset))
    else:
        print('prompts for cells')
        train_dataset_hf = Dataset.from_list(prepare_cell_examples(train_dataset)).shuffle(seed=config.datasets.shuffle_seed)
        eval_dataset_hf = Dataset.from_list(prepare_cell_examples(test_dataset))

    def _tokenize_function(prompts):

        tokenized_examples = tokenizer(prompts["text"], add_special_tokens=False , truncation=True, return_tensors="pt", padding="max_length", max_length=config.datasets.truncation_length)
        tokenized_examples["labels"] = tokenized_examples["input_ids"].clone()
        return tokenized_examples
    
    tokenized_train_dataset = train_dataset_hf.map(_tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset_hf.map(_tokenize_function, batched=True)
    return tokenized_train_dataset, tokenized_eval_dataset

print("Creating Dataset")
tokenized_train_dataset, tokenized_eval_dataset = build_datasets(train_dataset, test_dataset, tokenizer, config)
print("Dataset created")
print("Build trainer")
trainer = build_trainer(base_dir+f'model/{model_dir}/checkpoint_{cycle}', config.training, tokenizer, tokenized_train_dataset, tokenized_eval_dataset, model, optimizer, compile=False)

os.environ["TOKENIZERS_PARALLELISM"]="false"
# print("Start training")
# torch.cuda.empty_cache()
# res = trainer.train(resume_from_checkpoint=get_last_checkpoint(base_dir+f'model/{model_dir}/checkpoint_{cycle}/'))
# with open(base_dir+f"model/{model_dir}/train_{cycle}.json", "w") as f:
#     json.dump(res , f)

# print("Start evaluating")
# with torch.no_grad():
#     res = trainer.evaluate()
# with open(base_dir+f"model/{model_dir}/eval_{cycle}.json", "w") as f:
#     json.dump(res , f) 

save_model(trainer)
# CUDA_VISIBLE_DEVICES=6 /home/b27jin/bin/python /home/b27jin/ipynb-edit-internal/model/train_baseline.py
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node 7 /home/b27jin/ipynb-edit-internal/model/train_baseline.py
# deepspeed --include localhost:0,1,2,4,5,6,7 /home/b27jin/ipynb-edit-internal/model/train_baseline.py
