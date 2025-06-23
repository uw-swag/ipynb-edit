import os
import dataclasses
import random
import seutil as su
from tqdm import tqdm
from typing import List, Tuple
from pathlib import Path
from pprint import pprint
from collections import defaultdict

import nltk
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AdamW,
    TorchAoConfig,
    Trainer,
    TrainingArguments,
)
# from accelerate import Accelerator
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
# if not dist.is_initialized():
#     dist.init_process_group(backend="nccl")

# accelerator = Accelerator()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NUMEXPR_MAX_THREADS"] = "32"

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

def generate_few_shot_prompt_whole_file(training, fs):
    few_shot_list = random.sample(training, fs)
    result = ""
    for i in range (len(few_shot_list)):
        result += "### Instruction\n"
        result += "[\n"
        result += "Commit Message:" + "\" " + few_shot_list[i].message + "\"\n\n"
        result += "Original Code Cells:\n" 
        result += "\'\'\'\n"
        result += generate_code_prompt_whole_file(few_shot_list[i].old) + "\n"
        result += "\'\'\'\n"
        result += "]\n\n"
        result += "### Response:\n"
        result += generate_code_prompt_whole_file(few_shot_list[i].new) + "\n"
        result += "<|EOT|>\n\n"
    return result

def generate_few_shot_prompt_cell_only(training, fs):
    few_shot_list = random.sample(training, fs)
    result = ""
    for i in range (len(few_shot_list)):
        result += "### Instruction\n"
        result += "[\n"
        result += "Commit Message:" + "\" " + few_shot_list[i].message + "\"\n\n"
        result += "Original Code:\n" 
        result += "\'\'\'\n"
        result += generate_code_prompt_cell_only_old(few_shot_list[i].old, few_shot_list[i].cell_diff) + "\n"
        result += "\'\'\'\n"
        result += "]\n\n"
        result += "### Response:\n"
        result += generate_code_prompt_cell_only_new(few_shot_list[i].new, few_shot_list[i].cell_diff) + "\n"
        result += "<|EOT|>\n\n"
    return result


def generate_code_change_whole_file(tokenizer, model, commit_message, original_code_cells):

    original_code_prompt = generate_code_prompt_whole_file(original_code_cells)

    # src = f"""
    # You are a skilled software developer with immense knowledge in software analysis and debugging.
    # For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
    # Your job is to generate Jupyter notebook code cells changes given a commit message and original code cells.
    
    # ### Instruction:
    # [
    # Commit Message: "{commit_message}"
    # Original Code Cells:
    # '''
    # {original_code_prompt}
    # '''
    # ]
    
    # It is your turn to reponse the code only.
    # ### Response:
    # """

    result = ""
    result += "### Instruction\n"
    result += "[\n"
    result += "Commit Message:" + "\" " + commit_message + "\"\n\n"
    result += "Original Code Cells:\n" 
    result += "\'\'\'\n"
    result += original_code_prompt + "\n"
    result += "\'\'\'\n"
    result += "]\n\n"
    result += "### Response:\n"

    src_ids = tokenizer(result, return_tensors="pt").to(device)
    # src_ids = tokenizer(src, return_tensors="pt").to(device)
    with torch.no_grad():
        tgt_ids = model.generate(**src_ids, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    # tgt_ids = model.generate(**src_ids, max_new_tokens=512*2)
    tgt = tokenizer.decode(tgt_ids[0][len(src_ids.input_ids[0]):], skip_special_tokens=True)
    return tgt

def generate_code_change_cell_only(tokenizer, model, commit_message, old, cell_diff):
    original_code_prompt = generate_code_prompt_cell_only_old(old, cell_diff)
    # src = f"""
    # You are a skilled software developer with immense knowledge in software analysis and debugging.
    # For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
    # Your job is to generate Jupyter notebook python code changes given a commit message and original python code.
    
    # ### Instruction:
    # [
    # Commit Message: "{commit_message}"
    # Original Code:
    # '''
    # {original_code_prompt}
    # '''
    # ]
    
    # It is your turn to response code in a cell format.
    # ### Response:
    # """
    result = ""
    result += "### Instruction\n"
    result += "[\n"
    result += "Commit Message:" + "\" " + commit_message + "\"\n\n"
    result += "Original Code Cells:\n" 
    result += "\'\'\'\n"
    result += original_code_prompt + "\n"
    result += "\'\'\'\n"
    result += "]\n\n"
    result += "### Response:\n"
    
    src_ids = tokenizer(result, return_tensors="pt").to(device)
    # src_ids = tokenizer(src, return_tensors="pt").to(device)
    with torch.no_grad():
        tgt_ids = model.generate(**src_ids, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    # tgt_ids = model.generate(**src_ids, max_new_tokens=512*2)
    tgt = tokenizer.decode(tgt_ids[0][len(src_ids.input_ids[0]):], skip_special_tokens=True)
    return tgt

def generate_code_change_whole_file_fs(tokenizer, model, commit_message, original_code_cells, training, fs):
    original_code_prompt = generate_code_prompt_whole_file(original_code_cells)
    fs_promot = generate_few_shot_prompt_whole_file(training, fs)
    src = f"""
    You are a skilled software developer with immense knowledge in software analysis and debugging.  
    Your task is to generate Jupyter notebook Python code changes based on an Instruction provided.
    The Instruction includes a commit message describing the problem the developer is solving and the original code cells that needs modification.
    Your responses should only include the updated Python code in a cell format, reflecting the changes required by the commit message.

    You will see some examples.
    Each example contains an Instruction and a Response. The Response shows how the developer modifies the original code cells to solve the issue described in the commit message. 
    Learn from these examples below and apply the same approach to generate the response for Example 6, responding only with the updated Python code in a cell format.

    {fs_promot}

    ### Instruction:
    [
    Commit Message: "{commit_message}"
    Original Code Cells:
    '''
    {original_code_prompt}
    '''
    ]

    ### Response:
    """

    # src_ids = tokenizer(src, return_tensors="pt").to(accelerator.device)
    src_ids = tokenizer(src, return_tensors="pt").to(device)
    # tgt_ids = model.module.generate(**src_ids, max_new_tokens=512*2)
    with torch.no_grad():
        tgt_ids = model.generate(**src_ids, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    tgt = tokenizer.decode(tgt_ids[0][len(src_ids.input_ids[0]):], skip_special_tokens=True)

    return tgt

def generate_code_change_cell_only_fs(tokenizer, model, commit_message, old, cell_diff, training, fs):
    original_code_prompt = generate_code_prompt_cell_only_old(old, cell_diff)
    fs_promot = generate_few_shot_prompt_cell_only(training, fs)
    src = f"""
    You are a skilled software developer with immense knowledge in software analysis and debugging.  
    Your task is to generate Jupyter notebook Python code changes based on an Instruction provided.
    The Instruction includes a commit message describing the problem the developer is solving and the original code cells that needs modification.
    Your responses should only include the updated Python code in a cell format, reflecting the changes required by the commit message.

    You will see some examples.
    Each example contains an Instruction and a Response. The Response shows how the developer modifies the original code cells to solve the issue described in the commit message. 
    Learn from these examples below and apply the same approach to generate the response for Example 6, responding only with the updated Python code in a cell format.

    {fs_promot}

    ### Instruction:
    [
    Commit Message: "{commit_message}"
    Original Code Cells:
    '''
    {original_code_prompt}
    '''
    ]

    ### Response:
    """

    # src_ids = tokenizer(src, return_tensors="pt").to(accelerator.device)
    src_ids = tokenizer(src, return_tensors="pt").to(device)
    # tgt_ids = model.module.generate(**src_ids, max_new_tokens=512*2)
    with torch.no_grad():
        tgt_ids = model.generate(**src_ids, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    tgt = tokenizer.decode(tgt_ids[0][len(src_ids.input_ids[0]):], skip_special_tokens=True)
    return tgt


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("Start Loading")
data = su.io.load(Path.cwd() / "model/split/commits.jsonl", clz=RawData)
train, test, val = [], [], []
with open("model/split/train_index.txt", "r") as file:
    line = file.readline().strip()
    train = list(map(int, line.split(",")))
    train = [data[i] for i in train]

with open("model/split/test_index.txt", "r") as file:
    line = file.readline().strip()
    test = list(map(int, line.split(",")))
    test = [data[i] for i in test]

# with open("model/split/val_index.txt", "r") as file:
#     line = file.readline().strip()
#     val = list(map(int, line.split(",")))
#     val = [data[i] for i in val]

print("Loading Finished, Start Getting tokenizer and model")
# repo_groups = defaultdict(list)
# for d in data:
#     repo_groups[d.repo].append(d)
# repo_groups = list(repo_groups.items())
# random.shuffle(repo_groups)
# train, test, val = [], [], []
# # 70% train, 20% test, 10% validation
# for i, (repo, group) in enumerate(repo_groups):
#     if i % 10 < 7:
#         train.extend(group)
#     elif i % 10 < 9:
#         test.extend(group)
#     else:
#         val.extend(group)

# device_id = int(os.environ["LOCAL_RANK"])
# torch.cuda.set_device(device_id)

# to prevent CUDA OOM due to memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cell = ['', '/home/b27jin/ipynb-edit-internal/model/1.3b_cell/final_1','/home/b27jin/ipynb-edit-internal/model/1.3b_cell/final_2','/home/b27jin/ipynb-edit-internal/model/1.3b_cell/final_3']
file = ['', '/home/b27jin/ipynb-edit-internal/model/1.3b_file/final_1','/home/b27jin/ipynb-edit-internal/model/1.3b_file/final_2','/home/b27jin/ipynb-edit-internal/model/1.3b_file/final_3']
ftt=3
finetune = file[ftt]
tokenizer = AutoTokenizer.from_pretrained(finetune)
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")

quant_config = TorchAoConfig(quant_type="int8_weight_only")
# model = AutoModelForCausalLM.from_pretrained(
#     "deepseek-ai/deepseek-coder-6.7b-instruct",
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     quantization_config=quant_config
# )
model = AutoModelForCausalLM.from_pretrained(
    finetune,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # quantization_config=quant_config
)

# model, tokenizer = accelerator.prepare(model, tokenizer)

# move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Set device")

# model = torch.nn.parallel.DistributedDataParallel(model.to(device_id), device_ids=[device_id])
# model.to(device_id)
# model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
# print([device_id])

# print ("Tokenizer Ready, Start Base Line")
print ("Tokenizer Ready, Model is on " + str(device) + "\n Start Base Line")

# Expectation
# for d in test:
#     name = d.file.split("/")[-1]
#     file_name = "model/results/expected_whole_file/" + d.repo + "_" + d.commit + "_" + name + ".txt"
#     expected_output = generate_code_prompt_whole_file(d.new)
#     with open(file_name, "w") as text_file:
#         text_file.write(expected_output)
#     file_name = "model/results/expected_cell_only/" + d.repo + "_" + d.commit + "_" + name + ".txt"
#     expected_output = generate_code_prompt_cell_only_new(d.new, d.cell_diff)
#     with open(file_name, "w") as text_file:
#         text_file.write(expected_output)

# MAX_VALIDATION = 1

print("Whole file zero shot")
# count = 0
for d in tqdm(test):
    name = d.file.split("/")[-1]
    file_name = f"model/results/whole_file_zero_shot_1.3b_{ftt}/" + d.repo + "_" + d.commit + "_" + name + ".txt"
    torch.cuda.empty_cache()
    if not os.path.isfile(file_name):
        tgt = generate_code_change_whole_file(tokenizer, model, d.message, d.old)
        with open(file_name, "w") as text_file:
            text_file.write(tgt)
    # count += 1
    # if count >= MAX_VALIDATION:
    #     break

# print("Cell diff zero shot")
# # count = 0
# for d in tqdm(test):
#     name = d.file.split("/")[-1]
#     file_name = f"model/results/cell_only_zero_shot_1.3b_{ftt}/" + d.repo + "_" + d.commit + "_" + name + ".txt"
#     torch.cuda.empty_cache()
#     if not os.path.isfile(file_name):
#         tgt = generate_code_change_cell_only(tokenizer, model, d.message, d.old, d.cell_diff)
#         with open(file_name, "w") as text_file:
#             text_file.write(tgt)
#     # count += 1
#     # if count >= MAX_VALIDATION:
#     #     break

# print("Whole file one shot")
# count = 0
# for d in tqdm(test):
#     name = d.file.split("/")[-1]
#     tgt = generate_code_change_whole_file_fs(tokenizer, model, d.message, d.old, train, 1)
#     file_name = "model/results/whole_file_one_shot/" + d.repo + "_" + d.commit + "_" + name + ".txt"
#     with open(file_name, "w") as text_file:
#         text_file.write(tgt)
#     # count += 1
#     # if count >= MAX_VALIDATION:
#     #     break

# print("Cell only one shot")
# count = 0
# for d in tqdm(test):
#     name = d.file.split("/")[-1]
#     tgt = generate_code_change_cell_only_fs(tokenizer, model, d.message, d.old, d.cell_diff, train, 1)
#     file_name = "model/results/cell_only_one_shot/" + d.repo + "_" + d.commit + "_" + name + ".txt"
#     with open(file_name, "w") as text_file:
#         text_file.write(tgt)
#     # count += 1
#     # if count >= MAX_VALIDATION:
#     #     break

# print("Whole file Five shot")
# count = 0
# splitted_test = [test[i::5] for i in range(5)]
# for d in tqdm(splitted_test[3]):
#     name = d.file.split("/")[-1]
#     # file_name = "model/results/whole_file_five_shot_2/" + d.repo + "_" + d.commit + "_" + name + ".txt"
#     file_name = "model/results/whole_file_five_shot_7b/" + d.repo + "_" + d.commit + "_" + name + ".txt"

#     torch.cuda.empty_cache()
#     if not os.path.isfile(file_name):
#         tgt = generate_code_change_whole_file_fs(tokenizer, model, d.message, d.old, train, 5)
#         with open(file_name, "w") as text_file:
#             text_file.write(tgt)
#         # count += 1
#         # if count >= MAX_VALIDATION:
#         #     break

# print("Whole file Five shot")
# for d in tqdm(test):
#     name = d.file.split("/")[-1]
#     file_name = f"model/results/whole_file_five_shot_1.3b_{ftt}/" + d.repo + "_" + d.commit + "_" + name + ".txt"
#     torch.cuda.empty_cache()
#     # if not os.path.isfile(file_name):
#     tgt = generate_code_change_whole_file_fs(tokenizer, model, d.message, d.old, train, 5)
#     with open(file_name, "w") as text_file:
#         text_file.write(tgt)


#  CUDA_VISIBLE_DEVICES=0 /home/b27jin/bin/python /home/b27jin/ipynb-edit-internal/model/baseline.py
# print("Cell Only Five shot")
# # count = 0
# # splitted_test = [test[i::3] for i in range(3)]
# for d in tqdm(test):
#     name = d.file.split("/")[-1]
#     file_name = f"model/results/cell_only_five_shot_1.3b_{ftt}/" + d.repo + "_" + d.commit + "_" + name + ".txt"
#     torch.cuda.empty_cache()
#     # if not os.path.isfile(file_name):
#     tgt = generate_code_change_cell_only_fs(tokenizer, model, d.message, d.old, d.cell_diff, train, 5)
#     with open(file_name, "w") as text_file:
#         text_file.write(tgt)
#     # count += 1
#     # if count >= MAX_VALIDATION:
#     #     break