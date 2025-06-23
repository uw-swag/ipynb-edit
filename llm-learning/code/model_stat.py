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
    Trainer,
    TrainingArguments,
)

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

with open("model/split/val_index.txt", "r") as file:
    line = file.readline().strip()
    val = list(map(int, line.split(",")))
    val = [data[i] for i in val]

print("Loading Finished, Start getting commit message length")
message_length = []
for d in tqdm(data):
    message_length.append(len(d.message.split()))

with open('model/model_stat/commit_message_length.txt', 'w') as file:
    file.write(",".join(map(str, message_length)))

print("Message Length stat finish, Start collecting output token length of whole file")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is on" + str(device))
output_whole_file_token_length = []
input_whole_file_token_length = []
for d in tqdm(data):
    new = generate_code_prompt_whole_file(d.new)
    old = generate_code_prompt_whole_file(d.old)
    old_ids = tokenizer(old, return_tensors="pt").to(device)
    input_whole_file_token_length.append(old_ids['input_ids'].size(dim = 1))
    new_ids = tokenizer(new, return_tensors="pt").to(device)
    output_whole_file_token_length.append(new_ids['input_ids'].size(dim = 1))
with open('model/model_stat/whole_file_output_token_length.txt', 'w') as file:
    file.write(",".join(map(str, output_whole_file_token_length)))
with open('model/model_stat/whole_file_input_token_length.txt', 'w') as file:
    file.write(",".join(map(str, input_whole_file_token_length)))

print("Message Length stat finish, Start collecting output token length of cell only")
input_cell_only_token_length = []
output_cell_only_token_length = []
for d in tqdm(data):
    old = generate_code_prompt_cell_only_old(d.old, d.cell_diff)
    old_ids = tokenizer(old, return_tensors="pt").to(device)
    new = generate_code_prompt_cell_only_new(d.new, d.cell_diff)
    new_ids = tokenizer(new, return_tensors="pt").to(device)
    input_cell_only_token_length.append(old_ids['input_ids'].size(dim = 1))
    output_cell_only_token_length.append(new_ids['input_ids'].size(dim = 1))
with open('model/model_stat/cell_only_output_token_length.txt', 'w') as file:
    file.write(",".join(map(str, output_cell_only_token_length)))
with open('model/model_stat/cell_only_input_token_length.txt', 'w') as file:
    file.write(",".join(map(str, input_cell_only_token_length)))

print("All done...")