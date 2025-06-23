import os
import json
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

## Adding deleted and insert
def get_change_stat(old, new, cell, line):
    cell_change = 0
    line_change = 0
    li = 0
    for c in cell:
        old_cell = c[2] - c[1]
        new_cell = c[4] - c[3]
        cell_change = cell_change + new_cell + old_cell
        if c[0] == 'insert':
            for i in range(c[3], c[4]):
                line_change = line_change + len(new[i].split('\n'))
        elif c[0] == 'delete':
            for i in range(c[1], c[2]):
                line_change = line_change + len(old[i].split('\n'))
        elif c[0] == 'replace':
            cur_line_change = line[li]
            li = li + 1
            for l in cur_line_change:
                old_line = l[2] - l[1]
                new_line = l[4] - l[3]
                line_change = line_change + new_line + old_line
    return cell_change, line_change


def get_size(data):

    num_of_repo = 0         # across all repos
    line_of_codes = 0       # across all files

    commits_num = []          # across all repos
    change_file_num = []      # across all commits
    cell_change_num = []        # across all changed files  
    line_change_num = []        # across all changed files
    data_num = 0

    cur_repo_name = ''
    cur_commit_hash = ''
    commits_num.append(0)
    for d in tqdm(data):
        data_num += 1
        line_of_codes += len('\n'.join(d.new).split('\n'))
        if d.repo != cur_repo_name:
            cur_repo_name = d.repo
            cur_commit_hash = ''
            commits_num.append(0)
            num_of_repo += 1
        if d.commit != cur_commit_hash:
            cur_commit_hash = d.commit
            commits_num[-1] += 1
            change_file_num.append(1)
        else:
            change_file_num[-1] += 1
        cell_change, line_change = get_change_stat(d.old, d.new, d.cell_diff, d.line_diff)
        cell_change_num.append(cell_change)
        line_change_num.append(line_change)
    
    return (num_of_repo, np.sum(change_file_num), line_of_codes, np.sum(commits_num), data_num, np.sum(cell_change_num), np.sum(line_change_num))
    


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
full_size = get_size(data)
print(full_size)
train_size = get_size(train)
print(train_size)
val_size = get_size(val)
print(val_size)
test_size = get_size(test)
print(test_size)