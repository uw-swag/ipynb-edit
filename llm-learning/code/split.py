import os
import dataclasses
import random
import seutil as su
from typing import List, Tuple
from pathlib import Path
from pprint import pprint
from collections import defaultdict

import nltk
from tqdm import tqdm
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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
print("Start Loading")
data = su.io.load(Path.cwd() / "data_fetching/results/commits.jsonl", clz=RawData)

seen = set()

print("Start Filtering")
data_filtered = []
for d in tqdm(data):
    cur_str = d.repo + d.commit + d.file
    if (len(d.message.split()) >= 3) and (cur_str not in seen):
        data_filtered.append(d)
        seen.add(cur_str)
su.io.dump(Path.cwd() / "model/split/commits.jsonl", data_filtered)
data = data_filtered

print("Start Spliting")
repo_groups = defaultdict(list)
for index in range (len(data)):
    repo_groups[data[index].repo].append(index)
repo_groups = list(repo_groups.items())
random.shuffle(repo_groups)
train, test, val = [], [], []
# 70% train, 20% test, 10% validation
for i, (repo, indices) in enumerate(repo_groups):
    if i % 10 < 7:
        train.extend(indices)
    elif i % 10 < 9:
        test.extend(indices)
    else:
        val.extend(indices)

with open('model/split/train_index.txt', 'w') as file:
    file.write(",".join(map(str, train)))

with open('model/split/test_index.txt', 'w') as file:
    file.write(",".join(map(str, test)))

with open('model/split/val_index.txt', 'w') as file:
    file.write(",".join(map(str, val)))
