import os
import torch
from teco.eval.metrics import exact_match, bleu, code_bleu, edit_sim, rouge_l
from transformers import (
    AutoTokenizer
)
from tqdm import tqdm
import statistics
import pickle
import re
# move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_txt_file_names(folder_path):
    # List to store file names
    txt_files = []
    
    # Iterate through the files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file has a .txt extension
        if file_name.endswith('.txt'):
            txt_files.append(file_name)
    
    return txt_files

em_list = []
bleu_list = []
code_bleu_list = []
edit_sim_list = []
rouge_l_list = []

em_list_2 = []
bleu_list_2 = []
code_bleu_list_2 = []
edit_sim_list_2 = []
rouge_l_list_2 = []

output_folder = "model/results/whole_file_zero_shot_1.3b"
expected_folder = "model/results/expected_whole_file"

cell = ['', '/home/b27jin/ipynb-edit-internal/model/1.3b_cell/final_1','/home/b27jin/ipynb-edit-internal/model/1.3b_cell/final_2','/home/b27jin/ipynb-edit-internal/model/1.3b_cell/final_3']
file = ['', '/home/b27jin/ipynb-edit-internal/model/1.3b_file/final_1','/home/b27jin/ipynb-edit-internal/model/1.3b_file/final_2','/home/b27jin/ipynb-edit-internal/model/1.3b_file/final_3']

ftt = 2
file_names = get_txt_file_names(output_folder+f'_{ftt}')
finetune = file[ftt]
tokenizer = AutoTokenizer.from_pretrained(finetune)

for name in tqdm(file_names):
    e_file = expected_folder + "/" + name
    o_file = output_folder + f"_{ftt}/" + name
    e_string = []
    o_string = []
    with open(e_file, 'r') as file:
        e_string = file.read()
    with open(o_file, 'r') as file:
        o_string = file.read()

    e_token_id = tokenizer.encode(e_string, add_special_tokens=True)
    e_token = tokenizer.convert_ids_to_tokens(e_token_id)
    o_token_id = tokenizer.encode(o_string, add_special_tokens=True)
    o_token = tokenizer.convert_ids_to_tokens(o_token_id)
    em_list.append(exact_match(e_token, o_token))
    bleu_list.append(bleu(e_token, o_token))
    code_bleu_list.append(code_bleu(e_token,o_token, 'Python3'))
    edit_sim_list.append(edit_sim(e_token, o_token))
    rouge_l_list.append(rouge_l(e_token, o_token))

    patterns_to_remove = [
        r'```', 
        r"'''", 
        r"'''\s*##?# Response:", 
    ]
    for pattern in patterns_to_remove:
        o_string = re.sub(pattern, '', o_string, flags=re.DOTALL)

    instruction_pattern = r'##?# Instruction:\s*\[\s*Commit Message:.*?Original Code Cells:\s*\'\'\''
    o_string = re.sub(instruction_pattern, '', o_string, flags=re.DOTALL)
    instruction_pattern = r'##?# Response:'
    o_string = re.sub(instruction_pattern, '', o_string, flags=re.DOTALL)
    instruction_pattern = r'```python'
    o_string = re.sub(instruction_pattern, '', o_string, flags=re.DOTALL)

    e_token_id = tokenizer.encode(e_string, add_special_tokens=True)
    e_token = tokenizer.convert_ids_to_tokens(e_token_id)
    o_token_id = tokenizer.encode(o_string, add_special_tokens=True)
    o_token = tokenizer.convert_ids_to_tokens(o_token_id)
    em_list_2.append(exact_match(e_token, o_token))
    bleu_list_2.append(bleu(e_token, o_token))
    code_bleu_list_2.append(code_bleu(e_token,o_token, 'Python3'))
    edit_sim_list_2.append(edit_sim(e_token, o_token))
    rouge_l_list_2.append(rouge_l(e_token, o_token))

score = {"em":em_list, "bleu":bleu_list, "code_bleu":code_bleu_list, "sim":edit_sim_list, "rouge": rouge_l_list}
score_2 = {"em":em_list_2, "bleu":bleu_list_2, "code_bleu":code_bleu_list_2, "sim":edit_sim_list_2, "rouge": rouge_l_list_2}

print("metrics_whole_unpostprocessing_1.3b")
with open(f'model/results/metrics_whole_unpostprocessing_1.3b_finetuned_{ftt}.pkl', 'wb') as f:
    pickle.dump(score, f)
with open(f'model/results/metrics_whole_postprocessing_1.3b_finetuned_{ftt}.pkl', 'wb') as f:
    pickle.dump(score_2, f)
#     pickle.load('model/results/rouge_l_list_file.pkl')

print("Exact Match: " , statistics.mean(em_list))
print("BLEU: " ,statistics.mean(bleu_list))
print("Code BLEU: " ,statistics.mean(code_bleu_list))
print("Edit SIM: " ,statistics.mean(edit_sim_list))

sum_p = sum_r = sum_f = 0
for d in rouge_l_list:
    sum_p += d['p']
    sum_r += d['r']
    sum_f += d['f']

n = len(rouge_l_list)
avg_p = sum_p / n
avg_r = sum_r / n
avg_f = sum_f / n
# print("Rouge L-Precision: " ,avg_p)
# print("Rouge L-Recall: " ,avg_r)
print("Rouge L-F1: " ,avg_f)


print("metrics_whole_postprocessing_1.3b")

print("Exact Match: " , statistics.mean(em_list_2))
print("BLEU: " ,statistics.mean(bleu_list_2))
print("Code BLEU: " ,statistics.mean(code_bleu_list_2))
print("Edit SIM: " ,statistics.mean(edit_sim_list_2))

sum_p = sum_r = sum_f = 0
for d in rouge_l_list_2:
    sum_p += d['p']
    sum_r += d['r']
    sum_f += d['f']

n = len(rouge_l_list_2)
avg_p = sum_p / n
avg_r = sum_r / n
avg_f = sum_f / n
# print("Rouge L-Precision: " ,avg_p)
# print("Rouge L-Recall: " ,avg_r)
print("Rouge L-F1: " ,avg_f)
