import pickle,statistics

tpe = 'whole'
with open(f'model/results/metrics_{tpe}_postprocessing_1.3b_finetuned_1.pkl', 'rb') as f:
    score_1=pickle.load(f)
with open(f'model/results/metrics_{tpe}_postprocessing_1.3b_finetuned_2.pkl', 'rb') as f:
    score_2=pickle.load(f)
with open(f'model/results/metrics_{tpe}_postprocessing_1.3b_finetuned_3.pkl', 'rb') as f:
    score_3=pickle.load(f)

em_list = score_1['em']+ score_2['em'] + score_3['em']
bleu_list = score_1['bleu']+ score_2['bleu'] + score_3['bleu']
code_bleu_list = score_1['code_bleu']+ score_2['code_bleu'] + score_3['code_bleu']
edit_sim_list = score_1['sim']+ score_2['sim'] + score_3['sim']
rouge_l_list = score_1['rouge']+ score_2['rouge'] + score_3['rouge']

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
print("Rouge L-Precision: " ,avg_p)
print("Rouge L-Recall: " ,avg_r)
print("Rouge L-F1: " ,avg_f)

with open(f'model/results/metrics_{tpe}_unpostprocessing_1.3b_finetuned_1.pkl', 'rb') as f:
    score_1=pickle.load(f)
with open(f'model/results/metrics_{tpe}_unpostprocessing_1.3b_finetuned_2.pkl', 'rb') as f:
    score_2=pickle.load(f)
with open(f'model/results/metrics_{tpe}_unpostprocessing_1.3b_finetuned_3.pkl', 'rb') as f:
    score_3=pickle.load(f)

em_list = score_1['em']+ score_2['em'] + score_3['em']
bleu_list = score_1['bleu']+ score_2['bleu'] + score_3['bleu']
code_bleu_list = score_1['code_bleu']+ score_2['code_bleu'] + score_3['code_bleu']
edit_sim_list = score_1['sim']+ score_2['sim'] + score_3['sim']
rouge_l_list = score_1['rouge']+ score_2['rouge'] + score_3['rouge']

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
print("Rouge L-Precision: " ,avg_p)
print("Rouge L-Recall: " ,avg_r)
print("Rouge L-F1: " ,avg_f)