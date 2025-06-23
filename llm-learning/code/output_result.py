import numpy as np
message_length = []
whole_file_input_token_length = []
whole_file_output_token_length = []
cell_only_input_token_length = []
cell_only_output_token_length = []
with open("model/model_stat/commit_message_length.txt", "r") as file:
    line = file.readline().strip()
    message_length = list(map(int, line.split(",")))
with open("model/model_stat/whole_file_output_token_length.txt", "r") as file:
    line = file.readline().strip()
    whole_file_output_token_length = list(map(int, line.split(",")))
with open("model/model_stat/whole_file_input_token_length.txt", "r") as file:
    line = file.readline().strip()
    whole_file_input_token_length = list(map(int, line.split(",")))
with open("model/model_stat/cell_only_output_token_length.txt", "r") as file:
    line = file.readline().strip()
    cell_only_output_token_length = list(map(int, line.split(",")))
with open("model/model_stat/cell_only_input_token_length.txt", "r") as file:
    line = file.readline().strip()
    cell_only_input_token_length = list(map(int, line.split(",")))
print("======================")
print("Commit Message:")
print("Average: " + str(np.mean(message_length)))
print("Min: " + str(np.min(message_length)))
print("Med: " + str(np.median(message_length)))
print("75%: " + str(np.percentile(message_length, 75)))
print("90%: " + str(np.percentile(message_length, 90)))
print("Max: " + str(np.max(message_length)))

print("======================")
print("#Token file before diff:")
print("Average: " + str(np.mean(whole_file_input_token_length)))
print("Min: " + str(np.min(whole_file_input_token_length)))
print("Med: " + str(np.median(whole_file_input_token_length)))
print("75%: " + str(np.percentile(whole_file_input_token_length, 75)))
print("90%: " + str(np.percentile(whole_file_input_token_length, 90)))
print("Max: " + str(np.max(whole_file_input_token_length)))

print("======================")
print("#Token file after diff:")
print("Average: " + str(np.mean(whole_file_output_token_length)))
print("Min: " + str(np.min(whole_file_output_token_length)))
print("Med: " + str(np.median(whole_file_output_token_length)))
print("75%: " + str(np.percentile(whole_file_output_token_length, 75)))
print("90%: " + str(np.percentile(whole_file_output_token_length, 90)))
print("Max: " + str(np.max(whole_file_output_token_length)))

print("======================")
print("#Token cell before diff:")
print("Average: " + str(np.mean(cell_only_input_token_length)))
print("Min: " + str(np.min(cell_only_input_token_length)))
print("Med: " + str(np.median(cell_only_input_token_length)))
print("75%: " + str(np.percentile(cell_only_input_token_length, 75)))
print("90%: " + str(np.percentile(cell_only_input_token_length, 90)))
print("Max: " + str(np.max(cell_only_input_token_length)))

print("======================")
print("#Token cell after diff:")
print("Average: " + str(np.mean(cell_only_output_token_length)))
print("Min: " + str(np.min(cell_only_output_token_length)))
print("Med: " + str(np.median(cell_only_output_token_length)))
print("75%: " + str(np.percentile(cell_only_output_token_length, 75)))
print("90%: " + str(np.percentile(cell_only_output_token_length, 90)))
print("Max: " + str(np.max(cell_only_output_token_length)))
