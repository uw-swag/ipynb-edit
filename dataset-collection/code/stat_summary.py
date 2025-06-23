import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from statistics import mean
from tqdm import tqdm

stat = []

file_num = []  # number of ipynb file across all repo
line_num = []  # number of code lines across all files

def plot_and_save_distribution(data, filename, title = 'Distribution Plot',
                                xlabel = 'Value', ylabel = 'Frequency',
                                x_min = 0, x_max = 100, bin_width = 10):
    plt.clf()
    
    data_max = max(data)
    data_min = min(data)
    data_avg = mean(data)

    bins = np.arange(x_min, x_max + 1, bin_width)  # Bins for the specified range
    bins = np.concatenate(([-np.inf], bins, [np.inf]))  # Adding bins for values out of the range

    # Create the distribution plot
    sns.histplot(data, bins=bins, kde=True)

    # Add title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.figtext(0.15, 0.8, f'Min: {data_min}', fontsize=12, ha='left')
    plt.figtext(0.15, 0.75, f'Max: {data_max}', fontsize=12, ha='left')
    plt.figtext(0.15, 0.7, f'Average: {data_avg:.2f}', fontsize=12, ha='left')

    # Save the plot as an image
    plt.savefig(filename, format='png')


if not os.path.exists('stat_summary'):
    os.makedirs('stat_summary')

with open('data_fetching/results/stat.json') as file:
    stat = json.load(file)

cur_repo_name = ''
for item in tqdm(stat):
    if item['repo'] != cur_repo_name:
        cur_repo_name = item['repo']
        file_num.append(1)
    else:
        file_num[-1] += 1
    line_num.append(item['line_count'])

plot_and_save_distribution(file_num, 'stat_summary/file_num.png', title='Number of ipynb files for each repo', 
                           xlabel='Number of ipynb file', ylabel='Frequence',
                           x_min = 0, x_max = 1000, bin_width=100)

plot_and_save_distribution(line_num, 'stat_summary/line_num.png', title='Number of code lines for each ipynb file', 
                           xlabel='Number of code lines', ylabel='Frequence',
                           x_min = 0, x_max = 2000, bin_width=100)


    
