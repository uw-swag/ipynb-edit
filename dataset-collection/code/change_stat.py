import requests
import time
import json
import stat, os, shutil
import subprocess
import nbformat

import difflib
import dataclasses
import seutil as su

from typing import List, Tuple
from retry import retry
from git import rmtree
from pathlib import Path
from statistics import mean
from tqdm import tqdm
 
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

git_log_command = ['git', 'log', '--name-only', '--pretty=format:%H%n%s']
git_diff_command = 'git diff {}^1 -- {}'
repo_number = 1000

# Function to clone a repository
def clone_repository(repo_url, clone_dir):
    result = subprocess.run(['git', 'clone', repo_url, clone_dir], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to clone repository {repo_url}: {result.stderr}")
        return False
    return True

def readonly_to_writable(foo, file, err):
    if Path(file).suffix in ['.idx', '.pack'] and 'PermissionError' == err[0].__name__:
        os.chmod(file, stat.S_IWRITE)
        foo(file)

def extract_code_cells(notebook_content):
    try:
        notebook = nbformat.reads(notebook_content, as_version=4)
    except:
        return ""
    code_cells = []

    # Iterate through each cell in the notebook
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code = ''.join(cell.get('source', ''))  # Join the list of lines into a single string
            # Append the id and code to the list
            code_cells.append(code)

    return code_cells

def output_code_diff(string1, string2):
    result = difflib.SequenceMatcher(None, string1, string2).get_opcodes()
    result = [t for t in result if t[0] != "equal"]
    return result

def output_line_diff(string1, string2, celldiff):
    result = []
    replace = [t for t in celldiff if t[0] == 'replace']
    for t in replace:
        sub1 = '\n'.join(string1[t[1]:t[2]]).split('\n')
        sub2 = '\n'.join(string2[t[3]:t[4]]).split('\n')
        result.append(output_code_diff(sub1, sub2))
    return result


def concatenate_code_cells(notebook_content):
    try:
        notebook = nbformat.reads(notebook_content, as_version=4)
    except nbformat.reader.NotJSONError:
        return ""
    # Initialize a list to store code cells
    code_cells = []

    # Iterate through the cells in the notebook
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            code_cells.append(cell.source)
    
    # Concatenate all code cells into a single string
    concatenated_code = "<NEWCELL>\n".join(code_cells)
    return concatenated_code
    
@retry(tries=15, delay=2)
def retry_rmtree(directory):
    file_path = u"\\\\?\\" + os.path.join(os.getcwd(), directory)
    rmtree(file_path)


if not os.path.exists('clones'):
    os.makedirs('clones')

with open('data_fetching/results/top_1000_python_repos.json', 'r') as f:
    repos = json.load(f)[:repo_number]

ipynb_counts = []
all_commits = []
for repo in tqdm(repos):
    owner = repo['owner']['login']
    repo_name = repo['name']
    repo_url = repo['clone_url']
    clone_dir = os.path.join('clones', f"{owner}_{repo_name}")
    # Clone the repository
    if clone_repository(repo_url, clone_dir):
        git_log_output = subprocess.run(['git', 'log', '--name-only', '--pretty=format:%n%H%n%s']
                                    ,cwd=clone_dir, stdout=subprocess.PIPE, text=True).stdout
        commits = []
        current_commit = None
        process_flag = 0
        # # Process the log output
        for line in git_log_output.splitlines():
            if not line.strip():
                # empth line
                if current_commit:
                    commits.append(current_commit)
                process_flag = 0
                continue
            if process_flag == 0:
                # hash
                # print("hash: " + line)
                current_commit = {'commit': line, 'files': [], 'messages': []}
                process_flag = 1
            elif process_flag == 1:
                # Commit message
                # print("message: " + line)
                current_commit['messages'].append(line)
                process_flag = 2
            else:
                # File name
                #print("file: " + line)
                file_name = line.strip()
                if not file_name.endswith('.ipynb'):
                    continue
                file_path = os.path.join(clone_dir, file_name)
                #print(file_path)
                cur_commit = current_commit["commit"]
                pre_commit = cur_commit + '^'
                #cur_code = concatenate_code_cells(file_path)
                cur_content = subprocess.run(['git', 'show', f'{cur_commit}:{file_name}'], cwd=clone_dir,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
                pre_content = subprocess.run(['git', 'show', f'{pre_commit}:{file_name}'], cwd=clone_dir,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
                

                cur_code = extract_code_cells(cur_content)
                pre_code = extract_code_cells(pre_content)
                diff_output = output_code_diff(pre_code, cur_code)

                line_diff_output = output_line_diff(pre_code, cur_code, diff_output)

                current_commit['files'].append({
                    'file': file_name, 
                    'old' : pre_code,
                    'new' : cur_code,
                    'cell_diff': diff_output,
                    'line_diff': line_diff_output
                })
                
        if current_commit:
            commits.append(current_commit)

        # Append repo details to the main list
        for commit in commits:
            for file in commit['files']:
                rawdata = RawData(
                    repo=repo_name, 
                    commit=commit['commit'], 
                    file=file['file'], 
                    message= "".join(commit['messages']), 
                    old=file['old'], 
                    new=file['new'],
                    cell_diff=file['cell_diff'],
                    line_diff=file['line_diff']
                )
                all_commits.append(rawdata)

        try:
            retry_rmtree(clone_dir)
        except Exception as e:
            print(f"Failed to delete repository {clone_dir} after retries: {e}")
    else:
        try:
            retry_rmtree(clone_dir)
        except Exception as e:
            print(f"Failed to delete repository {clone_dir} after retries: {e}")

# Output to JSON 
# with open('commits2.json', 'w') as f:
#     json.dump(all_commits, f, indent=2)
su.io.dump(Path.cwd() / "data_fetching/results/commits.jsonl", all_commits)