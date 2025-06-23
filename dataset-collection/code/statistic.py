import requests
import time
import json
import stat, os, shutil
import subprocess
import nbformat
from retry import retry
from git import rmtree
from pathlib import Path
from statistics import mean
from tqdm import tqdm

ipynb_counts = []
ipynb_files = []
contents_api_url = "https://api.github.com/repos/{owner}/{repo}/contents/{path}"
repo_number = 100

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
    concatenated_code = "\n".join(code_cells)
    return concatenated_code


# Function to count .ipynb files in a repository
def count_ipynb_files(owner, repo, directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith('.ipynb'):
                continue
            count += 1
            file_name = file
            file_path = os.path.join(root, file)
            content = ''
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
            except:
                file_path = u"\\\\?\\" + os.path.join(os.getcwd(), file_path)
                with open(file_path, 'r') as file:
                    content = file.read()
            content = concatenate_code_cells(content)
            line_count = content.count('\n') + 1
            ipynb_files.append({'user': owner, 'repo': repo, 'file_name': file_path, 'line_count': line_count})
    return count

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

@retry(tries=15, delay=2)
def retry_rmtree(directory):
    file_path = u"\\\\?\\" + os.path.join(os.getcwd(), directory)
    rmtree(file_path)

if not os.path.exists('clones'):
    os.makedirs('clones')

with open('top_1000_python_repos.json', 'r') as f:
    repos = json.load(f)[:repo_number]

for repo in tqdm(repos):
    owner = repo['owner']['login']
    repo_name = repo['name']
    repo_url = repo['clone_url']
    clone_dir = os.path.join('clones', f"{owner}_{repo_name}")
    # Clone the repository
    if clone_repository(repo_url, clone_dir):
        # Count .ipynb files
        count = count_ipynb_files(owner, repo_name, clone_dir)
        ipynb_counts.append(count)
        # shutil.rmtree(clone_dir, onerror=readonly_to_writable)
        try:
            retry_rmtree(clone_dir)
        except Exception as e:
            print(f"Failed to delete repository {clone_dir} after retries: {e}")

if ipynb_counts:
    max_ipynb = max(ipynb_counts)
    min_ipynb = min(ipynb_counts)
    avg_ipynb = mean(ipynb_counts)
    print(f"Maximum .ipynb files: {max_ipynb}")
    print(f"Minimum .ipynb files: {min_ipynb}")
    print(f"Average .ipynb files: {avg_ipynb:.2f}")

# with open('ipynb_counts_1000.txt', 'w') as f:
#     for c in ipynb_counts:
#         f.write(f"{c}\n")
with open('stat.json', 'w') as f:
    json.dump(ipynb_files, f, indent=2)