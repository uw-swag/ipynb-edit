import requests
import json
import os

# Your GitHub personal access token
TOKEN = os.getenv("GITHUB_TOKEN")
if (TOKEN):
    print("TOKEN OK")
else:
    print("Filed to get TOKEN")

# Headers for the request
headers = {
    'Authorization': f'token {TOKEN}',
    'Accept': 'application/vnd.github.v3+json',
}

# Base URL for the GitHub API
base_url = 'https://api.github.com/search/repositories'

# Parameters for the search query
query_params = {
    'q': 'topic:jupyter-notebook topic:machine-learning',
    'sort': 'stars',
    'order': 'desc',
    'per_page': 100,
}

repositories = []

# Fetching top 1000 repositories
for page in range(1, 11):  # 10 pages, 100 results per page
    query_params['page'] = page
    response = requests.get(base_url, headers=headers, params=query_params)
    
    if response.status_code == 200:
        result = response.json()
        repositories.extend(result.get('items', []))
    else:
        print(f"Failed to fetch page {page}: {response.status_code}")
        break


# query_params['page'] = 1
# response = requests.get(base_url, headers=headers, params=query_params)

# if response.status_code == 200:
#     result = response.json()
#     repositories.extend(result.get('items', []))
# else:
#     print(f"Failed to fetch page {1}: {response.status_code}")




# Save the results to a file (optional)
with open('data_fetching/results/top_1000_python_repos.json', 'w') as file:
    json.dump(repositories, file, indent=4)

print(f"Fetched {len(repositories)} repositories.")