import requests
import json
import time
import os

TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise RuntimeError("Missing GITHUB_TOKEN environment variable.")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
URL = "https://api.github.com/graphql"

# 第一步：只抓取仓库名和ID
list_query = """
query($searchQuery: String!, $cursor: String) {
  search(query: $searchQuery, type: REPOSITORY, first: 50, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on Repository {
        nameWithOwner
        stargazerCount
      }
    }
  }
}
"""

# 第二步：针对单个仓库抓取详细信息（轻量化）
detail_query = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    repositoryTopics(first: 10) {
      nodes { topic { name } }
    }
    mentionableUsers(first: 50) {
      nodes { login }
    }
  }
}
"""

def get_repo_list(topic):
    repos = []
    cursor = None
    search_q = f"topic:{topic} stars:>1000 sort:stars"
    while len(repos) < 100:
        res = requests.post(URL, json={'query': list_query, 'variables': {"searchQuery": search_q, "cursor": cursor}}, headers=HEADERS).json()
        nodes = res['data']['search']['nodes']
        repos.extend(nodes)
        if not res['data']['search']['pageInfo']['hasNextPage']: break
        cursor = res['data']['search']['pageInfo']['endCursor']
        print(f"Listed {len(repos)} repos...")
    return repos

def get_details(repo_list):
    full_data = []
    for i, repo in enumerate(repo_list):
        owner, name = repo['nameWithOwner'].split('/')
        print(f"[{i+1}/{len(repo_list)}] Fetching details for {owner}/{name}...")
        
        try:
            res = requests.post(URL, json={'query': detail_query, 'variables': {"owner": owner, "name": name}}, headers=HEADERS).json()
            if 'data' in res and res['data']['repository']:
                repo.update(res['data']['repository'])
                full_data.append(repo)
            time.sleep(0.5) # 礼貌间歇
        except:
            print(f"Failed on {name}, skipping...")
    return full_data

# 执行
print("Step 1: Getting lists...")
all_list = get_repo_list("computer-vision") + get_repo_list("nlp")

print("Step 2: Getting details sequentially...")
final_results = get_details(all_list)

with open("raw_data.json", "w") as f:
    json.dump(final_results, f, indent=2)

print(f"Done! Saved {len(final_results)} items.")
