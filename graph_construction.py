import networkx as nx
from networkx.algorithms import bipartite
import json
from pyvis.network import Network
import community as community_louvain

# 1. 初始化二分图
B = nx.Graph()

print("Loading data and building bipartite graph...")
with open("raw_data.json", "r") as f:
    data = json.load(f)

# 记录每个用户参与的仓库数量，用于后续筛选“精英开发者”
user_repo_map = {}

for repo in data:
    repo_name = repo['nameWithOwner']
    topics = [t['topic']['name'] for t in repo['repositoryTopics']['nodes']]
    users = [u['login'] for u in repo['mentionableUsers']['nodes']]
    
    # 过滤 Bot
    users = [u for u in users if "[bot]" not in u.lower() and u != "web-flow"]
    
    for u in users:
        user_repo_map.setdefault(u, set()).add(repo_name)
    
    for topic in topics:
        for user in users:
            if B.has_edge(user, topic):
                B[user][topic]['weight'] += 1
            else:
                B.add_node(user, bipartite=0) # 开发者
                B.add_node(topic, bipartite=1) # 技术标签
                B.add_edge(user, topic, weight=1)

print(f"Bipartite Graph: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")

# 2. 筛选精英开发者 (可选但建议)
# 只保留参与过 2 个以上仓库的开发者，他们才是连接不同技术的“桥梁”
elite_users = [u for u, repos in user_repo_map.items() if len(repos) > 1]
print(f"Elite users count: {len(elite_users)}")

# 3. 进行 Topic 投影
# 投影结果：节点全是 Topic，边代表它们被同一个开发者共同贡献
print("Projecting to Topic-Topic network...")
topic_nodes = [n for n, d in B.nodes(data=True) if d['bipartite'] == 1]
# 使用加权投影：权重 = 共同拥有的开发者数量
T = bipartite.weighted_projected_graph(B, topic_nodes)

# 4. 过滤弱连接，让图更清晰
# 只有当 2 个以上的开发者同时掌握这两个 Topic 时，才保留连边
T.remove_edges_from([(u, v) for u, v, d in T.edges(data=True) if d['weight'] < 2])
# 移除孤立的 Topic 节点
T.remove_nodes_from(list(nx.isolates(T)))

print(f"Projected Graph: {T.number_of_nodes()} nodes, {T.number_of_edges()} edges")

# 5. 深度分析：中心度与社区发现 (让项目充实的关键)
# 计算 PageRank 中心度（影响力）
pr = nx.pagerank(T, weight='weight')
# 运行 Louvain 算法进行社区发现（技术流派）
partition = community_louvain.best_partition(T)

# 6. 使用 PyVis 可视化
net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)

# 准备可视化属性
for node in T.nodes():
    # 节点大小由 PageRank 决定
    size = pr[node] * 1000
    # 颜色由所属社区决定
    group = partition[node]
    net.add_node(node, size=size, title=f"PR Score: {pr[node]:.4f}", group=group)

# 添加边
for u, v, d in T.edges(data=True):
    net.add_edge(u, v, value=d['weight'], title=f"Shared Devs: {d['weight']}")

# 设置物理布局（因为节点少了，可以开启物理模拟，效果更酷炫）
net.force_atlas_2based()
net.show_buttons(filter_=['physics']) # 在页面下方显示物理参数调节按钮

print("Generating topic_projection.html...")
net.write_html("topic_projection.html")