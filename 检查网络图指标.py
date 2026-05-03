import pickle
import networkx as nx

# 假设您已有缓存的图文件 output/graphs/Chongqing_bus.graphml
G = nx.read_graphml('output/graphs/Hangzhou_bus.graphml')

print("节点总数:", G.number_of_nodes())
print("边总数:", G.number_of_edges())
print("是否连通:", nx.is_connected(G))

# 统计连通分量
components = list(nx.connected_components(G))
print("连通分量个数:", len(components))
largest_cc = max(components, key=len)
print("最大连通分量节点数:", len(largest_cc))
print("最大连通分量占比: {:.2%}".format(len(largest_cc) / G.number_of_nodes()))

# 查看度分布前10名
deg = sorted(dict(G.degree()).items(), key=lambda x: x[1], reverse=True)[:10]
print("度数最高的10个节点:", deg)

# 计算最大连通分量内部的效率
G_lcc = G.subgraph(largest_cc).copy()
eff_lcc = nx.global_efficiency(G_lcc)
print("最大连通分量内部的全局效率:", eff_lcc)