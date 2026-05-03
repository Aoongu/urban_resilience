"""
城市公共交通网络韧性评估与瓶颈识别
基于CPTOND-2025数据集（Shapefile格式）
要求：geopandas, networkx, karateclub, scikit-learn, matplotlib, seaborn, tqdm等
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from shapely.geometry import Point, LineString
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from karateclub import Graph2Vec
import pickle
import multiprocessing as mp
from functools import partial
import random

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 配置参数 =====================
DATA_DIR = "./dataset"          # 数据根目录，包含bus/shapefiles和metro/shapefiles
OUTPUT_DIR = "./output"         # 输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 公交与地铁的城市文件夹模式（根据实际文件命名调整）
# 假设城市文件夹在 bus/shapefiles/城市名/ 下，文件名为 拼音_bus_stops.shp 等
BUS_SHAPEFILE_ROOT = os.path.join(DATA_DIR, "bus", "shapefiles")
METRO_SHAPEFILE_ROOT = os.path.join(DATA_DIR, "metro", "shapefiles")

# 城市列表（可从文件夹自动获取，这里手动指定示例城市）
CITIES = [d for d in os.listdir(BUS_SHAPEFILE_ROOT)
          if os.path.isdir(os.path.join(BUS_SHAPEFILE_ROOT, d))]
#CITIES = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Chengdu", "Wuhan", "Xian", "Nanjing"]
#CITIES = ["Jinan"]
# 注意：实际运行时可通过 os.listdir 自动获取所有城市文件夹

# ===================== 1. 图构建函数 =====================
def build_graph_from_shapefile(city_name, mode='bus', max_snap_distance=300):
    """
    从shapefile构建NetworkX图
    mode: 'bus' 或 'metro'
    max_snap_distance: 用于合并距离极近的站点（米），避免重复节点
    返回: G (networkx.Graph), node_pos (dict)
    """
    if mode == 'bus':
        root = BUS_SHAPEFILE_ROOT
        stops_file = os.path.join(root, city_name, f"{city_name.lower()}_bus_stops.shp")
        routes_file = os.path.join(root, city_name, f"{city_name.lower()}_bus_routes.shp")
    else:
        root = METRO_SHAPEFILE_ROOT
        stops_file = os.path.join(root, city_name, f"{city_name.lower()}_metro_stops.shp")
        routes_file = os.path.join(root, city_name, f"{city_name.lower()}_metro_routes.shp")

    # 检查文件存在
    if not os.path.exists(stops_file) or not os.path.exists(routes_file):
        print(f"警告：{city_name} 的 {mode} 数据缺失，跳过")
        return None, None

    stops_gdf = gpd.read_file(stops_file)
    routes_gdf = gpd.read_file(routes_file)

    # 创建空图
    G = nx.Graph()

    # 添加节点：使用站点ID作为节点标识，存储经纬度属性
    node_pos = {}
    for idx, row in stops_gdf.iterrows():
        # 根据实际字段调整，常见字段：'stop_id', 'stop_name', 'lng', 'lat'
        # 若字段名不同，请根据gdf.columns修改
        if 'stop_id' in row.index:
            node_id = row['stop_id']
        else:
            node_id = f"{city_name}_{mode}_{idx}"

        # 获取坐标（假设geometry为Point）
        if row.geometry is not None:
            lon, lat = row.geometry.x, row.geometry.y
        else:
            continue
        G.add_node(node_id, x=lon, y=lat, mode=mode)
        node_pos[node_id] = (lon, lat)

    # 添加边：通过线路geometry提取相邻站点
    # 注意：如果数据已有segment shapefile，可直接用；否则需从线路geometry中提取站点序列
    # 这里演示从routes_gdf中提取（需routes_gdf包含站点序列信息，或使用segment文件）
    # 简易方法：使用线路LineString与站点空间匹配，但速度慢。推荐使用已生成的segment文件。
    # 下面使用假设的segment文件（若存在），否则回退到简易空间连接
    segment_file = os.path.join(root, city_name, f"{city_name.lower()}_{mode}_segments.shp")
    if os.path.exists(segment_file):
        seg_gdf = gpd.read_file(segment_file)
        # 假设segment文件有 s_stopid, e_stopid 字段
        for _, seg in seg_gdf.iterrows():
            u = seg['s_stopid']
            v = seg['e_stopid']
            if u in G and v in G:
                # 计算距离（米）
                dist = seg.geometry.length * 111000  # 粗略转换，实际可用pyproj
                G.add_edge(u, v, length=dist, mode=mode)
    else:
        print(f"未找到segment文件，使用空间邻近方法构建边（较慢）...")
        # 备选根据routes_gdf的geometry与站点空间连接，需自己实现，此处略
        #         # 为了演示，我们简单将同一线路上的站：点按顺序连接
        # 此处省略具体实现，实际建议使用segment文件
        pass

    # 合并距离过近的节点（可选）
    # 此处省略合并步骤，可按需添加

    return G, node_pos

def build_multilayer_graph(city_name, snap_threshold=300):
    """构建耦合网络（直接使用缓存版图构建函数）"""
    G_bus = build_and_cache_graph(city_name, mode='bus')
    G_metro = build_and_cache_graph(city_name, mode='metro')
    if G_bus is None or G_metro is None:
        return None

    G_multi = nx.Graph()
    # 添加公交层
    for node, data in G_bus.nodes(data=True):
        G_multi.add_node(f"bus_{node}", layer='bus', x=data.get('x', 0), y=data.get('y', 0))
    for u, v, data in G_bus.edges(data=True):
        G_multi.add_edge(f"bus_{u}", f"bus_{v}", layer='bus', length=data.get('length', 0))

    # 添加地铁层
    for node, data in G_metro.nodes(data=True):
        G_multi.add_node(f"metro_{node}", layer='metro', x=data.get('x', 0), y=data.get('y', 0))
    for u, v, data in G_metro.edges(data=True):
        G_multi.add_edge(f"metro_{u}", f"metro_{v}", layer='metro', length=data.get('length', 0))

    # 添加换乘边
    from scipy.spatial import KDTree
    bus_coords = np.array([[G_bus.nodes[n].get('x', 0), G_bus.nodes[n].get('y', 0)] for n in G_bus.nodes()])
    metro_coords = np.array([[G_metro.nodes[n].get('x', 0), G_metro.nodes[n].get('y', 0)] for n in G_metro.nodes()])
    bus_ids = list(G_bus.nodes())
    metro_ids = list(G_metro.nodes())

    tree_bus = KDTree(bus_coords)
    for i, m_coord in enumerate(metro_coords):
        dist, idx = tree_bus.query(m_coord)
        if dist * 111000 < snap_threshold:  # 粗略距离换算
            G_multi.add_edge(f"bus_{bus_ids[idx]}", f"metro_{metro_ids[i]}",
                             layer='inter', length=dist*111000)

    return G_multi

# ===================== 2. 图嵌入与聚类 =====================
def compute_graph_embeddings(graphs, dim=128):
    """
    使用Graph2Vec计算图嵌入向量
    graphs: 字典 {city_name: networkx.Graph}
    """
    # 转换图为Graph2Vec所需格式：节点必须为整数索引从0开始
    graph_list = []
    city_names = []
    for city, G in graphs.items():
        if G is None or G.number_of_nodes() == 0:
            continue
        # 将节点标签转换为整数索引，保留原始ID在属性中
        G_int = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute='original_id')
        graph_list.append(G_int)
        city_names.append(city)

    if len(graph_list) == 0:
        raise ValueError("No valid graphs for embedding")

    model = Graph2Vec(dimensions=dim, wl_iterations=2, attributed=False)
    model.fit(graph_list)
    embeddings = model.get_embedding()

    return embeddings, city_names

def cluster_cities(embeddings, city_names, n_clusters=4):
    """
    对城市嵌入向量进行KMeans聚类，并用t-SNE可视化
    """
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(emb_scaled)

    # t-SNE降维可视化
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(city_names)-1))
    emb_2d = tsne.fit_transform(emb_scaled)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='Set1', s=100)
    for i, name in enumerate(city_names):
        plt.annotate(name, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=9)
    plt.title("城市公交网络结构 t-SNE 投影与聚类")
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(os.path.join(OUTPUT_DIR, "city_clustering_tsne.png"), dpi=150)
    plt.close()

    # 保存聚类结果
    result_df = pd.DataFrame({'city': city_names, 'cluster': labels})
    result_df.to_csv(os.path.join(OUTPUT_DIR, "city_clusters.csv"), index=False)
    print("聚类结果已保存。")

    return labels

# ===================== 3. 韧性评估（级联失效模拟） =====================
def simulate_attack(G, attack_strategy='degree', fractions=np.linspace(0, 0.5, 21)):
    """
    模拟节点删除，返回韧性指标序列
    attack_strategy: 'random', 'degree', 'betweenness'
    fractions: 删除比例数组
    返回: (lcc_sizes, efficiencies)
    """
    N = G.number_of_nodes()
    if N == 0:
        return [], []

    # 确定删除顺序
    if attack_strategy == 'random':
        nodes = list(G.nodes())
        np.random.shuffle(nodes)
    elif attack_strategy == 'degree':
        nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    elif attack_strategy == 'betweenness':
        bc = nx.betweenness_centrality(G)
        nodes = sorted(G.nodes(), key=lambda x: bc[x], reverse=True)
    else:
        raise ValueError("Unknown strategy")

    lcc_frac = []
    eff_vals = []
    G_work = G.copy()

    for p in fractions:
        remove_count = int(p * N)
        if remove_count == 0:
            # 初始状态
            if nx.is_connected(G_work):
                lcc = 1.0
            else:
                largest_cc = max(nx.connected_components(G_work), key=len)
                lcc = len(largest_cc) / N
            lcc_frac.append(lcc)
            eff_vals.append(nx.global_efficiency(G_work))
            continue

        # 删除节点
        to_remove = nodes[:remove_count]
        G_work.remove_nodes_from(to_remove)

        # 计算最大连通分量比例
        if G_work.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(G_work), key=len)
            lcc = len(largest_cc) / N
        else:
            lcc = 0.0
        lcc_frac.append(lcc)

        # 计算全局效率
        if G_work.number_of_nodes() > 1:
            eff = nx.global_efficiency(G_work)
        else:
            eff = 0.0
        eff_vals.append(eff)

    return lcc_frac, eff_vals

# ===================== 4. 关键瓶颈站点识别 =====================
def identify_bottlenecks_with_cache(G, city_name, mode='bus', top_n=10, force_recompute=False):
    """
    计算中心性并识别瓶颈站点，结果保存为 CSV。
    """
    cache_dir = os.path.join(OUTPUT_DIR, "bottlenecks")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{city_name}_{mode}_bottlenecks.csv")

    if not force_recompute and os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        print(f"  从缓存加载 {city_name} {mode} 瓶颈站点")
        return df

    if G is None or G.number_of_nodes() == 0:
        return None

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()))

    nodes = list(G.nodes())
    df = pd.DataFrame({
        'node': nodes,
        'degree': [degree.get(n, 0) for n in nodes],
        'betweenness': [betweenness.get(n, 0) for n in nodes],
    })
    df['degree_norm'] = df['degree'] / df['degree'].max()
    df['btw_norm'] = df['betweenness'] / df['betweenness'].max()
    df['score'] = 0.5 * df['degree_norm'] + 0.5 * df['btw_norm']
    df = df.sort_values('score', ascending=False).head(top_n)

    df.to_csv(cache_file, index=False)
    print(f"  已缓存 {city_name} {mode} 瓶颈站点")

    # 同时绘制并保存瓶颈地图
    plot_bottleneck_map(G, df, city_name, mode)
    return df
def approximate_global_efficiency(G, sample_ratio=0.1, min_samples=100, seed=42):
    """
    近似全局效率：随机抽取一部分节点对计算最短路径，估算整体效率
    """
    if G.number_of_nodes() < 2:
        return 0.0
    nodes = list(G.nodes())
    n = len(nodes)
    # 确定采样对数
    max_pairs = n * (n - 1) / 2
    sample_size = max(min_samples, int(max_pairs * sample_ratio))
    sample_size = min(sample_size, max_pairs)

    random.seed(seed)
    pairs = set()
    while len(pairs) < sample_size:
        u, v = random.sample(nodes, 2)
        if u != v:
            pairs.add((u, v) if u < v else (v, u))  # 无向图

    total_inv_dist = 0.0
    for u, v in pairs:
        try:
            d = nx.shortest_path_length(G, source=u, target=v)
            total_inv_dist += 1.0 / d
        except nx.NetworkXNoPath:
            pass  # 不连通，贡献为0

    # 估算整体效率 = (1/(n(n-1))) * sum(1/d)
    efficiency = total_inv_dist / (n * (n - 1))
    return efficiency
def simulate_attack_fast(G, attack_strategy='degree', fractions=np.linspace(0, 0.5, 11)):
    """
    快速级联失效模拟，仅计算LCC和近似效率
    """
    N = G.number_of_nodes()
    if N == 0:
        return [], []

    # 确定删除顺序
    if attack_strategy == 'random':
        nodes = list(G.nodes())
        np.random.shuffle(nodes)
    elif attack_strategy == 'degree':
        nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    elif attack_strategy == 'betweenness':
        bc = nx.betweenness_centrality(G)
        nodes = sorted(G.nodes(), key=lambda x: bc[x], reverse=True)
    else:
        raise ValueError("Unknown strategy")

    lcc_frac = []
    eff_vals = []
    G_work = G.copy()
    # 预计算初始效率（可选）
    # init_eff = approximate_global_efficiency(G_work)

    for p in fractions:
        remove_count = int(p * N)
        if remove_count == 0:
            if nx.is_connected(G_work):
                lcc = 1.0
            else:
                largest_cc = max(nx.connected_components(G_work), key=len)
                lcc = len(largest_cc) / N
            lcc_frac.append(lcc)
            eff_vals.append(approximate_global_efficiency(G_work))
            continue

        # 只删除新增部分节点
        prev_remove = int((p - fractions[1]) * N) if len(fractions) > 1 else 0
        new_remove = nodes[prev_remove:remove_count]
        G_work.remove_nodes_from(new_remove)

        # LCC
        if G_work.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(G_work), key=len)
            lcc = len(largest_cc) / N
        else:
            lcc = 0.0
        lcc_frac.append(lcc)

        # 近似效率
        if G_work.number_of_nodes() > 1:
            eff = approximate_global_efficiency(G_work, sample_ratio=0.05, min_samples=50)
        else:
            eff = 0.0
        eff_vals.append(eff)

    return lcc_frac, eff_vals

def process_city_resilience(city, G, strategies=['random', 'degree']):
    """
    单个城市的韧性计算（用于并行）
    """
    print(f"  处理城市: {city}")
    results = {}
    for strat in strategies:
        lcc, eff = simulate_attack_fast(G, attack_strategy=strat)
        results[f'{strat}_lcc'] = lcc
        results[f'{strat}_eff'] = eff
        # 计算AUC
        auc = np.trapz(eff, dx=0.05)  # fractions步长0.05
        results[f'{strat}_auc'] = auc
    return city, results
# ===================== 5. 耦合网络分析 =====================


# ===================== 网络可视化函数 =====================

def compute_and_cache_embeddings(city_graphs, dim=128, force_recompute=False):
    cache_dir = os.path.join(OUTPUT_DIR, "embeddings")
    os.makedirs(cache_dir, exist_ok=True)
    emb_file = os.path.join(cache_dir, "embeddings.npy")
    names_file = os.path.join(cache_dir, "city_names.txt")

    if not force_recompute and os.path.exists(emb_file) and os.path.exists(names_file):
        embeddings = np.load(emb_file)
        with open(names_file, 'r') as f:
            city_names = [line.strip() for line in f.readlines()]
        print("从缓存加载图嵌入向量")
        return embeddings, city_names

    # 否则重新计算
    embeddings, city_names = compute_graph_embeddings(city_graphs, dim)
    np.save(emb_file, embeddings)
    with open(names_file, 'w') as f:
        f.write('\n'.join(city_names))
    print("已缓存图嵌入向量")
    return embeddings, city_names



def plot_bottleneck_map(G, top_df, city_name, mode):
    """绘制瓶颈站点地图"""
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', width=0.3)
    nx.draw_networkx_nodes(G, pos, node_size=5, node_color='lightgray', alpha=0.5)
    top_pos = {node: pos[node] for node in top_df['node'] if node in pos}
    nx.draw_networkx_nodes(G, top_pos, node_size=50, node_color='red', alpha=0.9)
    plt.title(f"{city_name} {mode.upper()} Top Bottlenecks")
    plt.axis('off')
    plt.savefig(os.path.join(fig_dir, f"{city_name}_{mode}_bottlenecks.png"), dpi=150)
    plt.close()


def simulate_attack_with_cache(G, city_name, mode='bus', attack_strategy='degree',
                               fractions=np.linspace(0, 0.5, 11), force_recompute=False):
    """
    执行级联失效模拟，并将结果缓存。
    返回包含每一步状态的字典。
    """
    cache_dir = os.path.join(OUTPUT_DIR, "simulations")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{city_name}_{mode}_{attack_strategy}.pkl")

    if not force_recompute and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  从缓存加载 {city_name} {mode} {attack_strategy} 模拟结果")
        return data

    N = G.number_of_nodes()
    if N == 0:
        return None

    # 确定删除顺序
    if attack_strategy == 'random':
        nodes = list(G.nodes())
        np.random.shuffle(nodes)
    elif attack_strategy == 'degree':
        nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    elif attack_strategy == 'betweenness':
        bc = nx.betweenness_centrality(G, k=min(500, N))
        nodes = sorted(G.nodes(), key=lambda x: bc[x], reverse=True)
    else:
        raise ValueError("Unknown strategy")

    # 存储每一步的数据
    steps = []
    G_work = G.copy()

    for i, p in enumerate(fractions):
        remove_count = int(p * N)
        removed_this_step = []
        if i > 0:
            prev_remove = int(fractions[i-1] * N)
            removed_this_step = nodes[prev_remove:remove_count]
            G_work.remove_nodes_from(removed_this_step)

        # 计算 LCC
        if G_work.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(G_work), key=len)
            lcc = len(largest_cc) / N
        else:
            lcc = 0.0

        # 近似全局效率
        if G_work.number_of_nodes() > 1:
            eff = approximate_global_efficiency(G_work)
        else:
            eff = 0.0

        steps.append({
            'fraction': p,
            'removed_nodes': removed_this_step.copy(),
            'num_nodes_removed': len(removed_this_step),
            'lcc': lcc,
            'efficiency': eff,
            'remaining_nodes': G_work.number_of_nodes(),
            'remaining_edges': G_work.number_of_edges(),
        })

    result = {
        'city': city_name,
        'mode': mode,
        'strategy': attack_strategy,
        'total_nodes': N,
        'total_edges': G.number_of_edges(),
        'steps': steps,
        'removal_order': nodes,  # 完整删除顺序
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"  已缓存 {city_name} {mode} {attack_strategy} 模拟结果")

    return result

def build_and_cache_graph(city_name, mode='bus', force_rebuild=False):
    """
    构建网络图，并缓存为 GraphML 文件。
    若缓存存在且 force_rebuild=False，则直接加载。
    """
    cache_dir = os.path.join(OUTPUT_DIR, "graphs")
    os.makedirs(cache_dir, exist_ok=True)
    graph_file = os.path.join(cache_dir, f"{city_name}_{mode}.graphml")

    if not force_rebuild and os.path.exists(graph_file):
        print(f"  从缓存加载 {city_name} {mode} 图...")
        G = nx.read_graphml(graph_file)
        # GraphML 保存时会将坐标转为字符串，需转换回 float
        for n, data in G.nodes(data=True):
            if 'x' in data:
                data['x'] = float(data['x'])
            if 'y' in data:
                data['y'] = float(data['y'])
        return G

    # 实际构建
    G, _ = build_graph_from_shapefile(city_name, mode)
    if G is not None:
        # 确保属性可序列化
        nx.write_graphml(G, graph_file)
        print(f"  已缓存 {city_name} {mode} 图")
    return G

def plot_aggregated_resilience(cities, strategies, output_dir):
    """
    从缓存的模拟结果中读取数据，绘制所有城市的韧性曲线对比图。
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 准备绘图数据
    all_data = {}
    for city in cities:
        for strat in strategies:
            cache_file = os.path.join(output_dir, "simulations", f"{city}_bus_{strat}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                all_data[f"{city}_{strat}"] = data

    if not all_data:
        print("没有可用的模拟数据，跳过绘图。")
        return

    # 提取分数序列（统一使用0~0.5，11个点）
    fractions = np.linspace(0, 0.5, 11)

    # 绘制 LCC 曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for key, data in all_data.items():
        city = data['city']
        strat = data['strategy']
        lcc_vals = [step['lcc'] for step in data['steps']]
        ax = axes[0] if strat == 'random' else axes[1]
        ax.plot(fractions, lcc_vals, marker='o', markersize=3, label=city)

    axes[0].set_title('Random Attack (LCC)')
    axes[0].set_xlabel('Fraction of nodes removed')
    axes[0].set_ylabel('Largest Connected Component size')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Degree Attack (LCC)')
    axes[1].set_xlabel('Fraction of nodes removed')
    axes[1].set_ylabel('Largest Connected Component size')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "aggregated_resilience_lcc.png"), dpi=200)
    plt.close()
    print("聚合韧性曲线已保存。")

def analyze_coupled_with_cache(city_name, snap_threshold=300, force_recompute=False):
    """
    构建公交-地铁耦合网络，执行韧性对比分析，并缓存结果。
    """
    cache_dir = os.path.join(OUTPUT_DIR, "coupled")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{city_name}_coupled.pkl")

    if not force_recompute and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            result = pickle.load(f)
        print(f"  从缓存加载 {city_name} 耦合网络分析结果")
        return result

    # 构建单层和耦合网络
    G_bus = build_and_cache_graph(city_name, mode='bus')
    G_metro = build_and_cache_graph(city_name, mode='metro')
    if G_bus is None or G_metro is None:
        print(f"  {city_name} 缺少公交或地铁数据，跳过耦合分析")
        return None

    G_multi = build_multilayer_graph(city_name, snap_threshold)
    if G_multi is None:
        return None

    # 地铁节点列表
    metro_nodes_orig = list(G_metro.nodes())
    # 计算地铁节点在耦合网络中的介数中心性
    bc_multi = nx.betweenness_centrality(G_multi, k=min(500, G_multi.number_of_nodes()))
    bc_metro = {n: bc_multi.get(f"metro_{n}", 0) for n in metro_nodes_orig}
    sorted_metro = sorted(metro_nodes_orig, key=lambda x: bc_metro[x], reverse=True)

    fractions = np.linspace(0, 0.5, 11)
    lcc_multi = []
    lcc_metro = []
    eff_multi = []
    eff_metro = []

    N_metro = len(metro_nodes_orig)
    for p in fractions:
        remove_count = int(p * N_metro)
        removed = sorted_metro[:remove_count]

        # 耦合网络攻击
        G_multi_copy = G_multi.copy()
        G_multi_copy.remove_nodes_from([f"metro_{n}" for n in removed])
        if G_multi_copy.number_of_nodes() > 0:
            largest = max(nx.connected_components(G_multi_copy), key=len)
            lcc_multi.append(len(largest) / G_multi.number_of_nodes())
        else:
            lcc_multi.append(0.0)
        eff_multi.append(approximate_global_efficiency(G_multi_copy))

        # 单层地铁攻击
        G_metro_copy = G_metro.copy()
        G_metro_copy.remove_nodes_from(removed)
        if G_metro_copy.number_of_nodes() > 0:
            largest = max(nx.connected_components(G_metro_copy), key=len)
            lcc_metro.append(len(largest) / G_metro.number_of_nodes())
        else:
            lcc_metro.append(0.0)
        eff_metro.append(approximate_global_efficiency(G_metro_copy))

    result = {
        'city': city_name,
        'fractions': fractions.tolist(),
        'lcc_multi': lcc_multi,
        'lcc_metro': lcc_metro,
        'eff_multi': eff_multi,
        'eff_metro': eff_metro,
        'removal_order': sorted_metro,
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"  已缓存 {city_name} 耦合网络分析结果")

    # 绘制对比图
    plot_coupled_comparison(result, OUTPUT_DIR)
    return result

def plot_coupled_comparison(result, output_dir):
    """绘制耦合网络与单层网络的韧性对比图"""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    city = result['city']
    fractions = result['fractions']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fractions, result['lcc_multi'], 'o-', label='Multilayer (Bus+Metro)')
    axes[0].plot(fractions, result['lcc_metro'], 's-', label='Metro only')
    axes[0].set_title(f'{city} - LCC under Metro Node Removal')
    axes[0].set_xlabel('Fraction of metro nodes removed')
    axes[0].set_ylabel('LCC size (relative to total nodes)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(fractions, result['eff_multi'], 'o-', label='Multilayer')
    axes[1].plot(fractions, result['eff_metro'], 's-', label='Metro only')
    axes[1].set_title(f'{city} - Global Efficiency under Metro Node Removal')
    axes[1].set_xlabel('Fraction of metro nodes removed')
    axes[1].set_ylabel('Global Efficiency')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{city}_coupled_vs_single.png"), dpi=200)
    plt.close()
# ===================== 6. 主流程 =====================
def main():
    print("=" * 50)
    print("城市公共交通网络韧性评估与瓶颈识别（带缓存版）")
    print("=" * 50)

    # 创建必要目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sub in ["graphs", "simulations", "bottlenecks", "embeddings", "coupled", "figures"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    # 获取可用城市
    available_cities = [c for c in CITIES if os.path.isdir(os.path.join(BUS_SHAPEFILE_ROOT, c))]
    print(f"发现 {len(available_cities)} 个城市数据：{available_cities}")

    # 第一步：构建并缓存所有城市的公交图
    city_graphs = {}
    for city in tqdm(available_cities, desc="构建公交网络"):
        G_bus = build_and_cache_graph(city, mode='bus')
        if G_bus:
            city_graphs[city] = G_bus
    print(f"成功加载/构建 {len(city_graphs)} 个城市公交网络。")

    # 第二步：图嵌入与聚类（带缓存）
    if len(city_graphs) >= 3:
        embeddings, city_names = compute_and_cache_embeddings(city_graphs)
        cluster_labels = cluster_cities(embeddings, city_names)
    else:
        cluster_labels = None

    # 第三步：韧性模拟（并行 + 缓存）
    print("开始韧性评估（并行 + 缓存）...")
    strategies = ['random', 'degree']
    tasks = []
    for city, G in city_graphs.items():
        for strat in strategies:
            tasks.append((G, city, 'bus', strat))

    with mp.Pool(processes=100) as pool:
        list(tqdm(pool.starmap(simulate_attack_with_cache, tasks),
                  total=len(tasks), desc="韧性模拟"))

    # 第四步：瓶颈识别（带缓存）
    for city, G in tqdm(city_graphs.items(), desc="瓶颈识别"):
        identify_bottlenecks_with_cache(G, city, mode='bus', top_n=10)

    # 第五步：生成韧性曲线汇总图
    plot_aggregated_resilience(available_cities, strategies, OUTPUT_DIR)

    # 第六步：耦合网络分析（所有有地铁的城市）
    metro_cities = [c for c in available_cities if os.path.isdir(os.path.join(METRO_SHAPEFILE_ROOT, c))]
    if metro_cities:
        print(f"开始耦合网络分析，共 {len(metro_cities)} 个城市...")
        for city in tqdm(metro_cities, desc="耦合网络"):
            analyze_coupled_with_cache(city)
    else:
        print("没有发现地铁数据，跳过耦合网络分析。")

    print("=" * 50)
    print("全部完成！所有中间数据已缓存，结果保存在:", OUTPUT_DIR)

if __name__ == "__main__":
    main()