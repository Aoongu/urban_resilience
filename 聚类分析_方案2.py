"""
城市公共交通网络特征聚类分析（方案一：统计特征聚类）
基于网络描述性指标，可解释性强
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
from scipy.spatial import KDTree, ConvexHull
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 配置参数 =====================
DATA_DIR = "./dataset"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUS_SHAPEFILE_ROOT = os.path.join(DATA_DIR, "bus", "shapefiles")
METRO_SHAPEFILE_ROOT = os.path.join(DATA_DIR, "metro", "shapefiles")

CITIES = [d for d in os.listdir(METRO_SHAPEFILE_ROOT)
          if os.path.isdir(os.path.join(METRO_SHAPEFILE_ROOT, d))]

# 聚类方案标识
SCHEME_NAME = "feature_based"   # 方案名称，用于文件前缀

# ===================== 图构建函数（复用原逻辑，略作精简） =====================
def build_graph_from_shapefile(city_name, mode='bus'):
    if mode == 'bus':
        root = BUS_SHAPEFILE_ROOT
        stops_file = os.path.join(root, city_name, f"{city_name.lower()}_bus_stops.shp")
        routes_file = os.path.join(root, city_name, f"{city_name.lower()}_bus_routes.shp")
    else:
        root = METRO_SHAPEFILE_ROOT
        stops_file = os.path.join(root, city_name, f"{city_name.lower()}_metro_stops.shp")
        routes_file = os.path.join(root, city_name, f"{city_name.lower()}_metro_routes.shp")

    if not os.path.exists(stops_file) or not os.path.exists(routes_file):
        return None

    stops_gdf = gpd.read_file(stops_file)
    routes_gdf = gpd.read_file(routes_file)

    G = nx.Graph()
    node_pos = {}
    for idx, row in stops_gdf.iterrows():
        node_id = row['stop_id'] if 'stop_id' in row.index else f"{city_name}_{mode}_{idx}"
        if row.geometry is None:
            continue
        lon, lat = row.geometry.x, row.geometry.y
        G.add_node(node_id, x=lon, y=lat, mode=mode)
        node_pos[node_id] = (lon, lat)

    segment_file = os.path.join(root, city_name, f"{city_name.lower()}_{mode}_segments.shp")
    if os.path.exists(segment_file):
        seg_gdf = gpd.read_file(segment_file)
        for _, seg in seg_gdf.iterrows():
            u = seg['s_stopid']
            v = seg['e_stopid']
            if u in G and v in G:
                dist = seg.geometry.length * 111000
                G.add_edge(u, v, length=dist, mode=mode)
    else:
        print(f"警告：{city_name} {mode} 缺少segment文件，边未添加")
    return G

def build_multilayer_graph(city_name, snap_threshold=300):
    G_bus = build_and_cache_graph(city_name, mode='bus')
    G_metro = build_and_cache_graph(city_name, mode='metro')
    if G_bus is None or G_metro is None:
        return None
    G_multi = nx.Graph()
    for node, data in G_bus.nodes(data=True):
        G_multi.add_node(f"bus_{node}", layer='bus', x=data['x'], y=data['y'])
    for u, v, data in G_bus.edges(data=True):
        G_multi.add_edge(f"bus_{u}", f"bus_{v}", layer='bus', length=data.get('length', 0))
    for node, data in G_metro.nodes(data=True):
        G_multi.add_node(f"metro_{node}", layer='metro', x=data['x'], y=data['y'])
    for u, v, data in G_metro.edges(data=True):
        G_multi.add_edge(f"metro_{u}", f"metro_{v}", layer='metro', length=data.get('length', 0))

    bus_coords = np.array([[G_bus.nodes[n]['x'], G_bus.nodes[n]['y']] for n in G_bus.nodes()])
    metro_coords = np.array([[G_metro.nodes[n]['x'], G_metro.nodes[n]['y']] for n in G_metro.nodes()])
    bus_ids = list(G_bus.nodes())
    metro_ids = list(G_metro.nodes())
    tree = KDTree(bus_coords)
    for i, mcoord in enumerate(metro_coords):
        dist, idx = tree.query(mcoord)
        if dist * 111000 < snap_threshold:
            G_multi.add_edge(f"bus_{bus_ids[idx]}", f"metro_{metro_ids[i]}",
                             layer='inter', length=dist * 111000)
    return G_multi

def build_and_cache_graph(city_name, mode='bus', force_rebuild=False):
    cache_dir = os.path.join(OUTPUT_DIR, "graphs")
    os.makedirs(cache_dir, exist_ok=True)
    graph_file = os.path.join(cache_dir, f"{city_name}_{mode}.graphml")
    if not force_rebuild and os.path.exists(graph_file):
        G = nx.read_graphml(graph_file)
        for n, data in G.nodes(data=True):
            if 'x' in data: data['x'] = float(data['x'])
            if 'y' in data: data['y'] = float(data['y'])
        return G
    G = build_graph_from_shapefile(city_name, mode)
    if G is not None:
        nx.write_graphml(G, graph_file)
    return G

# ===================== 网络特征计算 =====================
def compute_network_features(G, network_type='bus'):
    """计算单个网络的多种特征，返回字典"""
    if G is None or G.number_of_nodes() == 0:
        return None
    feat = {}
    N = G.number_of_nodes()
    E = G.number_of_edges()
    feat['nodes'] = N
    feat['edges'] = E
    feat['avg_degree'] = 2.0 * E / N if N > 0 else 0

    # 连通分量分析（取最大连通分量计算某些指标）
    if nx.is_connected(G):
        G_lcc = G
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(largest_cc).copy()
    feat['lcc_nodes'] = G_lcc.number_of_nodes()
    feat['lcc_edges'] = G_lcc.number_of_edges()
    feat['lcc_fraction'] = feat['lcc_nodes'] / N if N > 0 else 0

    # 平均最短路径长度和直径（在最大连通分量上计算，避免无穷大）
    if G_lcc.number_of_nodes() > 1:
        try:
            feat['avg_shortest_path'] = nx.average_shortest_path_length(G_lcc)
            feat['diameter'] = nx.diameter(G_lcc)
        except:
            feat['avg_shortest_path'] = np.inf
            feat['diameter'] = np.inf
    else:
        feat['avg_shortest_path'] = 0
        feat['diameter'] = 0

    # 全局效率（近似）
    eff_sum = 0.0
    n_lcc = G_lcc.number_of_nodes()
    if n_lcc > 1:
        for u in G_lcc.nodes():
            lengths = nx.single_source_shortest_path_length(G_lcc, u)
            for v, d in lengths.items():
                if u < v and d > 0:
                    eff_sum += 1.0 / d
        feat['global_efficiency'] = eff_sum / (n_lcc * (n_lcc - 1))
    else:
        feat['global_efficiency'] = 0

    # 聚类系数
    feat['avg_clustering'] = nx.average_clustering(G)

    # 度相关性（同配系数）
    feat['degree_assortativity'] = nx.degree_assortativity_coefficient(G)

    # 社区模块度（使用贪婪算法，若无社区则置零）
    try:
        communities = list(nx.algorithms.community.greedy_modularity_communities(G))
        if communities:
            mod = nx.algorithms.community.quality.modularity(G, communities)
            feat['modularity'] = mod
            feat['num_communities'] = len(communities)
        else:
            feat['modularity'] = 0
            feat['num_communities'] = 0
    except:
        feat['modularity'] = 0
        feat['num_communities'] = 0

    # 圈数（cyclomatic number: E - N + P，P为连通分量数）
    feat['cyclomatic_number'] = E - N + nx.number_connected_components(G)

    # 空间特征（如果有坐标）
    if all('x' in G.nodes[n] for n in G.nodes()):
        coords = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in G.nodes()])
        if len(coords) >= 3:
            try:
                hull = ConvexHull(coords)
                feat['convex_hull_area'] = hull.volume   # 对于2D，volume即面积
            except:
                feat['convex_hull_area'] = 0
        else:
            feat['convex_hull_area'] = 0
        # 平均边长度
        lengths = []
        for u, v, data in G.edges(data=True):
            if 'length' in data:
                lengths.append(data['length'])
            else:
                # 计算端点距离（近似）
                dx = G.nodes[u]['x'] - G.nodes[v]['x']
                dy = G.nodes[u]['y'] - G.nodes[v]['y']
                lengths.append(np.sqrt(dx**2 + dy**2) * 111000)
        feat['avg_edge_length'] = np.mean(lengths) if lengths else 0
        feat['edge_length_std'] = np.std(lengths) if lengths else 0
    else:
        feat['convex_hull_area'] = 0
        feat['avg_edge_length'] = 0
        feat['edge_length_std'] = 0

    # 针对耦合网络的额外特征
    if network_type == 'multi':
        # 换乘边比例
        inter_edges = sum(1 for _, _, d in G.edges(data=True) if d.get('layer') == 'inter')
        feat['inter_ratio'] = inter_edges / E if E > 0 else 0
        # 地铁节点比例
        metro_nodes = sum(1 for _, d in G.nodes(data=True) if d.get('layer') == 'metro')
        feat['metro_ratio'] = metro_nodes / N if N > 0 else 0

    return feat

def compute_all_features(graph_dict, network_type='bus'):
    """为多个城市计算特征表"""
    features = {}
    for city, G in tqdm(graph_dict.items(), desc=f"计算 {network_type} 特征"):
        feat = compute_network_features(G, network_type)
        if feat:
            features[city] = feat
    df = pd.DataFrame.from_dict(features, orient='index')
    df.index.name = 'city'
    return df

# ===================== 聚类与可视化 =====================
def cluster_by_features(df, network_type, n_clusters=None):
    """基于特征进行 KMeans 和层次聚类，返回标签"""
    if df.empty or len(df) < 2:
        return None
    # 移除可能无限大的列
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')
    if df_clean.empty:
        return None
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean)

    if n_clusters is None:
        n_clusters = min(4, len(X))
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels_kmeans = kmeans.fit_predict(X)

    # 层次聚类
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels_agg = agg.fit_predict(X)

    # 保存聚类结果
    result_df = pd.DataFrame({
        'city': df_clean.index,
        'kmeans': labels_kmeans,
        'hierarchical': labels_agg
    })
    out_csv = os.path.join(OUTPUT_DIR, f"{SCHEME_NAME}_{network_type}_clusters.csv")
    result_df.to_csv(out_csv, index=False)

    # ---------- 可视化 ----------
    # 1. t-SNE 降维散点图（用 KMeans 标签着色）
    perplexity = min(30, len(X)-1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_kmeans, cmap='Set1', s=100)
    for i, name in enumerate(df_clean.index):
        plt.annotate(name, (X_2d[i, 0], X_2d[i, 1]), fontsize=9)
    plt.title(f"{network_type} 网络特征 t-SNE 投影 ({SCHEME_NAME})")
    plt.colorbar(scatter, label='KMeans Cluster')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{SCHEME_NAME}_{network_type}_tsne.png"), dpi=300)
    plt.close()

    # 2. PCA 可视化（更忠实反映全局结构）
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='Set1', s=100)
    for i, name in enumerate(df_clean.index):
        plt.annotate(name, (X_pca[i, 0], X_pca[i, 1]), fontsize=9)
    plt.title(f"{network_type} 网络特征 PCA 投影 ({SCHEME_NAME})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.colorbar(scatter, label='KMeans Cluster')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{SCHEME_NAME}_{network_type}_pca.png"), dpi=300)
    plt.close()

    # 3. 特征热力图（聚类排序后）
    ordered_idx = np.argsort(labels_kmeans)
    sorted_features = df_clean.iloc[ordered_idx]
    sorted_labels = labels_kmeans[ordered_idx]
    plt.figure(figsize=(12, max(6, len(sorted_features)*0.3)))
    sns.heatmap(StandardScaler().fit_transform(sorted_features), 
                yticklabels=sorted_features.index, 
                xticklabels=sorted_features.columns,
                cmap='viridis', cbar_kws={'label': 'Z-score'})
    plt.title(f"{network_type} 网络特征热力图（按 KMeans 聚类排序）")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{SCHEME_NAME}_{network_type}_heatmap.png"), dpi=300)
    plt.close()

    # 4. 平行坐标图（观察各类别的特征分布）
    # 将特征标准化后，添加聚类标签列
    df_norm = pd.DataFrame(X, columns=df_clean.columns, index=df_clean.index)
    df_norm['cluster'] = labels_kmeans.astype(str)
    # 为简化绘图，选择前8个特征（如有更多）
    plot_cols = list(df_clean.columns[:8])
    if len(plot_cols) > 2:
        plt.figure(figsize=(10, 5))
        pd.plotting.parallel_coordinates(df_norm.reset_index(), 'cluster', 
                                         cols=plot_cols, colormap='Set1', alpha=0.7)
        plt.title(f"{network_type} 平行坐标图（标准化后特征）")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{SCHEME_NAME}_{network_type}_parallel.png"), dpi=300)
        plt.close()

    # 5. 聚类中心（雷达/柱状图）——展示每类的特征均值
    cluster_centers = kmeans.cluster_centers_
    # 用原始量纲的均值也可
    means = pd.DataFrame(X, columns=df_clean.columns)
    means['cluster'] = labels_kmeans
    cluster_means = means.groupby('cluster').mean()
    # 绘制雷达图（需极坐标）
    feature_names = cluster_means.columns
    n_features = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, row in cluster_means.iterrows():
        values = row.values.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=9)
    ax.set_title(f"{network_type} 聚类中心雷达图 ({SCHEME_NAME})")
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{SCHEME_NAME}_{network_type}_radar.png"), dpi=300)
    plt.close()

    return labels_kmeans, kmeans

# ===================== 主流程 =====================
def main():
    print("=" * 50)
    print(f"网络特征聚类分析（{SCHEME_NAME}）")
    print("=" * 50)
    for sub in ["graphs", "figures"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    available_cities = [c for c in CITIES
                        if os.path.isdir(os.path.join(BUS_SHAPEFILE_ROOT, c))
                        and os.path.isdir(os.path.join(METRO_SHAPEFILE_ROOT, c))]
    print(f"有效城市：{available_cities}")

    # 构建网络图集
    bus_graphs = {}
    metro_graphs = {}
    multi_graphs = {}
    for city in tqdm(available_cities, desc="构建网络"):
        G_bus = build_and_cache_graph(city, mode='bus')
        G_metro = build_and_cache_graph(city, mode='metro')
        if G_bus:
            bus_graphs[city] = G_bus
        if G_metro:
            metro_graphs[city] = G_metro
        if G_bus and G_metro:
            G_multi = build_multilayer_graph(city)
            if G_multi:
                multi_graphs[city] = G_multi

    # 对三种网络分别进行特征聚类
    for net_type, graph_dict in [("公交", bus_graphs), ("地铁", metro_graphs), ("耦合", multi_graphs)]:
        if len(graph_dict) < 2:
            print(f"{net_type}网络样本不足，跳过")
            continue
        df = compute_all_features(graph_dict, network_type=net_type)
        # 保存特征表
        df.to_csv(os.path.join(OUTPUT_DIR, f"{SCHEME_NAME}_{net_type}_features.csv"))
        print(f"{net_type}网络特征表：\n{df.describe()}\n")
        cluster_by_features(df, network_type=net_type)

    print("所有分析完成！结果保存在:", OUTPUT_DIR)

if __name__ == "__main__":
    main()