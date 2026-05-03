import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import warnings
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pyproj import Transformer
from sklearn.decomposition import PCA
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
CITIES = [d for d in os.listdir(METRO_SHAPEFILE_ROOT)#获取所有有地铁的城市
          if os.path.isdir(os.path.join(METRO_SHAPEFILE_ROOT, d))]

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
        G = nx.relabel_nodes(G, lambda x: str(x))
        # GraphML 保存时会将坐标转为字符串，需转换回 float
        for n, data in G.nodes(data=True):
            if 'x' in data:
                data['x'] = float(data['x'])
            if 'y' in data:
                data['y'] = float(data['y'])
        # ✅ 新增：修复边属性
        for u, v, data in G.edges(data=True):
            if 'length' in data:
                data['length'] = float(data['length'])
        return G

    # 实际构建
    result = build_graph_from_shapefile(city_name, mode)
    if result is None or result[0] is None:
        print(f"警告：{city_name} {mode} 图构建失败")
        return None
    G, _ = result
    print(f"城市：{city_name}, mode: {mode}, 节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    isolated = [n for n in G if G.degree(n) == 0]
    print(f"孤立节点数: {len(isolated)}")
    if G.number_of_edges() == 0:
        print("⚠️ 警告：图中没有任何边！请检查 segments 文件的读取逻辑或字段名。")
    if G is not None:
        # 确保属性可序列化
        nx.write_graphml(G, graph_file)
        print(f"  已缓存 {city_name} {mode} 图")
    return G

# =========================================================
# 1. 基础工具：图指标
# =========================================================

def _edge_weight_or_none(G, weight='length'):
    """
    判断图中是否存在可用权重属性。
    若不存在，则返回 None，NetworkX 将使用无权最短路。
    """
    for _, _, data in G.edges(data=True):
        if weight in data:
            return weight
    return None


def global_efficiency_weighted(G, weight='length'):
    """
    计算加权全局效率：
    E = 1 / (n(n-1)) * sum_{i!=j} 1 / d_ij
    对于不可达节点对，贡献为0。
    """
    n = G.number_of_nodes()
    if n < 2:
        return 0.0

    weight_attr = _edge_weight_or_none(G, weight)

    # 用单源最短路，避免重复计算
    nodes = list(G.nodes())
    inv_sum = 0.0

    for i, u in enumerate(nodes):
        if weight_attr is None:
            dist_u = nx.single_source_shortest_path_length(G, u)
        else:
            dist_u = nx.single_source_dijkstra_path_length(G, u, weight=weight_attr)

        for v in nodes[i + 1:]:
            d = dist_u.get(v, np.inf)
            if np.isfinite(d) and d > 0:
                inv_sum += 1.0 / d

    return 2.0 * inv_sum / (n * (n - 1))


def largest_connected_component_ratio(G):
    """
    最大连通分量占比
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    if G.is_directed():
        comps = nx.weakly_connected_components(G)
    else:
        comps = nx.connected_components(G)
    lcc = max(comps, key=len, default=set())
    return len(lcc) / n


def average_shortest_path_on_lcc(G, weight='length'):
    """
    仅在最大连通分量上计算平均最短路。
    若图过小则返回 nan。
    """
    if G.number_of_nodes() < 2:
        return np.nan

    if G.is_directed():
        comps = list(nx.weakly_connected_components(G))
    else:
        comps = list(nx.connected_components(G))

    if not comps:
        return np.nan

    lcc_nodes = max(comps, key=len)
    H = G.subgraph(lcc_nodes).copy()
    if H.number_of_nodes() < 2:
        return np.nan

    weight_attr = _edge_weight_or_none(H, weight)
    try:
        return nx.average_shortest_path_length(H, weight=weight_attr)
    except Exception:
        return np.nan


# =========================================================
# 2. 候选节点筛选
# =========================================================

def candidate_nodes_by_centrality(G, top_k=100, weight='length', seed=None):
    """
    通过多个中心性指标筛选候选节点。
    返回候选节点列表与完整中心性表。
    """
    weight_attr = _edge_weight_or_none(G, weight)

    deg = dict(G.degree())
    # 介数中心性：大图可改为近似算法
    btw = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()), normalized=True, weight=weight_attr, seed=seed)
    clo = nx.closeness_centrality(G, distance=weight_attr)
    pr = nx.pagerank(G, weight=weight_attr)

    # k-core
    try:
        core_num = nx.core_number(G)
    except Exception:
        core_num = {n: 0 for n in G.nodes()}

    # 割点
    if G.is_directed():
        articulation = set()
    else:
        articulation = set(nx.articulation_points(G))

    df = pd.DataFrame({
        'node': list(G.nodes()),
        'degree': [deg[n] for n in G.nodes()],
        'betweenness': [btw.get(n, 0.0) for n in G.nodes()],
        'closeness': [clo.get(n, 0.0) for n in G.nodes()],
        'pagerank': [pr.get(n, 0.0) for n in G.nodes()],
        'core_number': [core_num.get(n, 0) for n in G.nodes()],
        'is_articulation': [1 if n in articulation else 0 for n in G.nodes()],
    })

    # 标准化并做中心性共识分数
    feat_cols = ['degree', 'betweenness', 'closeness', 'pagerank', 'core_number', 'is_articulation']
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    Xs = StandardScaler().fit_transform(X)
    df['centrality_consensus'] = PCA(n_components=1).fit_transform(Xs).ravel()

    df = df.sort_values('centrality_consensus', ascending=False).reset_index(drop=True)
    candidates = df.head(min(top_k, len(df)))['node'].tolist()
    return candidates, df


# =========================================================
# 3. 单节点删除敏感度
# =========================================================

def sample_od_pairs(nodes, sample_size=2000, seed=42):
    """
    从节点集合中随机抽取 OD 对。
    """
    rng = np.random.default_rng(seed)
    nodes = list(nodes)
    n = len(nodes)
    if n < 2:
        return []

    max_pairs = n * (n - 1) // 2
    sample_size = min(sample_size, max_pairs)

    pairs = set()
    while len(pairs) < sample_size:
        u, v = rng.choice(nodes, size=2, replace=False)
        if u != v:
            a, b = (u, v) if str(u) < str(v) else (v, u)
            pairs.add((a, b))
    return list(pairs)


def pairwise_efficiency(G, pairs, weight='length'):
    """
    对给定 OD 对集合，计算平均 1/d。
    不可达则记为 0。
    """
    if not pairs:
        return 0.0

    weight_attr = _edge_weight_or_none(G, weight)
    total = 0.0

    # 为每个源点做一次单源最短路
    pairs_by_source = {}
    for u, v in pairs:
        pairs_by_source.setdefault(u, []).append(v)

    for u, targets in pairs_by_source.items():
        if u not in G:
            continue
        if weight_attr is None:
            dist_u = nx.single_source_shortest_path_length(G, u)
        else:
            dist_u = nx.single_source_dijkstra_path_length(G, u, weight=weight_attr)

        for v in targets:
            d = dist_u.get(v, np.inf)
            if np.isfinite(d) and d > 0:
                total += 1.0 / d

    return total / len(pairs)


def node_removal_impact_fast(G, node, base_pairs, pre_eff, pre_reachable, reachable_pairs, weight='length'):
    """
    评估删除单个节点后的网络影响（重用原图预计算结果）。

    参数：
        G: networkx.Graph，原图
        node: 待删除的节点
        base_pairs: list of tuple，全局 OD 对集合（原图的样本）
        pre_eff: float，原图上 base_pairs 的平均效率 (1/d)
        pre_reachable: int，原图上 base_pairs 中可达的 OD 对数量
        reachable_pairs: set，原图上可达的 OD 对集合 (u,v)，用于快速判断
        weight: str，边的权重属性名

    返回：
        dict 或 None（若节点不存在）
    """
    if node not in G:
        return None

    # 复制图并删除节点
    H = G.copy(as_view=True)
    H = H.copy()
    H.remove_node(node)

    # 过滤掉被删节点所在的 OD 对
    post_pairs = [(u, v) for (u, v) in base_pairs if u in H and v in H]
    if not post_pairs:
        # 所有样本对都受影响，效率完全损失
        return {
            'node': node,
            'pre_eff': pre_eff,
            'post_eff': 0.0,
            'efficiency_loss': pre_eff,
            'lcc_loss': 1.0,
            'reachability_loss': 1.0,
        }

    # 删除后图的效率
    post_eff = pairwise_efficiency(H, post_pairs, weight=weight)

    # 最大连通分量占比变化
    pre_lcc = largest_connected_component_ratio(G)
    post_lcc = largest_connected_component_ratio(H)
    lcc_loss = pre_lcc - post_lcc

    # 可达性损失：检查原图可达的对在删除后是否变得不可达
    unreachable = 0
    for u, v in post_pairs:
        pair = (u, v) if str(u) < str(v) else (v, u)
        if pair not in reachable_pairs:
            continue  # 原图就不可达，不计入损失
        # 测试在 H 中是否可达
        try:
            if weight is None:
                nx.shortest_path_length(H, u, v)
            else:
                nx.shortest_path_length(H, u, v, weight=weight)
        except nx.NetworkXNoPath:
            unreachable += 1

    reachability_loss = unreachable / pre_reachable if pre_reachable > 0 else 0.0

    return {
        'node': node,
        'pre_eff': pre_eff,
        'post_eff': post_eff,
        'efficiency_loss': pre_eff - post_eff,
        'lcc_loss': lcc_loss,
        'reachability_loss': reachability_loss,
    }
# =========================================================
# 4. 综合瓶颈评分
# =========================================================

def bottleneck_analysis(
    G,
    top_k_candidates=100,
    sample_pairs=3000,
    weight='length',
    seed=42,
    include_layer_features=True
):
    """
    生成瓶颈节点排序表。
    综合：中心性共识 + 删除敏感度 + 换乘/跨层特征
    """
    nodes = list(G.nodes())
    base_pairs = sample_od_pairs(nodes, sample_size=sample_pairs, seed=seed)

    # 原图效率
    pre_eff_global = pairwise_efficiency(G, base_pairs, weight=weight)

    # 原图可达性信息
    weight_attr = _edge_weight_or_none(G, weight)
    reachable_pairs = set()
    for u, v in base_pairs:
        try:
            if weight_attr is None:
                nx.shortest_path_length(G, u, v)
            else:
                nx.shortest_path_length(G, u, v, weight=weight_attr)
            # 统一存储为有序对 (较小节点, 较大节点)
            pair = (u, v) if str(u) < str(v) else (v, u)
            if pair in reachable_pairs:
                continue
            reachable_pairs.add(pair)
        except nx.NetworkXNoPath:
            pass
    pre_reachable = len(reachable_pairs)

    # ---------- 候选节点筛选 ----------
    candidates, centrality_df = candidate_nodes_by_centrality(G, top_k=top_k_candidates, weight=weight, seed=seed)
    centrality_df = centrality_df.copy()
    centrality_df['degree_rank'] = centrality_df['degree'].rank(ascending=False)

    centrality_map = centrality_df.set_index('node')['centrality_consensus'].to_dict()
    degree_rank_map = centrality_df.set_index('node')['degree_rank'].to_dict()


    # ---------- 逐个节点评估（使用快速版） ----------
    rows = []
    for node in tqdm(candidates, desc='节点删除敏感度'):
        impact = node_removal_impact_fast(
            G, node,
            base_pairs=base_pairs,
            pre_eff=pre_eff_global,
            pre_reachable=pre_reachable,
            reachable_pairs=reachable_pairs,
            weight=weight
        )
        if impact is None:
            continue

        row = {
            'node': node,
            **impact,
            'centrality_consensus': centrality_map.get(node, 0.0),
            'degree_rank': degree_rank_map.get(node, 0.0),
        }

        # 层间特征（如适用）
        if include_layer_features:
            layer_edges = 0
            inter_edges = 0
            for _, _, data in G.edges(node, data=True):
                layer_edges += 1
                if data.get('layer') == 'inter':
                    inter_edges += 1
            row['interlayer_ratio'] = inter_edges / max(layer_edges, 1)
        else:
            row['interlayer_ratio'] = 0.0

        rows.append(row)

    # 后续评分、标准化等保持不变...
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 处理极端值
    for col in ['efficiency_loss', 'lcc_loss', 'reachability_loss', 'centrality_consensus', 'interlayer_ratio']:
        if col not in df.columns:
            df[col] = 0.0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 标准化后加权求和
    score_cols = ['efficiency_loss', 'lcc_loss', 'reachability_loss', 'centrality_consensus', 'interlayer_ratio']
    X = df[score_cols].values
    Xs = StandardScaler().fit_transform(X)

    # 权重可以按论文需要调
    w = Xs.var(axis=0)
    w = w / w.sum()
    df['bottleneck_score'] = Xs @ w

    # 稳健性修正：波动越大，可信度越低
    # 这里给一个简单形式，可在 bootstrap 中替换为更精细的 std
    df['robust_score'] = df['bottleneck_score']

    df = df.sort_values('robust_score', ascending=False).reset_index(drop=True)
    return df


# =========================================================
# 5. Bootstrap 稳定性检验
# =========================================================

def bootstrap_bottleneck_stability(
    G,
    top_k_candidates=100,
    sample_pairs=3000,
    n_boot=10,
    weight='length',
    seed=42
):
    """
    通过重复抽样 OD 对，评估瓶颈排名稳定性。
    输出：
    - mean_score
    - std_score
    - stability_score
    """
    all_scores = {}

    for b in tqdm(range(n_boot), desc='Bootstrap 稳定性检验'):
        df = bottleneck_analysis(
            G,
            top_k_candidates=top_k_candidates,
            sample_pairs=sample_pairs,
            weight=weight,
            seed=seed + b
        )
        if df.empty:
            continue

        for _, row in df.iterrows():
            node = row['node']
            all_scores.setdefault(node, []).append(row['bottleneck_score'])

    rows = []
    for node, scores in all_scores.items():
        scores = np.array(scores, dtype=float)
        rows.append({
            'node': node,
            'mean_score': scores.mean(),
            'std_score': scores.std(ddof=1) if len(scores) > 1 else 0.0,
            'boot_times': len(scores),
            'stability_score': scores.mean() / (scores.std(ddof=1) + 1e-8) if len(scores) > 1 else scores.mean()
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values('stability_score', ascending=False).reset_index(drop=True)
    return out


def analyze_single_city(city, G=None, top_k=80, sample_pairs=2000, boot_pairs=1500, n_boot=15):
    """对一个城市的耦合网络进行瓶颈分析和稳定性检验，返回结果DataFrame"""
    print(f"开始城市 {city} 的瓶颈节点识别...")
    if G is None:
        G = build_multilayer_graph(city)
    # 瓶颈分析
    bottleneck_df = bottleneck_analysis(
        G, top_k_candidates=top_k, sample_pairs=sample_pairs, weight='length'
    )

    # 稳定性检验
    stability_df = bootstrap_bottleneck_stability(
        G, top_k_candidates=top_k, sample_pairs=boot_pairs, n_boot=n_boot, weight='length'
    )

    # 添加城市名（便于后续合并）
    bottleneck_df['city'] = city
    stability_df['city'] = city

    return bottleneck_df, stability_df

# =========================================================
# 6. 示例调用
# =========================================================
# 假设你的图已经构建好了，叫 G_multi
# 例如：
def main():
    print("=" * 50)
    print("城市公共交通网络韧性评估与瓶颈识别（带缓存版）")
    print("=" * 50)

    # 创建必要目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sub in ["graphs", "simulations", "figures"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    # 获取可用城市
    available_cities = [c for c in CITIES if os.path.isdir(os.path.join(BUS_SHAPEFILE_ROOT, c))]
    print(f"发现 {len(available_cities)} 个城市数据：{available_cities}")

    # 第一步：构建并缓存所有城市的公交图
    city_graphs: dict[Any, Any] = {}
    for city in tqdm(available_cities, desc="构建耦合网络"):
        G_multi = build_multilayer_graph(city)
        if G_multi:
            city_graphs[city] = G_multi
    print(f"成功加载/构建 {len(city_graphs)} 个城市耦合网络。")
    metro_cities = [c for c in available_cities if os.path.isdir(os.path.join(METRO_SHAPEFILE_ROOT, c))]
    # 并行分析参数
    top_k = 50
    sample_pairs = 500
    boot_pairs = 1500
    n_boot = 10
    max_workers = max(1, min(len(city_graphs), (os.cpu_count() or 8) - 4))

    all_bottleneck = []
    all_stability = []

    # 使用进程池提交任务
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                analyze_single_city, city, None,
                top_k, sample_pairs, boot_pairs, n_boot
            ): city
            for city, G in city_graphs.items()
        }

        # 带进度条收集结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="并行瓶颈分析"):
            city = futures[future]
            try:
                bd, sd = future.result()
                all_bottleneck.append(bd)
                all_stability.append(sd)

                # 即时保存单城市结果（可选，防止中途崩溃丢失）
                bd.to_csv(os.path.join(OUTPUT_DIR, f"{city}_bottleneck.csv"), index=False)
                sd.to_csv(os.path.join(OUTPUT_DIR, f"{city}_stability.csv"), index=False)
            except Exception as e:
                print(f"城市 {city} 分析失败: {e}")

    # 合并所有城市结果并保存汇总文件
    if all_bottleneck:
        master_bottleneck = pd.concat(all_bottleneck, ignore_index=True)
        master_bottleneck.to_csv(os.path.join(OUTPUT_DIR, "all_cities_bottleneck.csv"), index=False)
        master_bottleneck.sort_values('robust_score', ascending=False) \
            .to_csv(os.path.join(OUTPUT_DIR, "bottleneck_ranking_all.csv"), index=False)

    if all_stability:
        master_stability = pd.concat(all_stability, ignore_index=True)
        master_stability.to_csv(os.path.join(OUTPUT_DIR, "all_cities_stability.csv"), index=False)
        master_stability.sort_values('stability_score', ascending=False) \
            .to_csv(os.path.join(OUTPUT_DIR, "stability_ranking_all.csv"), index=False)

    print("全部完成！结果已保存至 ./output/ 目录")


if __name__ == "__main__":
    main()