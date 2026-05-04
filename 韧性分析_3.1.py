"""
城市公共交通网络韧性评估与瓶颈识别
基于CPTOND-2025数据集（Shapefile格式）
要求：geopandas, networkx, karateclub, scikit-learn, matplotlib, seaborn, tqdm等
"""

import multiprocessing as mp
import os
import pickle
import random
import warnings
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 配置参数 =====================
DATA_DIR = "./dataset"          # 数据根目录，包含bus/shapefiles和metro/shapefiles
OUTPUT_DIR = "./output_3.1"         # 输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 公交与地铁的城市文件夹模式（根据实际文件命名调整）
# 假设城市文件夹在 bus/shapefiles/城市名/ 下，文件名为 拼音_bus_stops.shp 等
BUS_SHAPEFILE_ROOT = os.path.join(DATA_DIR, "bus", "shapefiles")
METRO_SHAPEFILE_ROOT = os.path.join(DATA_DIR, "metro", "shapefiles")

# 城市列表（可从文件夹自动获取，这里手动指定示例城市）
CITIES = [d for d in os.listdir(METRO_SHAPEFILE_ROOT)  # 获取所有有地铁的城市
          if os.path.isdir(os.path.join(METRO_SHAPEFILE_ROOT, d))]

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
    efficiency = total_inv_dist / len(pairs)
    return efficiency


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
            eff = approximate_global_efficiency(G_work, sample_ratio=0.0, min_samples=1000)
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


# def plot_aggregated_resilience(cities, strategies, output_dir, mode):
#     """
#     从缓存的模拟结果中读取数据，绘制所有城市的韧性曲线对比图。
#     """
#     fig_dir = os.path.join(output_dir, "figures")
#     os.makedirs(fig_dir, exist_ok=True)

#     # 准备绘图数据
#     all_data = {}
#     for city in cities:
#         for strat in strategies:
#             cache_file = os.path.join(output_dir, "simulations", f"{city}_{mode}_{strat}.pkl")
#             if os.path.exists(cache_file):
#                 with open(cache_file, 'rb') as f:
#                     data = pickle.load(f)
#                 all_data[f"{city}_{strat}"] = data

#     if not all_data:
#         print("没有可用的模拟数据，跳过绘图。")
#         return

#     # 提取分数序列（统一使用0~0.5，11个点）
#     fractions = np.linspace(0, 0.5, 11)

#     # 绘制 LCC 曲线
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#     for key, data in all_data.items():
#         city = data['city']
#         strat = data['strategy']
#         lcc_vals = [step['lcc'] for step in data['steps']]
#         ax = axes[0] if strat == 'random' else axes[1]
#         ax.plot(fractions, lcc_vals, marker='o', markersize=3, label=city)

#     axes[0].set_title('Random Attack (LCC)')
#     axes[0].set_xlabel('Fraction of nodes removed')
#     axes[0].set_ylabel('Largest Connected Component size')
#     axes[0].legend()
#     axes[0].grid(alpha=0.3)

#     axes[1].set_title('Degree Attack (LCC)')
#     axes[1].set_xlabel('Fraction of nodes removed')
#     axes[1].set_ylabel('Largest Connected Component size')
#     axes[1].legend()
#     axes[1].grid(alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(os.path.join(fig_dir, "aggregated_resilience_lcc.png"), dpi=400)
#     plt.close()
#     print("聚合韧性曲线已保存。")

def plot_aggregated_resilience(cities, strategies, output_dir, mode):
    """
    从缓存的模拟结果中读取数据，绘制所有城市的韧性曲线对比图。
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. 准备并组织绘图数据
    all_data = {}
    for city in cities:
        for strat in strategies:
            cache_file = os.path.join(output_dir, "simulations", f"{city}_{mode}_{strat}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                if city not in all_data:
                    all_data[city] = {}
                all_data[city][strat] = data

    if not all_data:
        print("没有可用的模拟数据，跳过绘图。")
        return

    # 2. 筛选特点鲜明的几条线
    # (通过计算Degree攻击下的LCC面积，代表城市的网络韧性得分，以此来挑选最具代表性的几个城市)
    city_scores = {}
    for city, strats_data in all_data.items():
        if 'degree' in strats_data:
            lcc_vals = [step['lcc'] for step in strats_data['degree']['steps']]
            city_scores[city] = sum(lcc_vals)  # 简单的曲线下面积求和

    if not city_scores:
        print("缺少degree策略的数据，无法筛选。")
        return

    # 按韧性得分从大到小排序
    sorted_cities = sorted(city_scores.keys(), key=lambda c: city_scores[c], reverse=True)

    # 挑选出最强(0)、最弱(-1)，以及中间分布的几个城市（总共取5个特点鲜明的城市）
    if len(sorted_cities) > 5:
        selected_indices = [
            0,
            len(sorted_cities)//4,
            len(sorted_cities)//2,
            3*len(sorted_cities)//4,
            len(sorted_cities)-1
        ]
        selected_cities = [sorted_cities[i] for i in selected_indices]
    else:
        selected_cities = sorted_cities

    # 3. 提取分数序列（统一使用0~0.5，11个点）
    fractions = np.linspace(0, 0.5, 11)

    # 4. 绘制 LCC 曲线 (修改为3个子图)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 建立策略与子图的映射关系
    strat_to_ax = {
        'random': axes[0],
        'degree': axes[1],
        'betweenness': axes[2]
    }

    # 仅绘制筛选出的代表性城市
    for city in selected_cities:
        for strat in strategies:
            if strat in all_data[city]:
                data = all_data[city][strat]
                lcc_vals = [step['lcc'] for step in data['steps']]
                ax = strat_to_ax[strat]
                # 绘制曲线
                ax.plot(fractions, lcc_vals, marker='o', markersize=4, linewidth=1.5, label=f"{city}")

    # 5. 美化每个子图
    for strat, ax in strat_to_ax.items():
        ax.set_title(f'{strat.capitalize()} Attack (LCC)')
        ax.set_xlabel('Fraction of nodes removed')
        ax.set_ylabel('Largest Connected Component size')

        # 获取图例并去重
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "aggregated_resilience_lcc.png"), dpi=400)
    plt.close()

    print(f"聚合韧性曲线已保存。筛选出的代表性城市为: {selected_cities}")

# ===================== 6. 主流程 =====================


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

    # 第三步：韧性模拟（并行 + 缓存）
    print("开始韧性评估（并行 + 缓存）...")
    strategies = ['random', 'degree', 'betweenness']
    tasks = []
    for city, G in city_graphs.items():
        for strat in strategies:
            tasks.append((G, city, 'multi', strat))

    with mp.Pool(processes=None) as pool:
        list(tqdm(pool.starmap(simulate_attack_with_cache, tasks),
                  total=len(tasks), desc="韧性模拟"))

    plot_aggregated_resilience(metro_cities, strategies, OUTPUT_DIR, 'multi')


if __name__ == "__main__":
    main()
