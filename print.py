import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi
import geopandas as gpd

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
bot_df = pd.read_csv('output_bottleneck/all_cities_bottleneck.csv')
stab_df = pd.read_csv('output_bottleneck/all_cities_stability.csv')

# ================= 新增：站点名称查询与缓存系统 =================
# 全局缓存，避免重复读取同一个城市的 shapefile，极大提升出图速度
_station_name_cache = {}


def get_station_name(city, node_id):
    """根据城市名和节点ID，从 dbf 文件中查找真实的中文站名"""
    # 提取真实的节点 ID (去除数据集中可能存在的 bus_ 或 metro_ 前缀)
    real_node_id = str(node_id).split('_')[-1]

    # 如果缓存中还没有这个城市的数据，去读取它
    if city not in _station_name_cache:
        _station_name_cache[city] = {}
        # 同时尝试读取地铁和公交的 dbf 文件，建立完整的 ID->名称 映射字典
        for mode in ['metro', 'bus']:
            # 兼容大小写命名习惯 (先试原名，再试全小写)
            dbf_path = f"./dataset/{mode}/shapefiles/{city}/{city}_{mode}_stops_unique.dbf"
            if not os.path.exists(dbf_path):
                dbf_path = f"./dataset/{mode}/shapefiles/{city}/{city.lower()}_{mode}_stops_unique.dbf"

            if os.path.exists(dbf_path):
                try:
                    # 强制使用 utf-8 编码读取
                    gdf = gpd.read_file(dbf_path, encoding='utf-8')
                    for _, row in gdf.iterrows():
                        _station_name_cache[city][row['stop_id']] = row['stop_cn']
                except Exception as e:
                    print(f"⚠️ 读取 {dbf_path} 失败: {e}")

    # 从缓存中查找站名，如果找不到（可能是数据缺失或路径不对），则退化返回原短 ID
    return _station_name_cache[city].get(real_node_id, real_node_id)
# ================================================================


def plot_stability_quadrant(top_n_labels=12):
    """绘制 Bootstrap 稳定性四象限图，并标注绝对瓶颈区的极端节点"""
    merged = pd.merge(bot_df, stab_df, on=['city', 'node'])

    # 稍微增大画布尺寸，给文字留出空间
    plt.figure(figsize=(12, 9))
    sns.scatterplot(
        data=merged, x='mean_score', y='stability_score',
        hue='city', alpha=0.6, legend=False, s=50
    )

    # 画象限分割线 (以中位数划线)
    x_median = merged['mean_score'].median()
    y_median = merged['stability_score'].median()

    plt.axvline(x=x_median, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=y_median, color='red', linestyle='--', alpha=0.5)

    # 把 y 轴的乘数调小一点，让文字往下走，避开最高的那个点
    plt.text(merged['mean_score'].max() * 0.75, merged['stability_score'].max() * 0.85,
             '绝对瓶颈区\n(高破坏+高稳定)', fontsize=12, color='darkred', weight='bold')

    # ================= 高破坏高稳定点标注逻辑 =================
    # 1. 筛选出位于第一象限的点
    top_right = merged[(merged['mean_score'] > x_median) & (merged['stability_score'] > y_median)].copy()

    # 2. 构造综合极值分数 (破坏力和稳定性的归一化相加)
    top_right['combined_score'] = (top_right['mean_score'] / top_right['mean_score'].max()) + \
                                  (top_right['stability_score'] / top_right['stability_score'].max())

    # 3. 提取排名前 N 的极值点
    top_nodes = top_right.sort_values('combined_score', ascending=False).head(top_n_labels)

    # 4. 执行标注 (使用真实的站点名称)
    try:
        from adjustText import adjust_text
        texts = []
        for _, row in top_nodes.iterrows():
            # 这里调用新的函数获取真实中文名
            station_name = get_station_name(row['city'], row['node'])
            label = f"{row['city']}-{station_name}"

            texts.append(plt.text(row['mean_score'], row['stability_score'], label,
                                  fontsize=9, color='black', weight='bold'))

        # 自动调整文字位置，并添加指向性箭头
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.7))
        print("✅ 使用 adjustText 库成功优化标签防重叠布局。")

    except ImportError:
        print("⚠️ 未找到 adjustText 库。使用基础 matplotlib 标注，可能会有少许重叠。")
        print("💡 提示：运行 'pip install adjustText' 可获得更好的论文排版效果。")
        for _, row in top_nodes.iterrows():
            station_name = get_station_name(row['city'], row['node'])
            label = f"{row['city']}-{station_name}"

            plt.annotate(label,
                         (row['mean_score'], row['stability_score']),
                         textcoords="offset points",
                         xytext=(8, 8),
                         ha='left',
                         fontsize=8,
                         weight='bold',
                         arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.7))
    # ================================================================

    plt.title("各城市瓶颈站点 Bootstrap 稳定性检验四象限图", fontsize=16)
    plt.xlabel("破坏力均值 (Mean Score)", fontsize=13)
    plt.ylabel("统计稳定性得分 (Stability Score = Mean/Std)", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.savefig('Fig_Stability_Quadrant.png', dpi=400, bbox_inches='tight')  # 提高分辨率为 400 dpi
    plt.close()
    print("生成: Fig_Stability_Quadrant.png")


def plot_radar_chart(city_name="Hohhot", top_n=3):
    """绘制特定城市 Top 瓶颈的雷达图"""
    city_data = bot_df[bot_df['city'] == city_name].head(top_n)
    if city_data.empty:
        return

    categories = ['efficiency_loss', 'lcc_loss', 'reachability_loss', 'centrality_consensus']
    labels = ['全局效率损失', '连通性损失(LCC)', '网络阻断率', '中心性共识']

    # 归一化处理（为了雷达图好看）
    for col in categories:
        if city_data[col].max() != 0:
            city_data[col] = city_data[col] / city_data[col].max()

    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 设置雷达图样式
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], labels, size=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "Max"], color="grey", size=8)
    plt.ylim(0, 1.1)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (_, row) in enumerate(city_data.iterrows()):
        values = row[categories].values.flatten().tolist()
        values += values[:1]

        # 这里调用新的函数获取真实中文名
        station_name = get_station_name(city_name, row['node'])

        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=f"{station_name}", color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.15)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f"{city_name} 市 Top {top_n} 瓶颈站点致灾机理画像", size=15, y=1.1)
    plt.savefig(f'Fig_Radar_{city_name}.png', dpi=400, bbox_inches='tight')
    plt.close()
    print(f"生成: Fig_Radar_{city_name}.png")


# 执行出图
# 你可以解除下面这行的注释来生成全新的散点图
plot_stability_quadrant(top_n_labels=12)
plot_radar_chart("Hohhot", 3)
