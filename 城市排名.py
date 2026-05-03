import pandas as pd
import os


def generate_city_ranking():
    # 1. 读取站点级瓶颈数据
    df = pd.read_csv('output/all_cities_bottleneck.csv')

    # 2. 定义城市级聚合逻辑
    def calculate_city_metrics(group):
        # 提取该城市排名前 5 的顶级瓶颈
        top5 = group.sort_values('robust_score', ascending=False).head(5)

        return pd.Series({
            '候选节点总数': len(group),
            'Top5_平均综合破坏力': top5['robust_score'].mean(),
            'Top5_平均效率损失': top5['efficiency_loss'].mean(),
            'Top5_平均连通碎裂度(LCC_loss)': top5['lcc_loss'].mean(),
            'Top5_平均阻断率(Reachability_loss)': top5['reachability_loss'].mean(),
            '全局_平均网络连通碎裂度': group['lcc_loss'].mean(),
            '核心枢纽多模态依赖度': top5['interlayer_ratio'].mean()
        })

    # 3. 按城市分组计算
    print("正在聚合城市级特征...")
    city_ranking = df.groupby('city').apply(calculate_city_metrics).reset_index()

    # 4. 排序：以“Top5_平均综合破坏力”作为主排序指标（降序排列，越靠前越脆弱）
    city_ranking = city_ranking.sort_values('Top5_平均综合破坏力', ascending=False).reset_index(drop=True)

    # 增加名次列
    city_ranking.insert(0, '脆弱性排名', city_ranking.index + 1)

    # 5. 保存结果
    output_path = 'output/city_resilience_ranking.csv'
    city_ranking.to_csv(output_path, index=False, encoding='utf_8_sig')

    print(f"✅ 城市排名计算完成！已保存至 {output_path}")
    print("\n=== 最脆弱的 Top 5 城市 ===")
    print(city_ranking[['脆弱性排名', 'city', 'Top5_平均综合破坏力', 'Top5_平均效率损失']].head(5))

    print("\n=== 最具韧性的 Top 5 城市 ===")
    print(city_ranking[['脆弱性排名', 'city', 'Top5_平均综合破坏力', 'Top5_平均效率损失']].tail(5))


if __name__ == "__main__":
    generate_city_ranking()
