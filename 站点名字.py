import os
import geopandas as gpd

# 1. 设置你想查询的城市和对应的 node_id 列表
city_name = "Hohhot"  # 你可以替换成任何你想查询的城市
target_nodes = ["BV11401675"]

print(f"\n🔍 --- 开始全网（地铁+公交）搜寻 {city_name} 的目标站点 --- \n")

for node_id in target_nodes:
    # 去掉可能存在的网络前缀，只保留核心 ID (防止传进来的是 bus_BV... 这种格式)
    real_node_id = str(node_id).split('_')[-1]
    found = False

    # 2. 依次遍历 'metro' (地铁) 和 'bus' (公交) 两个数据源
    for mode in ['metro', 'bus']:
        # 拼接路径 (兼容首字母大写或全小写的情况)
        dbf_path = f"./dataset/{mode}/shapefiles/{city_name}/{city_name}_{mode}_stops_unique.dbf"
        if not os.path.exists(dbf_path):
            dbf_path = f"./dataset/{mode}/shapefiles/{city_name}/{city_name.lower()}_{mode}_stops_unique.dbf"

        # 如果文件存在，就进去翻找
        if os.path.exists(dbf_path):
            try:
                gdf = gpd.read_file(dbf_path, encoding='utf-8')
                # 在 stop_id 这一列中寻找匹配项
                result = gdf[gdf['stop_id'] == real_node_id]

                # 如果 result 不是空的，说明找到了！
                if not result.empty:
                    station_name = result['stop_cn'].values[0]

                    # ==========================================================
                    # 👉 按照您的要求，在这里打印出 节点、找到的地方 和 中文名
                    # ==========================================================
                    print(f"✅ 成功命中！")
                    print(f"   🔹 节点 ID: {node_id}")
                    print(f"   🔹 交通类型: {'🚌 公交系统' if mode == 'bus' else '🚇 地铁系统'}")
                    print(f"   🔹 数据来源: {dbf_path}")
                    print(f"   🔹 真实站名: 【 {station_name} 】")
                    print(f"   -------------------------------------------------")

                    found = True
                    break  # 既然在这个网络里找到了，就不去另一个网络找了，直接跳出 mode 循环

            except Exception as e:
                print(f"⚠️ 读取 {dbf_path} 失败: {e}")

    # 如果两个文件夹都翻完了还没找到
    if not found:
        print(f"❌ 查找失败！节点 ID: {node_id} 在 {city_name} 的公交和地铁库中均未找到。请核对数据集是否完整。")
        print(f"   -------------------------------------------------\n")

print("🎯 搜寻完毕！")
