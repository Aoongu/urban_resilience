import pickle
import numpy as np
import pandas as pd
import os
import glob

# 定义数据路径
input_dir = 'output/simulations'
output_csv = 'simulation_results_summary.csv'

# 用于存储结果的列表
results = []

# 获取目录下所有 pkl 文件
# 假设文件名格式为: {City}_bus_{Strategy}.pkl
file_paths = glob.glob(os.path.join(input_dir, '*.pkl'))

print(f"正在处理 {len(file_paths)} 个模拟文件...")

for path in file_paths:
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # 提取基础信息
        # 如果 pkl 中没写 city/strategy，则从文件名解析
        filename = os.path.basename(path)
        city = data.get('city', filename.split('_')[0])
        strategy = data.get('strategy', filename.split('_')[-1].replace('.pkl', ''))

        # 提取数值
        fractions = [step['fraction'] for step in data['steps']]
        lcc_vals = [step['lcc'] for step in data['steps']]
        eff_vals = [step['efficiency'] for step in data['steps']]

        # 计算 AUC
        # 使用 np.trapezoid (注意：若numpy版本较低请改回 np.trapz)
        lcc_auc = np.trapezoid(lcc_vals, x=fractions)
        eff_auc = np.trapezoid(eff_vals, x=fractions)

        # 存入列表
        results.append({
            'City': city,
            'Strategy': strategy,
            'LCC_AUC': round(lcc_auc, 6),
            'EFF_AUC': round(eff_auc, 6),
            'FileName': filename
        })

    except Exception as e:
        print(f"处理文件 {path} 时出错: {e}")

# 创建 DataFrame 并保存
df = pd.DataFrame(results)

# 排序（按城市和策略排序，方便查看）
df = df.sort_values(by=['City', 'Strategy'])

# 保存到 CSV
df.to_csv(input_dir+output_csv, index=False, encoding='utf_8_sig')

print(f"计算完成！结果已保存至: {input_dir+output_csv}")
print(df.head())
