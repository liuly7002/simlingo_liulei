import pickle
import pandas as pd
import os

"""
函数作用: 将存储桶路径的 pkl 文件转换为 csv 文件，方便查看和分析。
"""

# ==== 1. 输入路径 ====
pkl_path = "database/bucketsv2_simlingo/buckets_paths.pkl"   # 输入 pkl 文件路径
csv_path = "database/bucketsv2_simlingo/buckets_paths.csv"   # 输出 csv 文件路径

# ==== 2. 读取 pkl ====
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("数据类型:", type(data))


# ==== 3. 通用展开函数 ====
def flatten_data(data):
    rows = []

    # case 1: dict
    if isinstance(data, dict):
        for key, value in data.items():

            # value 是 list（最常见：bucket -> 路径列表）
            if isinstance(value, list):
                for v in value:
                    rows.append({
                        "bucket": key,
                        "value": v
                    })

            # value 是 dict
            elif isinstance(value, dict):
                for k2, v2 in value.items():
                    rows.append({
                        "bucket": key,
                        "sub_key": k2,
                        "value": v2
                    })

            else:
                rows.append({
                    "bucket": key,
                    "value": value
                })

    # case 2: list
    elif isinstance(data, list):
        for i, item in enumerate(data):
            rows.append({
                "index": i,
                "value": item
            })

    else:
        rows.append({"value": data})

    return rows


# ==== 4. 转 dataframe ====
rows = flatten_data(data)
df = pd.DataFrame(rows)

# ==== 5. 保存 csv ====
df.to_csv(csv_path, index=False)

print(f"已保存到: {csv_path}")
print(df.head())