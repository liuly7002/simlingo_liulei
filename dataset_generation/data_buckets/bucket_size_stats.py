import json

"""
函数作用: 计算并打印每个数据桶相对于总桶的相对比例(百分比格式),以便了解每个桶在整个数据集中的占比情况。
"""

# 1. 读取统计文件
file = 'database/bucketsv2_simlingo/buckets_stats.json'
with open(file, 'r') as f:
    data = json.load(f)

# 2. 计算每个桶的相对比例
bucket_relatives = {}
total = data['total']
for key in data.keys():
    if key == 'total':
        continue
    bucket_relatives[key] = data[key] / total

# 3. 打印结果(百分比格式)
for key in bucket_relatives.keys():
    print(f'{key}: {bucket_relatives[key]*100:.2f}%')