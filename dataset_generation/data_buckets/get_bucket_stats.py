import pickle
import ujson

"""
函数作用: 从 pkl 中读取 bucket 索引信息，统计各驾驶场景样本数量，并生成训练用的分布统计 json 文件。
"""

bucket_path = 'database/bucketsv2_simlingo/buckets_paths.pkl'
save_path = 'database/bucketsv2_simlingo/buckets_stats.json'
buckets_stats = {}

with open(bucket_path, 'rb') as f:
    buckets = pickle.load(f)

for key, value in buckets.items():
    buckets_stats[key] = len(value)

# find unique bucket values
unique_bucket_values = set()
for key, value in buckets.items():
    unique_bucket_values.update(value)

buckets_stats['total'] = len(unique_bucket_values)

# save buckets stats as json
with open(f'{save_path}', 'w') as f:
    ujson.dump(buckets_stats, f, indent=4)