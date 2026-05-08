
# 测试阶段

## 一、关于 "Driving" 数据的相关操作

### 1. 收集数据

使用如下命令来收集数据：
```
python data_collection.py
```
备注：需要修改的内容如下：
```
code_root   = r"/home/liulei/ll/simlingo"                         # 项目根目录
carla_root  = "/home/liulei/ll/simlingo/software/carla0915"       # Carla根位置
```

### 2. 筛选数据

首先, 使用命令一来清理不良数据：
```
python dataset_generation/delete_failed_runs.py
```
备注：需要修改的内容如下：
```
dataset_path = '/home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo'
```
然后, 使用命令二来清理不良数据：
```
python dataset_generation/delete_infraction_routes.py
```
备注：需要修改的内容如下：
```
data_save_root = '/home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo'
```

## 二、关于 data buckets 的操作

关于 data buckets 的内容都是在 /dataset_generation/data_buckets/ 目录下，
在做端到端自动驾驶时，这种 bucket 机制很关键，因为它解决"数据不均衡导致模型偏科问题"!!!

1. 【必选】首先，必须先使用以下命令将数据生成 data buckets

```
python dataset_generation/data_buckets/carla_get_buckets.py
```
备注：需要修改的内容如下：
```
data_path = '/home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28'  # 数据集目录
save_path = f'/home/liulei/ll/simlingo/database/bucketsv2_simlingo'     # 存储桶结果目录
```

2. 【可选】使用以下命令计算并打印每个数据桶相对于总桶的相对比例(百分比格式),以便了解每个桶在整个数据集中的占比情况。

```
python dataset_generation/data_buckets/bucket_size_stats.py
```

3. 【可选】使用以下命令从 pkl 中读取 bucket 索引信息，统计各驾驶场景样本数量，并生成训练用的分布统计 json 文件。

```
python dataset_generation/data_buckets/get_bucket_stats.py
```

4. 【可选】使用以下命令将存储桶路径的 pkl 文件转换为 csv 文件，方便查看和分析。

```
python dataset_generation/data_buckets/pkl2csv.py
```

## 三、关于 "Language" 数据的相关操作

1. 生成 drivelm 类型 .json.gz 文件
```
cd simlingo & conda activate simlingo & export PYTHONPATH=$PYTHONPATH:/root/simlingo
python dataset_generation/language_labels/drivelm/carla_vqa_generator_main.py
```
2. 对 .json.gz 文件内容进行分析验证
```
# 第一轮分析【风险推理链 QA 元信息完整性验证脚本, 非重点】
python a_validate/validate_risk_meta.py

# 第二轮分析【风险图 QA 连接关系与链路完整性验证脚本, 非重点】
python a_validate/validate_graph_structure.py

# 第三轮分析【风险图 QA 语义一致性验证脚本, 重点】
python a_validate/validate_risk_consistency.py

# 第四轮分析【重点】
python a_validate/validate_action_consistency.py
```

## 四、关于 "训练"
1. 开始训练
```
cd simlingo & conda activate simlingo & export PYTHONPATH=$PYTHONPATH:/root/simlingo
./train_simlingo_seed1.sh
```
2. 网页查看训练结果
```
# simlingo 根目录下执行
wandb sync ./outputs/2026_05_04_16_53_04_simlingo_seed1/wandb/offline-run-20260504_165419-2026_05_04_16_53_04_simlingo_seed1
```