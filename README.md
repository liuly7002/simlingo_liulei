
# 测试阶段

## 一、数据集收集&处理

### 1. 收集数据

使用如下命令来收集数据：

```bash
python data_collection.py
```

```text
备注：需要修改的内容如下：
code_root   = r"/home/kemove/ll/simlingo"                # 项目根目录
carla_root  = "/home/kemove/ll/simlingo/carla0915"       # Carla根位置
```

### 2. 数据清理

#### 2.1 使用命令一来进行第一轮(共两轮)数据清理：

```bash
python dataset_generation/delete_failed_runs.py
```

```text
# 备注：需要修改的内容如下：
# dataset_path = '/home/kemove/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo'
```

#### 2.2 使用命令二来进行第二轮(共两轮)数据清理：

```bash
python dataset_generation/delete_infraction_routes.py
```

```text
# 备注：需要修改的内容如下：
# data_save_root = '/home/kemove/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo'
```

## 二、关于 data buckets 的操作

关于 data buckets 的内容都是在 /dataset_generation/data_buckets/ 目录下，
在做端到端自动驾驶时，这种 bucket 机制很关键，因为它解决"数据不均衡导致模型偏科问题"!!!

### 1. 【必选】首先，必须先使用以下命令将数据生成 data buckets

```bash
python dataset_generation/data_buckets/carla_get_buckets.py
```

```text
备注：需要修改的内容如下：
data_path = '/home/kemove/ll/simlingo/database/simlingo_v2_2026_02_28'  # 数据集目录
save_path = f'/home/kemove/ll/simlingo/database/simlingo_v2_2026_02_28/bucketsv2_simlingo'     # 存储桶结果目录
```

### 2. 【可选】使用以下命令计算并打印每个数据桶相对于总桶的相对比例(百分比格式),以便了解每个桶在整个数据集中的占比情况。

```bash
python dataset_generation/data_buckets/bucket_size_stats.py
```

### 3. 【可选】使用以下命令从 pkl 中读取 bucket 索引信息，统计各驾驶场景样本数量，并生成训练用的分布统计 json 文件。

```bash
python dataset_generation/data_buckets/get_bucket_stats.py
```

### 4. 【可选】使用以下命令将存储桶路径的 pkl 文件转换为 csv 文件，方便查看和分析。

```bash
python dataset_generation/data_buckets/pkl2csv.py
```

## 三、关于 "Language" 数据的相关操作

### 1. 生成 drivelm 类型 .json.gz 文件

```bash
cd simlingo
```

```bash
conda activate simlingo
```

```bash
export PYTHONPATH=$PYTHONPATH:/home/kemove/ll/simlingo_liulei
```

```bash
python dataset_generation/language_labels/drivelm/carla_vqa_generator_main.py
```

```text
# 备注：需要修改的内容为
    path_group.add_argument('--data-directory', type=str, default='database/simlingo_v2_2026_07_03_19_20_14',
                            help='Data directory containing the dataset')  # 数据集根目录
    path_group.add_argument('--output-directory', type=str, default='database/simlingo_v2_2026_07_03_19_20_14/drivelm',
                            help='Output directory for the vqa-graph')     # VQA graph 保存位置
    path_group.add_argument('--output-graph-examples-directory', type=str, default='database/simlingo_v2_2026_07_03_19_20_14/drivelm',
                            help='Output directory for examples of the vqa-graph')
```

#### 1.1 对 .json.gz 文件内容进行分析验证【可选，目前没必要验证】

```bash
# 第一轮分析【风险推理链 QA 元信息完整性验证脚本, 非重点】
python a_validate/validate_risk_meta.py
```

```bash
# 第二轮分析【风险图 QA 连接关系与链路完整性验证脚本, 非重点】
python a_validate/validate_graph_structure.py
```

```bash
# 第三轮分析【风险图 QA 语义一致性验证脚本, 重点】
python a_validate/validate_risk_consistency.py
```

```bash
# 第四轮分析【重点】
python a_validate/validate_action_consistency.py
```

### 2. 生成 Commentary 类型 .json.gz 文件

```bash
python dataset_generation/language_labels/commentary/carla_commentary_generator_main.py
```

```text
# 备注：需要修改的内容为：
    path_group.add_argument('--data-directory', type=str, default='database/simlingo_v2_2026_07_17_23_23_22',
                            help='Data directory containing the dataset')  # 数据集根目录
    path_group.add_argument('--output-directory', type=str, default='database/simlingo_v2_2026_07_17_23_23_22/commentary',
                            help='Output directory for the vqa-graph')  # 保存Commentary标签的目录
    path_group.add_argument('--output-examples-directory', type=str, default='database/simlingo_v2_2026_07_17_23_23_22/commentary',
                            help='Output directory for examples of the vqa-graph')
```

### 3. 生成 Dreamer 类型 .json.gz 文件

```bash
python dataset_generation/dreamer_data/dreamer_generator.py
```

```text
# 备注：需要修改的内容有：
    base_folder = 'database'   # 数据集根目录
    dataset_name = 'simlingo_v2_2026_07_03_19_20_14'  # 数据集名称
```

### 4. 生成lg相关内容【创新点相关】

```text
请按照"lg_waypoint_planner_project/Readme.md"内容进行
```

### 5. 生成driving相关内容【创新点相关】

```text
请按照"driving_structured_world_project/README.md"内容进行
```


## 四、关于 "训练"
### 1. 开始训练
```bash
cd simlingo_liulei
```
```bash
export PYTHONPATH=$PYTHONPATH:/home/kemove/ll/simlingo_liulei
```
```bash
./train_simlingo_seed1.sh
```
### 2. 网页查看训练结果
```
# simlingo_liulei 根目录下执行
wandb sync ./outputs/2026_05_04_16_53_04_simlingo_seed1/wandb/offline-run-20260504_165419-2026_05_04_16_53_04_simlingo_seed1
```