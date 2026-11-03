## ！！！按顺序依次完成下列内容！！！


### 0. 进入 conda 环境
```bash
conda activate simlingo
export PYTHONPATH=$PYTHONPATH:/home/kemove/ll/simlingo_liulei

```

### 1. 生成 cost map
```bash
python lg_waypoint_planner_project/tools_bev/generate_costmap_from_masks.py
```
```text
# 备注：需要修改的内容为lg_waypoint_planner_project/configs/simple_bev_collision_map.yaml
  input: /home/kemove/ll/simlingo_liulei/database/simlingo_v2_2026_07_21_16_16_03/data/simlingo
  recursive: true
```

### 2. 生成语言标签
```bash
python lg_waypoint_planner_project/tools_bev/run_language_grounded_waypoint_planner.py
```
```text
# 备注：需要修改的内容为lg_waypoint_planner_project/configs/language_grounded_waypoint.yaml
  input: /home/kemove/ll/simlingo_liulei/database/simlingo_v2_2026_07_21_16_16_03/data/simlingo
  recursive: true
```

### 3. 生成3通道结构化世界标签
```bash
python lg_waypoint_planner_project/tools_bev/generate_future_interaction_grids.py
```
```text
# 备注：需要修改的内容为lg_waypoint_planner_project/configs/future_interaction_grid.yaml
  input: /home/kemove/ll/simlingo_liulei/database/simlingo_v2_2026_07_21_16_16_03/data/simlingo
  recursive: true
```

#### 3.1 检查3通道结构化世界标签质量以用于优化网络【可选,网络优化阶段选择,暂时不再使用这个脚本】
```bash
python lg_waypoint_planner_project/tools_bev/analyze_future_interaction_grids.py
```
```text
# 备注：需要修改的内容为
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/kemove/ll/simlingo_liulei/database/simlingo_v2_2026_07_21_16_16_03/data/simlingo"),  # 需要修改的内容,数据集目录
        help=(
            "数据集根目录、单条 route 目录，或 future_interaction_grids 目录。"
        ),
    )
```

#### 3.2 全量标签质量检查
```bash
python lg_waypoint_planner_project/tools_bev/check_all_generated_labels.py \
  --input /root/autodl-tmp/database/simlingo_v2_2026_09_12_22_24_28/data/simlingo \
  --output /root/autodl-tmp/lg_driving_label_quality_report \
  --progress-every 2000
```

### 4. CVAA 对照实验

```bash
python lg_waypoint_planner_project/tools_bev/run_language_grounded_waypoint_planner_keyframes.py
```