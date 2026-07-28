## ！！！按顺序依次完成下列内容！！！


### 0. 进入 conda 环境
```bash
conda activate simlingo
export PYTHONPATH=$PYTHONPATH:/home/kemove/ll/simlingo_liulei

```


### 1. 生成4通道结构化世界标签
```bash
python driving_structured_world_project/tools_bev/generate_driving_structured_world.py
```
```text
# 备注：需要修改的内容为driving_structured_world_project/configs/driving_structured_world.yaml
  input: /home/kemove/ll/simlingo_liulei/database/simlingo_v2_2026_07_21_16_16_03/data/simlingo
  recursive: true
```


#### 1.1 检查4通道结构化世界标签质量以用于优化网络【可选,网络优化阶段选择】
```bash
python driving_structured_world_project/tools_bev/analyze_future_interaction_grids.py
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


### 2. 四个通道的含义

```text
C0 Expert Route
专家参考路径上自车 footprint 的等权占用。

C1 Expert Ego Future
专家未来 10 帧自车 footprint 的等权占用。

C2 Expert-conditioned Primary Actor
复用 LG 反事实 actor 选择方法，但以与专家轨迹最接近的完整场景响应作为
专家条件响应，重新选择主要 actor，并绘制该 actor 未来 10 帧占用。

C4 Expert-conditioned Secondary Actor
在同一组专家条件反事实 actor 测试中排除主要 actor 后，仅保留具有独立、
足够强的边际因果作用，并具有充分未来帧支持且与专家轨迹时序接近的次要 actor。
```


### 3. 每条 route 下的输出

```text
<route>/
├── driving_future_interaction_grids/
│   └── 0015.npz
├── driving_expert_conditioned_actor_selection/
│   └── 0015.json.gz
└── driving_future_interaction_grids_debug/       # debug.save_debug=true 时
    └── 0015_future_interaction.png
```

NPZ 主标签形状：

```text
future_interaction_grid: [4, H, W], float32, [0, 1]
```

其中 `H,W` 与当前帧 `costmap` 完全一致。当前数据通常为 `256×256`。
