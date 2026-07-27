# Driving Structured World Project

该目录用于为普通 `Driving` 数据生成四通道结构化未来世界标签。当前只修改Driving生成流程，LG标签生成流程保持不变。目录与
`lg_waypoint_planner_project` 并列，**不会修改或覆盖原有 LG 标签生成工程**。

推荐放置结构：

```text
simlingo_liulei/
├── lg_waypoint_planner_project/
├── driving_structured_world_project/
├── simlingo_training/
└── database/
```

本工程将当前 Git 中的 `lg_waypoint_planner_project/tools_bev/lg_waypoint_planner`
作为只读依赖，以复用相同的场景解析、候选响应、反事实 actor 移除和因果评分逻辑。
生成结果全部保存到新的 Driving 专用目录。

## 四个通道

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

Driving标签现在按以下四通道顺序保存；后续接入训练时，网络读取逻辑也需要同步调整：

```text
selected_route_ego_footprint_occupancy
future_ego_footprint_occupancy
primary_causal_actor_future_footprint_occupancy
secondary_actor_future_footprint_occupancy
```

## actor 重新选择逻辑

1. 使用当前 LG 方法构造完整场景候选响应并完成安全/可行性评估；
2. 在有效候选中选择与专家未来轨迹最接近的响应，作为专家条件下的完整场景响应；
3. 对候选 actor 逐个执行场景移除；
4. 每次移除后重新诊断场景、重新生成候选并重新选择最小充分响应；
5. 将移除前后响应差异最大的、通过 LG 因果阈值的 actor 作为 C2；
6. 排除 C2 后，对其余actor执行严格C4筛选：因果分数阈值、相对主要actor
   分数比例、未来帧覆盖、与专家轨迹的时序距离，以及与主要actor反事实响应
   是否重复；
7. 若多个非主要actor产生相同的反事实意图、候选变体和作用向量，则将其视为
   无法唯一归因的等价响应组，整组不生成C4；
8. 只有全部条件通过且不存在归因歧义的最高分actor才作为C4，否则C4保持全零。

采用“专家匹配候选”而不是直接把专家轨迹塞入 LG 候选池，是为了让完整场景和
反事实场景均由同一套运动生成与评价机制产生，降低专家轨迹与规划器轨迹直接比较
带来的系统偏差。C1严格使用真实专家未来轨迹。原C3因正样本过少已从Driving标签中删除。

## C3删除说明

当前Driving标签只保存C0、C1、C2、C4。原C3
`time_aligned_primary_future_interaction`在安全专家轨迹中几乎始终为零，
会使辅助任务严重偏向全背景，因此先从Driving生成与统计流程中删除。LG版本将在
Driving流程稳定后再统一调整。

语义编号继续保留C4，但张量内部连续存储顺序为：

```text
index 0 -> C0
index 1 -> C1
index 2 -> C2
index 3 -> C4
```

## C4严格选择条件

C4候选必须同时满足：

- 反事实测试有效并通过因果接受；
- 因果分数不低于`min_causal_score`；
- 因果分数不低于主要actor分数的指定比例；
- 未来actor状态达到最小帧数；
- 与专家轨迹在同一未来时刻的最小中心距离不超过阈值；
- 不能与主要actor产生数值上完全相同的反事实响应；
- 不能属于由多个非主要actor共同形成的等价反事实响应组。

若没有候选通过，C4保持全零，这不会令整张标签无效。

## 运行

在仓库根目录执行：

```bash
python driving_structured_world_project/tools_bev/generate_driving_structured_world.py
```

指定配置：

```bash
python driving_structured_world_project/tools_bev/generate_driving_structured_world.py \
  --config driving_structured_world_project/configs/driving_structured_world.yaml
```

单帧检查可直接修改 YAML 中的：

```yaml
run:
  input: /path/to/one/route
  recursive: false
  frame: "0015"

debug:
  save_debug: true
```

## 每条 route 下的输出

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

## 有效性规则

Driving 标签不会继承 LG 的 `risk_label_valid`。它独立检查：

- 专家参考路径是否存在；
- 专家未来 10 帧是否完整；
- 是否存在与专家轨迹足够接近的有效 LG 响应候选；
- 主要 actor 存在时，是否至少找到一个未来时刻的 actor 状态；
- 四通道 shape、有限性及 `[0,1]` 数值范围是否合法。

没有主要 actor 或次要 actor 本身不会令标签失效，对应通道保持全零。主要 actor
已被确认但未来 10 帧完全找不到时，标签置为无效。次要 actor 未来缺失不会使整张
标签失效，C4 保持全零并在元数据中记录原因。

## 重要配置

`expert_match` 决定完整场景候选是否足够接近专家轨迹。若大量标签因
`expert_match_threshold_failed` 失效，应先统计 ADE/FDE 分布，再调整阈值，不能直接
关闭检查，否则 C2/C4 可能不再真正代表专家驾驶策略下的关键对象。
