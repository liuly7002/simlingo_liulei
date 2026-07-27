# Driving Structured World Project

该目录用于为普通 `Driving` 数据生成五通道结构化未来世界标签。目录与
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

## 五个通道

```text
C0 Expert Route
专家参考路径上自车 footprint 的等权占用。

C1 Expert Ego Future
专家未来 10 帧自车 footprint 的等权占用。

C2 Expert-conditioned Primary Actor
复用 LG 反事实 actor 选择方法，但以与专家轨迹最接近的完整场景响应作为
专家条件响应，重新选择主要 actor，并绘制该 actor 未来 10 帧占用。

C3 Expert–Primary Interaction
专家自车与 Driving 主要 actor 在同一未来时刻的 footprint 交互区域，
逐帧计算后等权平均。

C4 Expert-conditioned Secondary Actor
在同一组专家条件反事实 actor 测试中，排除主要 actor 后，重新选择
因果影响最大的次要 actor，并绘制其未来 10 帧占用。
```

标签通道名称和顺序与当前网络读取逻辑保持一致：

```text
selected_route_ego_footprint_occupancy
future_ego_footprint_occupancy
primary_causal_actor_future_footprint_occupancy
time_aligned_primary_future_interaction
secondary_actor_future_footprint_occupancy
```

## actor 重新选择逻辑

1. 使用当前 LG 方法构造完整场景候选响应并完成安全/可行性评估；
2. 在有效候选中选择与专家未来轨迹最接近的响应，作为专家条件下的完整场景响应；
3. 对候选 actor 逐个执行场景移除；
4. 每次移除后重新诊断场景、重新生成候选并重新选择最小充分响应；
5. 将移除前后响应差异最大的、通过 LG 因果阈值的 actor 作为 C2；
6. 排除 C2 后，从同一组通过验证的 actor 测试中选择分数最高者作为 C4。

采用“专家匹配候选”而不是直接把专家轨迹塞入 LG 候选池，是为了让完整场景和
反事实场景均由同一套运动生成与评价机制产生，降低专家轨迹与规划器轨迹直接比较
带来的系统偏差。C1 和 C3 仍严格使用真实专家未来轨迹。

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
future_interaction_grid: [5, H, W], float32, [0, 1]
```

其中 `H,W` 与当前帧 `costmap` 完全一致。当前数据通常为 `256×256`。

## 有效性规则

Driving 标签不会继承 LG 的 `risk_label_valid`。它独立检查：

- 专家参考路径是否存在；
- 专家未来 10 帧是否完整；
- 是否存在与专家轨迹足够接近的有效 LG 响应候选；
- 主要 actor 存在时，是否至少找到一个未来时刻的 actor 状态；
- 五通道 shape、有限性及 `[0,1]` 数值范围是否合法。

没有主要 actor 或次要 actor 本身不会令标签失效，对应通道保持全零。主要 actor
已被确认但未来 10 帧完全找不到时，标签置为无效。次要 actor 未来缺失不会使整张
标签失效，C4 保持全零并在元数据中记录原因。

## 重要配置

`expert_match` 决定完整场景候选是否足够接近专家轨迹。若大量标签因
`expert_match_threshold_failed` 失效，应先统计 ADE/FDE 分布，再调整阈值，不能直接
关闭检查，否则 C2/C4 可能不再真正代表专家驾驶策略下的关键对象。
