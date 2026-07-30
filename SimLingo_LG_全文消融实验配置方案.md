# SimLingo-LG 全文消融实验配置方案

> 适用代码版本：`c6e0435713956ba7cfca2a457016ae2c4ac373f8`  
> 提交说明：`统一标签生成侧和训练侧`  
> 本方案用于论文中三个核心创新点的统一消融，后续正式实验应固定该代码版本、数据标签和训练设置。

---

## 1. 三个创新点及对应开关

本文将三个创新点记为：

- **A：导航意图引导的六视角注意力**
  - 两个 `TARGET_POINT` 形成导航查询；
  - 导航查询预测六个相机的注意力权重；
  - LG与普通Driving的主要关键actor投影标签提供显式监督。

- **L：关键actor驱动的语言—动作一致性监督**
  - LG语言描述与LG重规划waypoints来自同一关键actor和驾驶响应过程；
  - 通过同时启用LG四问语言和LG重规划waypoints体现。

- **S：四通道结构化未来世界辅助预测**
  - C0：参考路线；
  - C1：自车未来运动；
  - C2：主要关键actor未来运动；
  - C4：次要关键actor未来运动；
  - 普通Driving和LG均提供该监督。

### 1.1 A：六视角注意力开关

完整开启A时，三个开关必须同时为 `true`：

```yaml
model:
  vision_model:
    use_target_point_camera_attention: true
  use_lg_camera_attention_supervision: true

data_module:
  driving_dataset:
    driving_use_camera_attention_supervision: true
```

完整关闭A时，三个开关必须同时为 `false`：

```yaml
model:
  vision_model:
    use_target_point_camera_attention: false
  use_lg_camera_attention_supervision: false

data_module:
  driving_dataset:
    driving_use_camera_attention_supervision: false
```

> 当前变量名 `use_lg_camera_attention_supervision` 为历史名称。当前代码中该损失已经同时接收LG和普通Driving的有效六视角标签。

### 1.2 L：语言—动作一致性开关

完整开启L：

```yaml
data_module:
  dreamer_dataset:
    lg_use_language: true
    lg_use_waypoints: true
    lg_language_mode: four_questions
```

完整关闭L：

```yaml
data_module:
  dreamer_dataset:
    lg_use_language: false
    lg_use_waypoints: false
    lg_language_mode: none
```

关闭L后仍保留同一批LG样本和同一数据采样比例，但该数据槽使用waypoint-only提示与原始专家waypoints，从而尽量只移除语言—重规划动作的一致性监督，避免改变训练样本数量。

### 1.3 S：结构化未来世界开关

完整开启S：

```yaml
model:
  use_future_interaction_prediction: true

data_module:
  driving_dataset:
    driving_use_future_interaction_grid: true
  dreamer_dataset:
    lg_use_future_interaction_grid: true
```

完整关闭S：

```yaml
model:
  use_future_interaction_prediction: false

data_module:
  driving_dataset:
    driving_use_future_interaction_grid: false
  dreamer_dataset:
    lg_use_future_interaction_grid: false
```

---

## 2. 所有正式实验必须保持不变的设置

主消融实验只改变A、L、S及其内部子模块。以下设置必须固定：

```yaml
data_module:
  base_dataset:
    route_as: target_point
    use_lmdrive_commands: false
    pred_len: 11

model:
  lr: 3e-5
  predict_route_as_wps: true
  speed_wps_mode: 2d
  lg_camera_attention_loss_weight: 0.05
  future_interaction_output_size: 128
  future_interaction_loss_weight: 0.05
  future_interaction_dice_loss_weight: 1.0
  future_interaction_positive_weights: [20.0, 20.0, 80.0, 100.0]

max_epochs: 15
val_every_n_epochs: 2
resume: false
enable_checkpointing: true
checkpoint_save_last: true
seed: 9876
```

同时固定：

- 六视角输入及其相机顺序；
- 普通Driving与LG的训练采样比例；
- 训练集、验证集及Town划分；
- batch size、优化器、学习率和训练轮数；
- 数据增强设置；
- 初始预训练模型；
- 标签版本；
- 评估路线和评估随机种子。

### 重要原则

1. **所有主消融均使用 `route_as=target_point`。**  
   关闭A时只关闭注意力模块，不能改回HLC，否则“导航表示变化”和“注意力模块变化”会混在一起。

2. **所有实验从同一预训练权重独立开始。**  
   不允许从完整模型或其他消融模型继续训练。

3. **每个实验必须保存checkpoint。**  
   当前实验YAML中 `enable_checkpointing` 默认是 `false`，正式实验必须覆盖为 `true`。

---

## 3. 主消融：A、L、S三因素全组合

主表采用完整的 `2^3` 因子设计，共8组。这样不仅可以验证三个创新点各自的贡献，还可以观察它们之间是否存在互补关系。

| 编号 | A 六视角注意力 | L 语言—动作一致性 | S 结构化未来世界 | 实验名称 |
|---|---:|---:|---:|---|
| M0 | × | × | × | `abl_m0_base` |
| M1 | √ | × | × | `abl_m1_attention` |
| M2 | × | √ | × | `abl_m2_language_action` |
| M3 | × | × | √ | `abl_m3_structured_world` |
| M4 | √ | √ | × | `abl_m4_attention_language` |
| M5 | √ | × | √ | `abl_m5_attention_structured` |
| M6 | × | √ | √ | `abl_m6_language_structured` |
| M7 | √ | √ | √ | `abl_m7_full` |

### 3.1 通用运行命令

```bash
./train_simlingo_seed1.sh \
  experiment=simlingo_lg_seed1 \
  enable_checkpointing=true \
  checkpoint_save_last=true \
  resume=false \
  seed=9876 \
  data_module.base_dataset.route_as=target_point \
  data_module.base_dataset.use_lmdrive_commands=false \
  <该实验的覆盖参数>
```

### 3.2 M0：基础模型

```bash
name=abl_m0_base \
model.vision_model.use_target_point_camera_attention=false \
model.use_lg_camera_attention_supervision=false \
data_module.driving_dataset.driving_use_camera_attention_supervision=false \
data_module.dreamer_dataset.lg_use_language=false \
data_module.dreamer_dataset.lg_use_waypoints=false \
data_module.dreamer_dataset.lg_language_mode=none \
model.use_future_interaction_prediction=false \
data_module.driving_dataset.driving_use_future_interaction_grid=false \
data_module.dreamer_dataset.lg_use_future_interaction_grid=false
```

### 3.3 M1：仅六视角注意力

```bash
name=abl_m1_attention \
model.vision_model.use_target_point_camera_attention=true \
model.use_lg_camera_attention_supervision=true \
data_module.driving_dataset.driving_use_camera_attention_supervision=true \
data_module.dreamer_dataset.lg_use_language=false \
data_module.dreamer_dataset.lg_use_waypoints=false \
data_module.dreamer_dataset.lg_language_mode=none \
model.use_future_interaction_prediction=false \
data_module.driving_dataset.driving_use_future_interaction_grid=false \
data_module.dreamer_dataset.lg_use_future_interaction_grid=false
```

### 3.4 M2：仅语言—动作一致性

```bash
name=abl_m2_language_action \
model.vision_model.use_target_point_camera_attention=false \
model.use_lg_camera_attention_supervision=false \
data_module.driving_dataset.driving_use_camera_attention_supervision=false \
data_module.dreamer_dataset.lg_use_language=true \
data_module.dreamer_dataset.lg_use_waypoints=true \
data_module.dreamer_dataset.lg_language_mode=four_questions \
model.use_future_interaction_prediction=false \
data_module.driving_dataset.driving_use_future_interaction_grid=false \
data_module.dreamer_dataset.lg_use_future_interaction_grid=false
```

### 3.5 M3：仅结构化未来世界

```bash
name=abl_m3_structured_world \
model.vision_model.use_target_point_camera_attention=false \
model.use_lg_camera_attention_supervision=false \
data_module.driving_dataset.driving_use_camera_attention_supervision=false \
data_module.dreamer_dataset.lg_use_language=false \
data_module.dreamer_dataset.lg_use_waypoints=false \
data_module.dreamer_dataset.lg_language_mode=none \
model.use_future_interaction_prediction=true \
data_module.driving_dataset.driving_use_future_interaction_grid=true \
data_module.dreamer_dataset.lg_use_future_interaction_grid=true
```

### 3.6 M4：六视角注意力 + 语言—动作一致性

```bash
name=abl_m4_attention_language \
model.vision_model.use_target_point_camera_attention=true \
model.use_lg_camera_attention_supervision=true \
data_module.driving_dataset.driving_use_camera_attention_supervision=true \
data_module.dreamer_dataset.lg_use_language=true \
data_module.dreamer_dataset.lg_use_waypoints=true \
data_module.dreamer_dataset.lg_language_mode=four_questions \
model.use_future_interaction_prediction=false \
data_module.driving_dataset.driving_use_future_interaction_grid=false \
data_module.dreamer_dataset.lg_use_future_interaction_grid=false
```

### 3.7 M5：六视角注意力 + 结构化未来世界

```bash
name=abl_m5_attention_structured \
model.vision_model.use_target_point_camera_attention=true \
model.use_lg_camera_attention_supervision=true \
data_module.driving_dataset.driving_use_camera_attention_supervision=true \
data_module.dreamer_dataset.lg_use_language=false \
data_module.dreamer_dataset.lg_use_waypoints=false \
data_module.dreamer_dataset.lg_language_mode=none \
model.use_future_interaction_prediction=true \
data_module.driving_dataset.driving_use_future_interaction_grid=true \
data_module.dreamer_dataset.lg_use_future_interaction_grid=true
```

### 3.8 M6：语言—动作一致性 + 结构化未来世界

```bash
name=abl_m6_language_structured \
model.vision_model.use_target_point_camera_attention=false \
model.use_lg_camera_attention_supervision=false \
data_module.driving_dataset.driving_use_camera_attention_supervision=false \
data_module.dreamer_dataset.lg_use_language=true \
data_module.dreamer_dataset.lg_use_waypoints=true \
data_module.dreamer_dataset.lg_language_mode=four_questions \
model.use_future_interaction_prediction=true \
data_module.driving_dataset.driving_use_future_interaction_grid=true \
data_module.dreamer_dataset.lg_use_future_interaction_grid=true
```

### 3.9 M7：完整模型

```bash
name=abl_m7_full \
model.vision_model.use_target_point_camera_attention=true \
model.use_lg_camera_attention_supervision=true \
data_module.driving_dataset.driving_use_camera_attention_supervision=true \
data_module.dreamer_dataset.lg_use_language=true \
data_module.dreamer_dataset.lg_use_waypoints=true \
data_module.dreamer_dataset.lg_language_mode=four_questions \
model.use_future_interaction_prediction=true \
data_module.driving_dataset.driving_use_future_interaction_grid=true \
data_module.dreamer_dataset.lg_use_future_interaction_grid=true
```

---

## 4. 创新点A内部消融：注意力结构与显式监督

固定 `L=true`、`S=true`，比较以下4种设置。

| 编号 | 注意力结构 | LG显式监督 | Driving显式监督 | 对应实验 |
|---|---:|---:|---:|---|
| A0 | × | × | × | 复用M6 |
| A1 | √ | × | × | 新增 |
| A2 | √ | √ | × | 新增 |
| A3 | √ | √ | √ | 复用M7 |

### A1：仅注意力结构，无显式监督

```bash
name=abl_a1_attention_implicit \
model.vision_model.use_target_point_camera_attention=true \
model.use_lg_camera_attention_supervision=false \
data_module.driving_dataset.driving_use_camera_attention_supervision=false \
data_module.dreamer_dataset.lg_use_language=true \
data_module.dreamer_dataset.lg_use_waypoints=true \
data_module.dreamer_dataset.lg_language_mode=four_questions \
model.use_future_interaction_prediction=true \
data_module.driving_dataset.driving_use_future_interaction_grid=true \
data_module.dreamer_dataset.lg_use_future_interaction_grid=true
```

### A2：注意力结构 + 仅LG显式监督

```bash
name=abl_a2_attention_lg_supervision \
model.vision_model.use_target_point_camera_attention=true \
model.use_lg_camera_attention_supervision=true \
data_module.driving_dataset.driving_use_camera_attention_supervision=false \
data_module.dreamer_dataset.lg_use_language=true \
data_module.dreamer_dataset.lg_use_waypoints=true \
data_module.dreamer_dataset.lg_language_mode=four_questions \
model.use_future_interaction_prediction=true \
data_module.driving_dataset.driving_use_future_interaction_grid=true \
data_module.dreamer_dataset.lg_use_future_interaction_grid=true
```

A2与M7的差异只在普通Driving显式注意力标签，因此：

- A2 → M7：验证普通Driving关键actor投影监督的增益；
- A1 → A2：验证LG显式监督相对于纯隐式学习的增益；
- A0 → A1：验证目标点引导注意力网络结构本身的增益。

> 当前配置不能单独关闭LG注意力标签而只保留Driving标签，因此不安排“Driving-only显式监督”组，避免通过非正式方式制造不受代码支持的实验。

---

## 5. 创新点L内部消融：语言与重规划动作

固定 `A=true`、`S=true`。

| 编号 | LG四问语言 | LG重规划waypoints | 对应实验 |
|---|---:|---:|---|
| L0 | × | × | 复用M5 |
| L1 | √ | × | 新增 |
| L2 | × | √ | 新增 |
| L3 | √ | √ | 复用M7 |

### L1：仅语言，动作仍使用专家waypoints

```bash
name=abl_l1_language_only \
model.vision_model.use_target_point_camera_attention=true \
model.use_lg_camera_attention_supervision=true \
data_module.driving_dataset.driving_use_camera_attention_supervision=true \
data_module.dreamer_dataset.lg_use_language=true \
data_module.dreamer_dataset.lg_use_waypoints=false \
data_module.dreamer_dataset.lg_language_mode=four_questions \
model.use_future_interaction_prediction=true \
data_module.driving_dataset.driving_use_future_interaction_grid=true \
data_module.dreamer_dataset.lg_use_future_interaction_grid=true
```

### L2：仅重规划waypoints，无四问语言

```bash
name=abl_l2_replanned_waypoints_only \
model.vision_model.use_target_point_camera_attention=true \
model.use_lg_camera_attention_supervision=true \
data_module.driving_dataset.driving_use_camera_attention_supervision=true \
data_module.dreamer_dataset.lg_use_language=false \
data_module.dreamer_dataset.lg_use_waypoints=true \
data_module.dreamer_dataset.lg_language_mode=none \
model.use_future_interaction_prediction=true \
data_module.driving_dataset.driving_use_future_interaction_grid=true \
data_module.dreamer_dataset.lg_use_future_interaction_grid=true
```

该组实验用于区分：

- 仅增加语言监督是否有效；
- 仅增加重规划动作监督是否有效；
- 二者同时由同一关键actor与响应过程生成时，是否产生额外的一致性增益。

---

## 6. 创新点S内部消融：四通道逐步组成

固定 `A=true`、`L=true`。

当前解码器固定输出4通道。通道消融通过把被移除通道的任务权重设为0实现，不修改网络结构，从而保持参数规模和计算结构一致。

| 编号 | C0路线 | C1自车 | C2主要actor | C4次要actor | 对应实验 |
|---|---:|---:|---:|---:|---|
| S0 | × | × | × | × | 复用M4 |
| S1 | √ | √ | × | × | 新增 |
| S2 | √ | √ | √ | × | 新增 |
| S3 | √ | √ | √ | √ | 复用M7 |

### S1：仅C0 + C1

```bash
name=abl_s1_route_ego \
model.vision_model.use_target_point_camera_attention=true \
model.use_lg_camera_attention_supervision=true \
data_module.driving_dataset.driving_use_camera_attention_supervision=true \
data_module.dreamer_dataset.lg_use_language=true \
data_module.dreamer_dataset.lg_use_waypoints=true \
data_module.dreamer_dataset.lg_language_mode=four_questions \
model.use_future_interaction_prediction=true \
data_module.driving_dataset.driving_use_future_interaction_grid=true \
data_module.dreamer_dataset.lg_use_future_interaction_grid=true \
model.future_interaction_channel_weights=[1.0,1.0,0.0,0.0]
```

### S2：C0 + C1 + C2

```bash
name=abl_s2_route_ego_primary \
model.vision_model.use_target_point_camera_attention=true \
model.use_lg_camera_attention_supervision=true \
data_module.driving_dataset.driving_use_camera_attention_supervision=true \
data_module.dreamer_dataset.lg_use_language=true \
data_module.dreamer_dataset.lg_use_waypoints=true \
data_module.dreamer_dataset.lg_language_mode=four_questions \
model.use_future_interaction_prediction=true \
data_module.driving_dataset.driving_use_future_interaction_grid=true \
data_module.dreamer_dataset.lg_use_future_interaction_grid=true \
model.future_interaction_channel_weights=[1.0,1.0,2.0,0.0]
```

S2 → M7主要验证次要关键actor通道C4的作用。

---

## 7. 随机种子与实验总数

### 7.1 必做实验

以 `seed=9876` 完成：

- 主消融M0—M7：8组；
- A内部新增A1、A2：2组；
- L内部新增L1、L2：2组；
- S内部新增S1、S2：2组。

共计：

```text
14组独立训练
```

### 7.2 稳定性复验

至少对以下两组增加两个随机种子：

- M0基础模型；
- M7完整模型。

新增：

```yaml
seed: 1
seed: 42
```

因此稳定性复验增加4组，总训练数为：

```text
14 + 4 = 18组
```

论文中：

- 主消融表可报告 `seed=9876` 的统一结果；
- M0与M7报告3个随机种子的均值与标准差；
- 计算资源允许时，可将全部14组扩展到3个随机种子，但不能只选择表现最好的种子。

---

## 8. 论文中建议报告的指标

### 8.1 闭环驾驶指标

主结论应优先基于闭环评估，包括：

- Driving Score；
- Route Completion；
- 成功率；
- 碰撞相关违规；
- 红灯、道路偏离等安全违规；
- 平均速度或停滞情况。

所有消融必须使用完全一致的评估路线、交通设置和随机种子。

### 8.2 开环规划指标

至少报告：

- ADE；
- FDE；
- waypoint逐时域误差；
- route预测误差；
-停止场景与运动场景分组误差。

### 8.3 六视角注意力指标

对A0—A3报告：

- 有效注意力标签样本数和比例；
- 注意力软标签交叉熵；
- 六个相机平均预测权重；
- 归一化注意力熵；
- 预测最高权重相机与标签最高权重相机的一致率；
- LG与普通Driving分别统计的结果。

### 8.4 四通道结构化世界指标

对S0—S3报告：

- C0、C1、C2、C4各自的BCE；
- C0、C1、C2、C4各自的Dice；
- 各通道有效正样本数量；
- C2、C4稀疏通道的单独结果。

---

## 9. 论文表格组织建议

### 表A：三个创新点的主消融

| 方法 | A | L | S | Driving Score ↑ | Route Completion ↑ | ADE ↓ | FDE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 |  |  |  |  |  |  |  |
| M1 | √ |  |  |  |  |  |  |
| M2 |  | √ |  |  |  |  |  |
| M3 |  |  | √ |  |  |  |  |
| M4 | √ | √ |  |  |  |  |  |
| M5 | √ |  | √ |  |  |  |  |
| M6 |  | √ | √ |  |  |  |  |
| M7 | √ | √ | √ |  |  |  |  |

### 表B：六视角注意力内部消融

| 方法 | 目标点注意力结构 | LG显式监督 | Driving显式监督 | Driving Score ↑ | 注意力CE ↓ |
|---|---:|---:|---:|---:|---:|
| A0 |  |  |  |  |  |
| A1 | √ |  |  |  |  |
| A2 | √ | √ |  |  |  |
| A3 | √ | √ | √ |  |  |

### 表C：语言—动作一致性内部消融

| 方法 | 四问语言 | 重规划waypoints | Driving Score ↑ | ADE ↓ | 语言—动作一致性指标 ↑ |
|---|---:|---:|---:|---:|---:|
| L0 |  |  |  |  |  |
| L1 | √ |  |  |  |  |
| L2 |  | √ |  |  |  |
| L3 | √ | √ |  |  |  |

### 表D：结构化未来世界通道消融

| 方法 | C0 | C1 | C2 | C4 | Driving Score ↑ | ADE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| S0 |  |  |  |  |  |  |
| S1 | √ | √ |  |  |  |  |
| S2 | √ | √ | √ |  |  |  |
| S3 | √ | √ | √ | √ |  |  |

---

## 10. 正式实验前检查

### 10.1 标签只生成一次并冻结

使用当前代码重新生成普通Driving标签，使以下文件同时包含最新结果：

```text
driving_future_interaction_grids/<frame>.npz
driving_expert_conditioned_actor_selection/<frame>.json.gz
```

普通Driving的actor选择JSON必须包含：

```text
visual_grounding.camera_attention_target
visual_grounding.camera_attention_valid
```

为保证论文元数据一致，LG标签也建议使用当前统一投影代码重新生成一次。完成后冻结标签，所有消融使用完全相同的标签文件。

### 10.2 标签统计

正式训练前至少确认：

```text
LG camera_attention_valid > 0
Driving camera_attention_valid > 0
LG future_interaction_valid > 0
Driving future_interaction_valid > 0
```

并检查：

- 有效六维注意力标签之和约等于1；
- 相机顺序统一为：
  `front, front_left, front_right, rear, rear_left, rear_right`；
- 所有结构化世界标签shape为 `[4,H,W]`；
- 通道顺序统一为C0、C1、C2、C4；
- 标签中不存在NaN或Inf。

### 10.3 单batch联调

先用：

```bash
data_module.batch_size=1
data_module.num_workers=0
```

分别确认普通Driving batch和LG batch均能：

- 完成前向传播；
- 完成反向传播；
- 产生有限loss；
- 在有效样本中产生注意力辅助损失；
- 在有效样本中产生四通道结构化世界损失。

### 10.4 正式训练要求

- `resume=false`；
- 不加载旧的五通道checkpoint；
- 不在不同消融之间共享中间checkpoint；
- 每组使用唯一的 `name`；
- 保存完整配置、Git commit、checkpoint和评估结果；
- 训练失败后重新运行时必须保持同一配置和种子。

---

## 11. 最终实验清单

按以下顺序运行，便于先验证主结论，再完成内部分析：

1. M0、M7；
2. M1、M2、M3；
3. M4、M5、M6；
4. A1、A2；
5. L1、L2；
6. S1、S2；
7. M0与M7的 `seed=1`、`seed=42` 复验。

最终形成：

```text
8组主消融
+ 6组内部消融
+ 4组稳定性复验
= 18组正式训练
```
