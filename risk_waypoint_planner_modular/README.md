# Modular risk waypoint planner

This folder is a direct modular split of the uploaded monolithic `plan_waypoints_bicycle_from_costmap.py`.
The intended behavior is unchanged: the entry script keeps the same command-line interface, and the original function bodies were moved into responsibility-based modules.

## Structure

- `tools_bev/plan_waypoints_bicycle_from_costmap.py`: entry point.
- `tools_bev/risk_waypoint_planner/config.py`: argparse parameters.
- `logging_utils.py`: logging setup.
- `io_utils.py`: JSON/GZip/NumPy/list IO helpers.
- `geometry_utils.py`: route/polyline/coordinate geometry helpers.
- `measurement_utils.py`: measurement/meta parsing and expert future waypoints.
- `temporal_costmap.py`: future costmap warping and temporal fusion.
- `behavior_candidates.py`: scene diagnosis and behavior-conditioned candidates.
- `bicycle_rollout.py`: pure-pursuit + kinematic bicycle rollout.
- `scoring.py`: costmap/footprint scoring and best rollout selection.
- `qa_annotation.py`: Chinese/English QA and language annotation generation.
- `visualization.py`: BEV and RGB debug visualization.
- `processor.py`: frame/route processing orchestration.

## Usage

Run with the same command style as before, for example:

```bash
python risk_waypoint_planner_modular/tools_bev/plan_waypoints_bicycle_from_costmap.py \
  --input /home/liulei/ll/simlingo/database/simlingo_v2_2026_06_02/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_1553_route0_06_02_16_40_27 \
  --recursive \
  --save_debug \
  --save_rgb_debug \
  --verbose \
  --score_footprint \
  --behavior_candidate_policy all \
  --left_nudge_offset_m 1.0 \
  --right_nudge_offset_m 1.0 \
  --route_deviation_weight 8.0 \
  --max_route_deviation_m 4.0 \
  --max_hard_ratio 0.45 \
  --save_dreamer_candidates \
  --dreamer_include_invalid
```


```bash
# 先跑规则检查
python risk_waypoint_planner_modular/tools_bev/evaluate_dreamer_qa_with_gpt.py \
  --input /home/liulei/ll/simlingo/database/simlingo_v2_2026_06_02/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_1553_route0_06_02_16_40_27 \
  --recursive \
  --folder_name risk_waypoints_bicycle_dreamer \
  --output_jsonl qa_eval_rules.jsonl \
  --output_csv qa_eval_rules.csv \
  --max_files 200 \
  --rules_only

# 再抽样调用GPT
python risk_waypoint_planner_modular/tools_bev/evaluate_dreamer_qa_with_gpt.py \
  --input /home/liulei/ll/simlingo/database/simlingo_v2_2026_06_02/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_1553_route0_06_02_16_40_27 \
  --recursive \
  --folder_name risk_waypoints_bicycle_dreamer \
  --output_jsonl qa_eval_gpt.jsonl \
  --output_csv qa_eval_gpt.csv \
  --model gpt-4o-mini \
  --max_files 200
```
