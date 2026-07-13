### 0. conda环境
```
conda activate simlingo

# 如果收集数据则运行
./start.sh
```

### 1. 生成cost map
```
python lg_waypoint_planner_project/lg_waypoint_planner_project/tools_bev/generate_costmap_from_masks.py
```
### 2. 生成语言标签
```
python lg_waypoint_planner_project/lg_waypoint_planner_project/tools_bev/run_language_grounded_waypoint_planner.py
```