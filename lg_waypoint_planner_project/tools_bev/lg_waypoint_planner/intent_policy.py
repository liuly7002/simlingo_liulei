# -*- coding: utf-8 -*-

from typing import Dict, List

# 7种驾驶意图(行为假设空间)
INTENT_ID = {
    "route_follow": 0,      # 路径跟随
    "cautious_follow": 1,   # 谨慎跟随(可以理解为减速跟随)
    "yield_stop": 2,        # 让行停车
    "left_nudge": 3,        # 左侧微调
    "right_nudge": 4,       # 右侧微调
    "creep": 5,             # 缓慢前进
    "emergency_brake": 6,   # 紧急制动
}

INTENT_NAMES = list(INTENT_ID.keys())

INTENT_ZH = {
    "route_follow": "follow the route",
    "cautious_follow": "slow down and follow cautiously",
    "yield_stop": "yield and stop",
    "left_nudge": "nudge slightly to the left",
    "right_nudge": "nudge slightly to the right",
    "creep": "creep forward slowly",
    "emergency_brake": "brake hard",
}

INTENT_EN = {
    "route_follow": "follow the route",
    "cautious_follow": "slow down and follow cautiously",
    "yield_stop": "yield and stop",
    "left_nudge": "nudge slightly to the left",
    "right_nudge": "nudge slightly to the right",
    "creep": "creep forward slowly",
    "emergency_brake": "brake hard",
}


def infer_intents(factor: Dict, cfg) -> List[Dict]:
    ftype = factor.get("type", "unknown")
    actor = factor.get("critical_actor", {})  # 关键actor
    intents: List[Dict] = []

    def add(name: str, active: bool, reason: str, priority: int):
        if name not in INTENT_ID:
            return
        if bool(cfg.behaviors.save_all_intents) or active:
            intents.append({
                "intent_id": INTENT_ID[name],
                "intent_name": name,
                "intent_zh": INTENT_ZH[name],
                "intent_en": INTENT_EN[name],
                "active": bool(active),
                "activation_reason": reason,
                "priority": int(priority),
            })

     # 无论什么场景，route_follow都是默认的行为
    add("route_follow", True, "nominal reference behavior", 0) 

    # 红灯：交通灯是动作约束的主因，停止线只是约束位置。
    # 当前帧已经是红灯时，最终动作语义应明确为停车等待，不允许
    # cautious_follow 以“改动更小”为由抢走 yield_stop。未来时刻才
    # 会变红时，仍保留提前减速候选。
    if ftype == "red_light_stop_line":
        red_rule = factor.get("red_light_rule", {}) or {}
        current_red_active = bool(red_rule.get("current_red_active", False))
        add(
            "cautious_follow",
            not current_red_active,
            "future red light may require early deceleration" if not current_red_active
            else "current red light requires a stop rather than continued following",
            1,
        )
        add("yield_stop", True, "red traffic light requires stopping and waiting behind the line", 2)
        add(
            "emergency_brake",
            False if current_red_active else True,
            "diagnostic hard-braking candidate",
            6,
        )
        add("left_nudge", False, "lateral motion does not bypass a red-light rule", 4)
        add("right_nudge", False, "lateral motion does not bypass a red-light rule", 5)
        add("creep", False, "creep is not a default response to an active red light", 3)
    # 停止标志：必须先完成停车，不能通过横向绕行规避规则。
    elif ftype == "stop_sign_control":
        add("cautious_follow", True, "stop sign may require early deceleration", 1)
        add("yield_stop", True, "stop sign requires a complete stop before proceeding", 2)
        add("emergency_brake", True, "diagnostic hard-braking candidate", 6)
        add("left_nudge", False, "lateral motion does not bypass a stop-sign rule", 4)
        add("right_nudge", False, "lateral motion does not bypass a stop-sign rule", 5)
        add("creep", False, "creep is not a substitute for a complete stop", 3)
    # 1. 前方静态障碍物，激活全部7种驾驶意图
    elif ftype == "front_static_obstacle":
        add("cautious_follow", True, "front static obstacle may require lower speed", 1)
        add("yield_stop", True, "front static obstacle may require yielding", 2)
        add("left_nudge", True, "left bypass alternative around the front static obstacle", 3)
        add("right_nudge", True, "right bypass alternative around the front static obstacle", 4)
        add("creep", True, "static obstacle may require low-speed probing", 5)
        add("emergency_brake", True, "diagnostic hard-braking candidate", 6)
    # 2. 前方actor或行人，激活5种驾驶意图(主要考虑:正前方普通动态对象默认优先考虑纵向响应而不是主动绕行)
    elif ftype in ["front_actor", "pedestrian_crossing_or_near_path"]:
        add("cautious_follow", True, "front factor requires lower speed", 1)
        add("yield_stop", True, "front factor may require yielding", 2)
        add("creep", actor.get("class") == "pedestrian", "pedestrian may require low-speed probing", 3)
        add("emergency_brake", True, "diagnostic hard-braking candidate", 6)
        add("left_nudge", False, "lateral candidate kept for comparison", 4)
        add("right_nudge", False, "lateral candidate kept for comparison", 5)
    # 3.右前方actor或静态障碍物，激活5种驾驶意图(主要考虑:右前方对象默认优先考虑左侧绕行而不是右侧绕行)
    elif ftype in ["front_right_actor", "front_right_static_obstacle"]:
        add("left_nudge", True, "front-right factor reduces right-side clearance", 1)
        add("cautious_follow", True, "nearby factor requires lower speed", 2)
        add("yield_stop", actor.get("class") in ["traffic_cone", "traffic_warning", "barrier"], "static obstacle may require yielding", 4)
        add("right_nudge", False, "right nudge is likely unsafe when risk is on the right", 5)
        add("emergency_brake", True, "diagnostic hard-braking candidate", 6)
    # 4.左前方actor或静态障碍物，激活5种驾驶意图(主要考虑:左前方对象默认优先考虑右侧绕行而不是左侧绕行)
    elif ftype in ["front_left_actor", "front_left_static_obstacle"]:
        add("right_nudge", True, "front-left factor reduces left-side clearance", 1)
        add("cautious_follow", True, "nearby factor requires lower speed", 2)
        add("yield_stop", actor.get("class") in ["traffic_cone", "traffic_warning", "barrier"], "static obstacle may require yielding", 4)
        add("left_nudge", False, "left nudge is likely unsafe when risk is on the left", 5)
        add("emergency_brake", True, "diagnostic hard-braking candidate", 6)
    # 5. 静态障碍物在自车附近，激活7种驾驶意图
    elif ftype == "static_obstacle_nearby":
        add("cautious_follow", True, "static obstacle requires lower speed", 1)
        add("yield_stop", True, "static obstacle may require yielding", 2)
        add("left_nudge", True, "left lateral alternative", 3)
        add("right_nudge", True, "right lateral alternative", 4)
        add("creep", True, "conservative probing near static obstacle", 5)
        add("emergency_brake", True, "diagnostic hard-braking candidate", 6)
    # 6. costmap阻塞
    elif ftype == "costmap_blocked_corridor":
        add("cautious_follow", True, "high cost on reference corridor", 1)
        add("yield_stop", True, "blocked corridor may require yielding", 2)
        add("left_nudge", True, "left lateral alternative", 3)
        add("right_nudge", True, "right lateral alternative", 4)
        add("creep", True, "conservative probing through uncertain space", 5)
        add("emergency_brake", True, "diagnostic hard-braking candidate", 6)
    else:
        add("cautious_follow", False, "backup slower route-follow candidate", 1)
        add("left_nudge", False, "backup lateral candidate", 2)
        add("right_nudge", False, "backup lateral candidate", 3)
        add("yield_stop", False, "backup stop candidate", 4)
        add("creep", False, "backup low-speed candidate", 5)
        add("emergency_brake", False, "diagnostic hard-braking candidate", 6)

    # De-duplicate while preserving first priority.
    seen = set(); out = []
    for item in sorted(intents, key=lambda x: x["priority"]):
        name = item["intent_name"]
        if name in seen:
            continue
        seen.add(name); out.append(item)
    return out
