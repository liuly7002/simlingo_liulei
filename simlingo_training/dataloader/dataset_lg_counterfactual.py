# -*- coding: utf-8 -*-

import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from simlingo_training.dataloader.dataset_lg import Data_LG


_COUNTERFACTUAL_INTENT_TEXT = {
    "route_follow": "Continue along the planned route.",
    "cautious_follow": "Proceed cautiously at a moderated speed.",
    "yield_stop": "Slow down and yield, stopping if the remaining scene requires it.",
    "left_nudge": "Make a slight adjustment to the left while continuing forward.",
    "right_nudge": "Make a slight adjustment to the right while continuing forward.",
    "creep": "Move forward slowly while observing the remaining scene.",
    "emergency_brake": "Brake firmly and be ready to stop.",
}


class Data_LG_Counterfactual(Data_LG):  # pylint: disable=invalid-name
    """
    在LG语言、轨迹、四通道和参与者空间监督之上，读取对象移除后的
    真实反事实重规划结果，并由该结果构造对应的四问语言监督。

    轨迹来自标签生成阶段真实执行参与者删除与重新规划后保存的
    ``reference.object_removed_reference_waypoints``。反事实语言则由同一次
    重规划保存的factor、intent和轨迹确定，不使用完整场景答案作为替代。
    """

    @staticmethod
    def _actor_identity(actor: Dict) -> Tuple[str, str]:
        """
        通过 id 和 class 来作为actor的身份信息
        """
        
        # 安全性检查
        if not isinstance(actor, dict):
            return "", ""

        # id
        actor_id = actor.get("id", None)
        
        # class
        actor_class = str(actor.get("semantic_class",actor.get("class", ""),))

        return ("" if actor_id is None else str(actor_id), actor_class,)

    @classmethod
    def _same_actor(cls, first: Dict, second: Dict) -> bool:
        """
        判断两个 actor 字典是否表示同一个交通参与者
        """

        # 提取第一个actor的身份信息
        first_id, first_class = cls._actor_identity(first)

        # 提取第二个actor的身份信息
        second_id, second_class = cls._actor_identity(second)


        # 两个 actor 都有id的时候就比较id
        if first_id and second_id:
            return first_id == second_id

        return bool(first_class) and first_class == second_class

    def _selected_counterfactual_test(self, causal_analysis: Dict,) -> Optional[Dict]:

        # 获取具体的参与因果测试的 object
        tests = causal_analysis.get("object_tests", [])
        # 安全性检查
        if not isinstance(tests, list):
            return None

        causal_object = causal_analysis.get("causal_object", {})
        matching: List[Dict] = []
        accepted: List[Dict] = []
        for test in tests:
            # 安全性检查
            if not isinstance(test, dict):
                continue
            # "counterfactual_valid" 表示该测试对象移除后,能否得到一条合法可用的反事实重规划轨迹？
            if not bool(test.get("counterfactual_valid", False)):
                continue

            # "causal_accepted" 表示这条反事实轨迹与完整场景轨迹的差异,是否足以证明该对象具有因果作用?
            if bool(test.get("causal_accepted", False)):
                accepted.append(test)

            if self._same_actor(test.get("actor", {}),causal_object,):
                matching.append(test)

        candidates = matching or accepted
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: float(item.get("causal_score", 0.0)),
        )

    @staticmethod
    def _counterfactual_attention_answer(factor_type: str, remaining_actor_count: int,) -> str:
        factor = str(factor_type).lower()
        if "red_light" in factor or "yellow_light" in factor or "green_light" in factor:
            return "The traffic signal ahead deserves priority attention."
        if "stop_sign" in factor:
            return "The stop sign ahead deserves priority attention."
        if remaining_actor_count > 0 and factor not in {
            "route_clear",
            "clear_route",
            "none",
            "unknown",
        }:
            return "A remaining road user near the planned route deserves attention."
        return "No dynamic road user currently requires priority attention."

    @staticmethod
    def _counterfactual_constraint_answer(factor_type: str,) -> str:
        factor = str(factor_type).lower()
        if "red_light" in factor:
            return "The red-light stop requirement still constrains the motion ahead."
        if "yellow_light" in factor:
            return "The yellow traffic signal requires a cautious approach."
        if "green_light" in factor:
            return "The signal permits progress, subject to the remaining traffic scene."
        if "stop_sign" in factor:
            return "The stop sign still requires a complete stop before proceeding."
        if factor in {
            "route_clear",
            "clear_route",
            "none",
            "unknown",
            "no_critical_factor",
        }:
            return "The planned route is clear enough to continue."
        return "The remaining traffic context constrains how the vehicle should proceed."

    @staticmethod
    def _counterfactual_motion_answer(waypoints: np.ndarray,speeds: np.ndarray,) -> str:
        points = np.asarray(waypoints, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
            return "The next movement cannot be determined reliably."

        endpoint = points[-1]
        max_displacement = float(
            np.linalg.norm(points, axis=-1).max(initial=0.0)
        )
        mean_lateral = float(np.mean(points[:, 1]))
        end_lateral = float(endpoint[1])

        speed_values = np.asarray(speeds, dtype=np.float32).reshape(-1)
        speed_values = speed_values[np.isfinite(speed_values)]
        if speed_values.size > 0:
            mean_speed = float(speed_values.mean())
            speed_delta = float(
                speed_values[-1] - speed_values[0]
            )
        else:
            previous = np.concatenate(
                [
                    np.zeros((1, 2), dtype=np.float32),
                    points[:-1],
                ],
                axis=0,
            )
            segment_speed = np.linalg.norm(
                points - previous,
                axis=-1,
            ) * 4.0
            mean_speed = float(segment_speed.mean())
            speed_delta = float(
                segment_speed[-1] - segment_speed[0]
            )

        if max_displacement < 0.20 or mean_speed < 0.15:
            return "The ego vehicle will remain stopped."

        if end_lateral > 0.65 or mean_lateral > 0.35:
            direction = "make a slight adjustment to the right"
        elif end_lateral < -0.65 or mean_lateral < -0.35:
            direction = "make a slight adjustment to the left"
        else:
            direction = "continue mostly along the planned route"

        if speed_delta < -0.8:
            speed_phrase = "while gradually reducing speed"
        elif speed_delta > 0.8:
            speed_phrase = "while gradually increasing speed"
        elif mean_speed < 1.5:
            speed_phrase = "at a low and steady speed"
        else:
            speed_phrase = "with little speed change"
        return f"The ego vehicle will {direction} {speed_phrase}."

    def _build_counterfactual_answer(
        self,
        causal_analysis: Dict,
        counterfactual_test: Dict,
        counterfactual_waypoints: np.ndarray,
        counterfactual_speeds: np.ndarray,) -> str:
        factor_type = str(
            counterfactual_test.get(
                "counterfactual_factor_type",
                "unknown",
            )
        )
        intent_name = str(
            counterfactual_test.get(
                "counterfactual_intent_name",
                "route_follow",
            )
        )
        remaining_actor_count = int(
            counterfactual_test.get(
                "counterfactual_remaining_actor_count",
                0,
            )
        )

        answers = (
            self._counterfactual_attention_answer(
                factor_type,
                remaining_actor_count,
            ),
            self._counterfactual_constraint_answer(factor_type),
            _COUNTERFACTUAL_INTENT_TEXT.get(
                intent_name,
                "Proceed according to the remaining traffic constraints.",
            ),
            self._counterfactual_motion_answer(
                counterfactual_waypoints,
                counterfactual_speeds,
            ),
        )
        return " ".join(
            f"A{index}: {answer}"
            for index, answer in enumerate(answers, start=1)
        ) + " Waypoints:"

    @staticmethod
    def _replace_conversation_answer(conversation: list,answer: str,) -> list:
        output = copy.deepcopy(conversation)
        assistant_found = False
        for message in output:
            if str(message.get("role", "")) != "assistant":
                continue
            content = message.get("content", [])
            if not isinstance(content, list) or len(content) == 0:
                continue
            content[0]["text"] = str(answer)
            assistant_found = True
            break
        if not assistant_found:
            output.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": str(answer),
                        }
                    ],
                }
            )
        return output

    def _extract_counterfactual_supervision(self,payload: Dict,) -> Tuple[np.ndarray, bool, float, Optional[str]]:
        
        # 未来 waypoints 的数量, 10个
        expected_count = int(self.pred_len) - 1  # 10
        
        # 创建一个空数组 [10,2], 将其作为无效情况下的默认轨迹
        empty = np.zeros((expected_count, 2), dtype=np.float32)

        # 如果不使用反事实监督
        if not bool(getattr(self,"lg_use_counterfactual_supervision",False,)):
            return empty, False, 0.0, None

        # 加载标签文件的 "causal_analysis" 字段 该字段保存主要参与者的因果分析结果
        causal_analysis = payload.get("causal_analysis", {})
        
        # 加载标签文件的 "reference" 字段 该字段保存参考场景和反事实重规划结果
        reference = payload.get("reference", {})
        
        if (
            not isinstance(causal_analysis, dict)  # 安全性检查
            or not isinstance(reference, dict)     # 安全性检查
            or not bool(causal_analysis.get("has_causal_object", False))   # false表示没有因果对象,true表示当前帧存在一个经过对象移除干预验证的主要因果参与者
        ):
            return empty, False, 0.0, None

        # reference字段的"source"对应的值只有明确标记为"object_removed_counterfactual"时候才能证明轨迹是删除主要因果参与者后重新规划得到的轨迹
        if str(reference.get("source", "")) != ("object_removed_counterfactual"):
            return empty, False, 0.0, None

        counterfactual_test = self._selected_counterfactual_test(causal_analysis)
        if counterfactual_test is None:
            return empty, False, 0.0, None

        # 读取真正的反事实轨迹,它表示删除主要因果参与者以后，规划器针对剩余场景重新规划得到的未来轨迹
        counterfactual = np.asarray(reference.get("object_removed_reference_waypoints",[],),dtype=np.float32,)
        # 安全性检查
        if counterfactual.shape != (expected_count, 2):
            raise ValueError(
                "A valid LG causal label must contain "
                "reference.object_removed_reference_waypoints with "
                f"shape ({expected_count}, 2), but received "
                f"{tuple(counterfactual.shape)}."
            )
        # 安全性检查
        if not np.isfinite(counterfactual).all():
            raise ValueError(
                "Counterfactual waypoints contain non-finite values."
            )

        # 不要超过100m
        max_abs = float(getattr(self, "lg_max_abs_waypoint_m", 100.0))
        if np.max(np.abs(counterfactual), initial=0.0) > max_abs:
            raise ValueError(
                "Counterfactual waypoints exceed the configured range."
            )

        # 完整场景中的lg轨迹
        full_scene = self._extract_lg_waypoints(payload)
        # 安全性检查
        if full_scene.shape != counterfactual.shape:
            raise ValueError(
                "Full-scene and counterfactual waypoint shapes do not match: "
                f"{tuple(full_scene.shape)} vs "
                f"{tuple(counterfactual.shape)}."
            )

        # 计算对象移除造成的平均轨迹变化
        mean_effect = float(np.linalg.norm(full_scene - counterfactual,axis=-1,).mean())
        min_effect = float(getattr(self,"lg_counterfactual_min_effect_m",0.03,))
        if mean_effect < min_effect:
            return empty, False, 0.0, None

        # 读取主要参与者的因果分数
        causal_score_value = causal_analysis.get("final_causal_score",causal_analysis.get("causal_score", 0.0),)

        try:
            causal_score = float(causal_score_value)
        except (TypeError, ValueError):
            causal_score = 0.0
        if not np.isfinite(causal_score):
            causal_score = 0.0

        # 读取反事实未来速度
        counterfactual_speeds = np.asarray(reference.get("object_removed_reference_speeds",[],),dtype=np.float32,)
        
        # 生成反事实四问语言答案
        counterfactual_answer = self._build_counterfactual_answer(
            causal_analysis=causal_analysis,
            counterfactual_test=counterfactual_test,
            counterfactual_waypoints=counterfactual,
            counterfactual_speeds=counterfactual_speeds,
        )

        return (
            counterfactual.astype(np.float32),  # 反事实轨迹 [10,2]
            True,                               # 反事实监督是否有效
            max(causal_score, 0.0),             # 主要参与者的因果强度
            counterfactual_answer,              # 反事实四问答案
        )

    def __getitem__(self, index):

        # 调用父类 Data_LG.__getitem__() 获得普通lg样本
        sample = super().__getitem__(index)

        # 当前lg有效样本的文件路径(这里的有效样本是指存在主要actor的样本)-path/to/language_grounded_waypoints/0026.json.gz
        lg_path = Path(self._decode_path(self.lg_label_paths[index]))

        # 加载并解析当前帧样本的lg标签的内容  实际上就是0026.json.gz本身
        payload = self._load_gzip_json(lg_path)

        counterfactual_waypoints,counterfactual_waypoints_valid,counterfactual_causal_score,counterfactual_answer = self._extract_counterfactual_supervision(payload)

        counterfactual_conversation = None
        if counterfactual_answer is not None:
            counterfactual_conversation = (
                self._replace_conversation_answer(
                    sample.conversation,
                    counterfactual_answer,
                )
            )

        # 补充 DatasetOutput 样本对象的内容
        return sample._replace(
            counterfactual_waypoints=(counterfactual_waypoints),             # 移除主要因果参与者以后,重新规划得到的未来轨迹,[10,2]
            counterfactual_waypoints_valid=(counterfactual_waypoints_valid), # 反事实轨迹是否能够作为有效监督,True：反事实重规划结果有效,可以参与训练,False：没有有效反事实监督
            counterfactual_causal_score=(counterfactual_causal_score),       # 移除主要参与者后,自车规划行为变化所对应的因果强度分数
            counterfactual_conversation=(counterfactual_conversation),       # 表示与反事实场景对应的完整对话
        )
