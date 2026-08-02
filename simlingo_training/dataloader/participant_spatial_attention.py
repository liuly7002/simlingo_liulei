# -*- coding: utf-8 -*-

from typing import Dict, Tuple

import numpy as np


#############################################################################################################
#############################################################################################################

# 预定义的六视角相机顺序
CAMERA_ORDER: Tuple[str, ...] = ("front","front_left","front_right","rear","rear_left","rear_right",)

# 预定义的每个相机的patch的数量
NUM_PATCHES_PER_CAMERA = 2

# 预定义的每个相机的视觉token网格的高度
TOKEN_GRID_HEIGHT = 4

# 预定义的每个相机的视觉token网格的宽度
TOKEN_GRID_WIDTH = 8

# 预定义的每个相机的视觉token的总数量 2 * 4 * 8 = 64
TOKENS_PER_CAMERA = (NUM_PATCHES_PER_CAMERA * TOKEN_GRID_HEIGHT * TOKEN_GRID_WIDTH)

# 预定义的标签平滑参数
LABEL_SMOOTHING = 0.01

#############################################################################################################
#############################################################################################################












def _choose_two_patch_layout(image_width: int,image_height: int,) -> Tuple[int, int]:
    """
    根据当前相机图像的宽高比，判断 InternVL 的两个视觉 patch 应该采用“左右排列”还是“上下排列”
    """

    # 安全性检查
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    # 计算图像的宽高比 宽高比越大,图像越"宽"；宽高比越小,图像越"高"
    aspect_ratio = float(image_width) / float(image_height)

    # 定义两种候选布局：左右排列和上下排列
    candidates = (
        (2, 1),  # 两个patch左右排列
        (1, 2),  # 两个patch上下排列
    )

    # 从候选布局中选择与图像宽高比最接近的布局(这段代码会计算"原始宽高比"与"候选布局宽高比"之间的差距,选择差值最小的布局)
    # 一般返回(2, 1)
    return min(candidates,key=lambda layout: abs(aspect_ratio - float(layout[0]) / float(layout[1])),)


def _bbox_to_patch_token_weights(bbox_normalized_xyxy: np.ndarray,patch_layout: Tuple[int, int],) -> np.ndarray:
    """
    将 "主要actor在图像中的二维框" 转换为 "与 InternVL 实际视觉 token 一一对应的 64 维软监督权重"
    """

    # columns = 2 rows = 1 左右排列
    columns, rows = patch_layout

    # 安全性检查,确保每个相机的视觉patch数量为2
    if columns * rows != NUM_PATCHES_PER_CAMERA:
        raise ValueError(
            "Participant attention requires exactly two visual patches "
            f"per camera, but received layout={patch_layout}."
        )

    # 将归一化的bbox坐标转换为浮点数,并确保它们在[0,1]范围内
    x0, y0, x1, y1 = (float(value) for value in bbox_normalized_xyxy.tolist())

    # 将归一化的bbox坐标裁剪到[0,1]范围内,确保它们不会超出图像边界
    x0 = float(np.clip(x0, 0.0, 1.0))
    y0 = float(np.clip(y0, 0.0, 1.0))
    x1 = float(np.clip(x1, 0.0, 1.0))
    y1 = float(np.clip(y1, 0.0, 1.0))

    # 安全性检查,如果裁剪后的bbox的宽度或高度为负,则返回一个全零的视觉token权重数组
    if x1 <= x0 or y1 <= y0:
        return np.zeros((TOKENS_PER_CAMERA,), dtype=np.float32)

    # 最难理解的内容
    # 含义:将原始整张图像的归一化坐标，转换到由两个单位 patch 拼成的"大画布坐标系"
    canvas_bbox = np.asarray(
        (
            x0 * columns,  # 由于是左右排列,所以x坐标需要乘以列数
            y0 * rows,
            x1 * columns,  # 由于是左右排列,所以x坐标需要乘以列数
            y1 * rows,
        ),
        dtype=np.float64,
    )

    patch_maps = []
    cell_width = 1.0 / float(TOKEN_GRID_WIDTH)    # 每个cell单元的宽度
    cell_height = 1.0 / float(TOKEN_GRID_HEIGHT)  # 每个cell单元的高度
    cell_area = cell_width * cell_height

    # 遍历每个patch,计算每个patch中每个视觉token的权重
    for patch_index in range(NUM_PATCHES_PER_CAMERA):

        # 计算当前patch在"大画布"中的行列位置
        patch_column = patch_index % columns
        patch_row = patch_index // columns

        # 把全局归一化的bbox坐标转换为当前patch的局部归一化坐标,确保它们在[0,1]范围内
        # 作用是求参与者框与当前 patch 的交集，并把交集转换成当前 patch 内部的 [0,1] 坐标
        local_x0 = max(canvas_bbox[0] - patch_column, 0.0)
        local_y0 = max(canvas_bbox[1] - patch_row, 0.0)
        local_x1 = min(canvas_bbox[2] - patch_column, 1.0)
        local_y1 = min(canvas_bbox[3] - patch_row, 1.0)

        # 将每个patch划分为4×8个token区域
        # ┌───┬───┬───┬───┬───┬───┬───┬───┐
        # │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
        # ├───┼───┼───┼───┼───┼───┼───┼───┤
        # │ 8 │ 9 │10 │11 │12 │13 │14 │15 │
        # ├───┼───┼───┼───┼───┼───┼───┼───┤
        # │16 │17 │18 │19 │20 │21 │22 │23 │
        # ├───┼───┼───┼───┼───┼───┼───┼───┤
        # │24 │25 │26 │27 │28 │29 │30 │31 │
        # └───┴───┴───┴───┴───┴───┴───┴───┘
        token_map = np.zeros((TOKEN_GRID_HEIGHT, TOKEN_GRID_WIDTH),dtype=np.float64,)
        
        if local_x1 > local_x0 and local_y1 > local_y0:

            # 先遍历每一行，再遍历每一列，计算当前视觉token与参与者框的重叠面积，并将其归一化为权重
            for row_index in range(TOKEN_GRID_HEIGHT):

                # 计算当前 token 行的纵向范围
                cell_y0 = row_index * cell_height        # 上端
                cell_y1 = (row_index + 1) * cell_height  # 下端
                
                # 计算目标框与这一行的纵向交集
                overlap_height = max(min(local_y1, cell_y1) - max(local_y0, cell_y0), 0.0,)
                if overlap_height <= 0.0:
                    continue

                for column_index in range(TOKEN_GRID_WIDTH):

                    # 计算当前 token 行的纵向范围
                    cell_x0 = column_index * cell_width       # 左端
                    cell_x1 = (column_index + 1) * cell_width # 右端
                    
                    # 计算目标框与这一列的横向交集
                    overlap_width = max(min(local_x1, cell_x1) - max(local_x0, cell_x0), 0.0,)
                    if overlap_width <= 0.0:
                        continue

                    # 主要actor二维框与token单元的重叠面积/token单元的面积,得到当前视觉token的权重,数值处于[0,1],可以理解为占比
                    token_map[row_index, column_index] = (overlap_width * overlap_height / cell_area)

        # 每个 4x8 的 token_map 被展开为 32 维,展开顺序是逐行展开,[[0.0],[0.4],[0.5],...[0.0]],两个patch之间直接append,所以最终patch_maps是[64,]的概率图
        patch_maps.append(token_map.reshape(-1))

    return np.concatenate(patch_maps).astype(np.float32)


def _apply_training_crop(bbox_normalized_xyxy: np.ndarray,image_width: int,image_height: int,cut_bottom_quarter: bool,) -> Tuple[np.ndarray, int, int]:
    """
    返回图像裁减后,主要actor边界框的新的[x_min, y_min, x_max, y_max] 和 裁减后的图像的尺寸
    """

    # 复制bbox_normalized_xyxy为bbox
    bbox = np.asarray(bbox_normalized_xyxy,dtype=np.float64,).copy()
    if bbox.shape != (4,):
        raise ValueError("Normalized participant bbox must contain four values.")

    # 如果不需要裁剪底部四分之一区域，直接返回原始bbox和原始图像尺寸
    if not cut_bottom_quarter:
        return bbox, int(image_width), int(image_height)

    # 计算裁减后的图像高度，裁剪掉底部约30%的区域
    crop_height = int(image_height - (image_height * 4.8) // 16)

    # 将裁减后的图像高度限制在1到原始图像高度之间,主要用于安全检查,正常情况下这里不会出问题
    crop_height = max(min(crop_height, image_height), 1)
    
    # 计算裁剪比例,即裁剪后的高度与原始高度的比值,实际上这里可以理解为裁减后图像的最底部的y坐标在原始图像中的归一化位置
    crop_ratio = float(crop_height) / float(image_height)


    # 只需要对bbox的y坐标进行裁剪和归一化处理,因为裁剪是从图像的底部进行的
    bbox[1] = np.clip(bbox[1], 0.0, crop_ratio)
    bbox[3] = np.clip(bbox[3], 0.0, crop_ratio)
    bbox[1] /= crop_ratio  # y_min 重新归一化到裁剪后的图像高度
    bbox[3] /= crop_ratio  # y_max 重新归一化到裁剪后的图像高度

    return bbox, int(image_width), int(crop_height)


def extract_participant_spatial_attention_supervision(payload: Dict,*,cut_bottom_quarter: bool,use_global_img: bool = False,) -> Tuple[np.ndarray, bool]:
    """
    从标签中的六视角参与者投影框生成6×64视觉token软标签。

    返回：
        target: [6,64],全局概率和为1
        valid: 是否存在可用的主要关键参与者视觉投影
    """


    ###################################### 安全性检查 ######################################
    
    # 创建一个空的目标数组,形状[6, 64],用于在没有有效参与者投影的情况下返回
    empty_target = np.zeros((len(CAMERA_ORDER), TOKENS_PER_CAMERA),dtype=np.float32,)

    # 1. 如果 use_global_img=True,则抛出异常,因为参与者的空间注意力目前假定每个摄像头恰好有两个图像块
    if use_global_img:
        # 参与者的空间注意力目前假定每个摄像头恰好有两个图像块;use_global_img 必须保持 False 状态.
        raise ValueError(
            "Participant spatial attention currently assumes exactly two "
            "patches per camera; use_global_img must remain False."
        )

    # 2. 获取lg标签中的 visual_grounding 字段
    visual_grounding = payload.get("visual_grounding", {})
    # 如果 visual_grounding 字段不是字典类型,返回空的目标数组和 False
    if not isinstance(visual_grounding, dict):
        return empty_target, False
    
    # 3. 如果 visual_grounding 中的 participant_spatial_attention_valid 字段为 False,返回空的目标数组和 False
    # participant_spatial_attention_valid 表示当前帧是否能够为"主要关键参与者"生成有效的六视角空间注意力监督
    # 它表示的是整帧空间监督是否有效,并不表示主要参与者在六个相机里都可见.只要它在至少一个相机中有有效投影,就可能为 True
    if not bool(visual_grounding.get("participant_spatial_attention_valid",False,)):
        return empty_target, False
    
    # 4. 如果 visual_grounding 中的 camera_order 字段与预定义的 CAMERA_ORDER 不一致,返回空的目标数组和 False
    if tuple(visual_grounding.get("camera_order", [])) != CAMERA_ORDER:
        return empty_target, False

    # 5. 获取 visual_grounding 中的 per_camera_evidence 字段
    per_camera_evidence = visual_grounding.get("per_camera_evidence",{},)
    # 如果 per_camera_evidence 字段不是字典类型,返回空的目标数组和 False
    if not isinstance(per_camera_evidence, dict):
        return empty_target, False


    ###################################### 构造"目标注意力软标签" ######################################

    # 初始化"目标注意力软标签"为与 empty_target 相同形状的零数组
    target = np.zeros_like(empty_target, dtype=np.float64)

    # 遍历六视角相机的每一个视角的相机(一共六轮循环)
    # camera_index=0, camera_name='front', camera_index=1, camera_name='front_left', camera_index=2, camera_name='front_right', camera_index=3, camera_name='rear', camera_index=4, camera_name='rear_left', camera_index=5, camera_name='rear_right'
    for camera_index, camera_name in enumerate(CAMERA_ORDER):
        
        evidence = per_camera_evidence.get(camera_name, {})  # evidence = front, front_left, front_right, rear, rear_left, rear_right
        if not isinstance(evidence, dict):
            continue
        if not bool(evidence.get("visible", False)):
            continue

        # 1. 获取主要actor归一化后的边界框信息
        # bbox_normalized_xyxy 表示主要关键actor在某一个相机图像中的二维投影框,坐标已经归一化到 [0,1],顺序为[x_min, y_min, x_max, y_max]
        bbox = np.asarray(evidence.get("bbox_normalized_xyxy", []),dtype=np.float64,)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            continue

        # 2. 获取图像的宽度和高度信息(原始图像)
        image_width = int(evidence.get("image_width", 0))
        image_height = int(evidence.get("image_height", 0))
        if image_width <= 0 or image_height <= 0:
            continue

        # 3. 获取裁减图像后主要actor归一化后的边界框信息 
        # bbox为裁减后的归一化的边界框信息,processed_width为裁减后图像的宽度,processed_height为裁减后图像的高度
        bbox, processed_width, processed_height = (
            _apply_training_crop(
                bbox,
                image_width=image_width,   # 生成该相机投影框时,当前相机原始图像的宽度,单位是像素
                image_height=image_height, # 生成该相机投影框时,当前相机原始图像的高度,单位是像素
                cut_bottom_quarter=cut_bottom_quarter,  # 是否裁减
            )
        )
        # 如果裁剪后的bbox的宽度或高度为负,则跳过该相机的处理
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        # 一般 patch_layout = (2, 1) 表示两个patch左右排列
        patch_layout = _choose_two_patch_layout(processed_width,processed_height,)

        # 🔔 将归一化的边界框转换为与视觉token严格对齐的软占用权重,返回一个形状为(64,)的数组,表示每个视觉token的权重 🔔
        token_weights = _bbox_to_patch_token_weights(bbox,patch_layout,).astype(np.float64)

        evidence_weight = float(evidence.get("raw_score", 0.0))
        if not np.isfinite(evidence_weight) or evidence_weight <= 0.0:
            continue

        # 将当前相机的视觉token权重乘以证据权重,并存储到目标数组中
        target[camera_index] = token_weights * evidence_weight

    # 计算 target 的总和,如果总和为非有限值或小于等于0,则返回空的目标数组和 False
    target_sum = float(target.sum())
    if not np.isfinite(target_sum) or target_sum <= 0.0:
        return empty_target, False

    # 归一化 target，使其总和为 1，并应用标签平滑
    target /= target_sum
    target = ((1.0 - LABEL_SMOOTHING) * target + LABEL_SMOOTHING / float(target.size))
    target /= max(float(target.sum()), 1e-12)

    return target.astype(np.float32), True
