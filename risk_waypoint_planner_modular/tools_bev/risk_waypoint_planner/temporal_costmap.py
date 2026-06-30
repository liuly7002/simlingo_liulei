# -*- coding: utf-8 -*-

from .common import *
from .io_utils import load_json_gz, load_costmap
from .measurement_utils import get_meters_per_pixel, get_ego_center, get_pose_global_xy_yaw

def current_local_grid_for_costmap(
    cost_shape: Tuple[int, int],
    ego_center: List[float],
    meters_per_pixel: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build current ego-local x/y arrays for every pixel of a target costmap.

    Pixel convention:
        col = ego_center_x + y / meters_per_pixel
        row = ego_center_y - x / meters_per_pixel
    Therefore:
        x = (ego_center_y - row) * meters_per_pixel
        y = (col - ego_center_x) * meters_per_pixel
    """
    h, w = cost_shape
    rows, cols = np.indices((h, w), dtype=np.float32)
    cx, cy = float(ego_center[0]), float(ego_center[1])

    x_local = (cy - rows) * float(meters_per_pixel)
    y_local = (cols - cx) * float(meters_per_pixel)
    return x_local.astype(np.float32), y_local.astype(np.float32)

def transform_current_local_grid_to_source_local(
    x_current: np.ndarray,
    y_current: np.ndarray,
    current_measurement: Dict,
    source_measurement: Dict,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Transform a grid of points from the current ego-local frame to a source
    frame ego-local coordinate system.

    This is the key step that makes future costmaps usable for current-frame
    planning. A future costmap is normally centered/aligned with the future
    expert ego pose. We warp it back into the current ego frame before scoring
    candidate rollouts.
    """
    current_pose = get_pose_global_xy_yaw(current_measurement)
    source_pose = get_pose_global_xy_yaw(source_measurement)
    if current_pose is None or source_pose is None:
        return None

    current_pos, current_yaw = current_pose
    source_pos, source_yaw = source_pose

    c0 = math.cos(float(current_yaw))
    s0 = math.sin(float(current_yaw))

    # Current local -> global.
    global_x = current_pos[0] + x_current * c0 - y_current * s0
    global_y = current_pos[1] + x_current * s0 + y_current * c0

    # Global -> source local.
    dx = global_x - source_pos[0]
    dy = global_y - source_pos[1]

    c1 = math.cos(float(source_yaw))
    s1 = math.sin(float(source_yaw))

    x_source = dx * c1 + dy * s1
    y_source = -dx * s1 + dy * c1
    return x_source.astype(np.float32), y_source.astype(np.float32)

def warp_costmap_to_current_frame(
    source_costmap: np.ndarray,
    source_meta: Dict,
    source_measurement: Dict,
    current_measurement: Dict,
    target_shape: Tuple[int, int],
    target_ego_center: List[float],
    target_meters_per_pixel: float,
    args,
) -> Optional[np.ndarray]:
    """
    Warp a source-frame costmap into the current frame's ego-local grid.

    source_costmap is usually a future frame. Its actor positions are true
    future positions, but its coordinate system is the future expert ego frame.
    The output has the same pixel grid as the current costmap, so candidate
    rollouts generated in the current ego-local frame can be sampled directly.
    """
    # 读取未来帧的mmp和自车在costmap中的位置
    source_mpp = get_meters_per_pixel(source_meta, args.default_pixels_per_meter)
    source_ego_center = get_ego_center(source_meta, source_costmap.shape)

    # 构建当前帧每个像素对应的ego-local坐标 坐标系为x像前 y向右
    x_current, y_current = current_local_grid_for_costmap(
        cost_shape=target_shape,
        ego_center=target_ego_center,
        meters_per_pixel=target_meters_per_pixel,
    )

    # 把当前帧 local grid 转到未来帧 local grid
    transformed = transform_current_local_grid_to_source_local(
        x_current=x_current,
        y_current=y_current,
        current_measurement=current_measurement,
        source_measurement=source_measurement,
    )
    if transformed is None:
        return None

    x_source, y_source = transformed
    src_cx, src_cy = float(source_ego_center[0]), float(source_ego_center[1])

    map_x = src_cx + y_source / float(source_mpp)
    map_y = src_cy - x_source / float(source_mpp)

    warped = cv2.remap(
        source_costmap.astype(np.float32),  # 源图:未来帧costmap
        map_x.astype(np.float32),  # 每个像素应该在源图的哪个位置采样
        map_y.astype(np.float32),  # 每个像素应该在源图的哪个位置采样
        interpolation=cv2.INTER_LINEAR,  # 双线性插值
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(args.future_warp_border_cost),
    )
    return warped.astype(np.float32)

def combine_current_and_future_costmap(
    current_costmap: np.ndarray,
    future_costmap_in_current_frame: np.ndarray,
    args,
) -> np.ndarray:
    """
    Combine current-frame costmap with a warped future-frame costmap.

    Default is max because it preserves the current static cost field while
    allowing future actor risk to appear at its future true position. Additive
    fusion is available but can over-count static road/background cost if the
    future costmap is not actor-only.
    """
    # 未来 costmap 先乘以一个权重 如果想要削弱未来风险 可以设置为0.5 想强调未来风险可以设置为2.0 原样使用未来风险就设置为1.0
    future = future_costmap_in_current_frame.astype(np.float32) * float(args.future_costmap_weight)
    current = current_costmap.astype(np.float32)

    mode = str(args.future_costmap_combine).lower()
    if mode == "max":  # 最合理的模式 逐像素取最大值
        combined = np.maximum(current, future)
    elif mode == "add":
        combined = current + future
    elif mode == "future_only":
        combined = future
    else:
        raise ValueError(f"Unsupported future_costmap_combine: {args.future_costmap_combine}")

    if float(args.temporal_cost_clip_max) > 0.0:
        combined = np.clip(combined, 0.0, float(args.temporal_cost_clip_max))

    return combined.astype(np.float32)

def costmap_to_debug_bgr(
    cost: np.ndarray,
    title: str = "",
    ego_center: Optional[List[float]] = None,
    vmax: float = 0.0,
) -> np.ndarray:
    """
    Convert a 2D costmap to a colored BGR debug image.

    This function is only used for visualization. It does not change planning.
    """
    cost = np.asarray(cost, dtype=np.float32)
    finite = np.isfinite(cost)

    if not np.any(finite):
        gray = np.zeros(cost.shape, dtype=np.uint8)
    else:
        vals = cost[finite]

        if float(vmax) <= 0.0:
            v_max = float(np.percentile(vals, 99.5))
            if v_max <= 1e-6:
                v_max = float(np.max(vals))
            if v_max <= 1e-6:
                v_max = 1.0
        else:
            v_max = float(vmax)

        norm = np.clip(cost / v_max, 0.0, 1.0)
        gray = (norm * 255.0).astype(np.uint8)

    img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    if ego_center is not None:
        cx = int(round(float(ego_center[0])))
        cy = int(round(float(ego_center[1])))
        cv2.circle(img, (cx, cy), radius=3, color=(255, 255, 255), thickness=-1)
        cv2.circle(img, (cx, cy), radius=6, color=(0, 0, 0), thickness=1)

    if title:
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), thickness=-1)
        cv2.putText(
            img,
            title,
            (5, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return img

def save_cumulative_temporal_costmap_debug(
    route_dir: Path,
    frame_name: str,
    cumulative_costmap: np.ndarray,
    current_ego_center: List[float],
    fused_future_frames: List[str],
    args,
) -> None:
    """
    Save one cumulative temporal costmap for the current frame.

    This debug map is:
        current costmap + all successfully warped future costmaps

    It is only for visualization/debug. It is not used by score_rollout().
    """
    debug_dir = route_dir / args.temporal_debug_folder
    debug_dir.mkdir(parents=True, exist_ok=True)

    num_future = len(fused_future_frames)
    title = f"{frame_name}: current + {num_future} warped future costmaps"

    img = costmap_to_debug_bgr(
        cumulative_costmap,
        title=title,
        ego_center=current_ego_center,
        vmax=float(args.temporal_debug_vmax),
    )

    out_png = debug_dir / f"{frame_name}_cumulative_temporal_costmap.png"
    cv2.imwrite(str(out_png), img)

    if args.temporal_debug_save_npy:
        out_npy = debug_dir / f"{frame_name}_cumulative_temporal_costmap.npy"
        np.save(str(out_npy), cumulative_costmap.astype(np.float32))

    out_txt = debug_dir / f"{frame_name}_cumulative_temporal_costmap_frames.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"current_frame: {frame_name}\n")
        f.write(f"num_fused_future_frames: {num_future}\n")
        f.write("fused_future_frames:\n")
        for name in fused_future_frames:
            f.write(f"  {name}\n")

def offset_frame_name(frame_name: str, offset: int, stride: int = 1) -> Optional[str]:
    try:
        frame_int = int(frame_name)
    except ValueError:
        return None
    width = len(frame_name)
    return str(frame_int + int(offset) * int(stride)).zfill(width)

def append_missing_temporal_map(
    temporal_costmaps: List[np.ndarray],
    current_costmap: np.ndarray,
    args,
) -> np.ndarray:
    policy = str(args.future_missing_policy).lower()
    if policy == "repeat_last" and len(temporal_costmaps) > 0:
        return temporal_costmaps[-1].copy()
    if policy == "current":
        return current_costmap.copy()
    if policy == "zero":
        return np.zeros_like(current_costmap, dtype=np.float32)
    raise ValueError(f"Unsupported future_missing_policy: {args.future_missing_policy}")

def build_temporal_costmap_bundle(
    route_dir: Path,    # 当前route的根目录，包含costmaps、measurements、meta等子目录
    frame_name: str,    # 当前帧编号,例如"0023" 根据frame_name和offset计算未来帧编号
    current_measurement: Dict,  # 当前帧的measurement的内容
    current_costmap: np.ndarray,# 当前帧的costmap数据 二维数组,[H,W]
    current_meta: Dict,  # 当前帧的meta信息
    current_ego_center: List[float],  # 当前costmap中自车中心的像素坐标,一般为[256/2, 256/2]
    current_meters_per_pixel: float,  # 当前 costmap 每个像素对应多少米,一般为0.2
    args,
) -> Dict:
    """
    Build a time-indexed list of costmaps for rollout scoring.

    Index 0 is the current costmap. Index k>0 is frame+k costmap warped into
    the current frame. This lets the planner evaluate candidate ego positions
    against surrounding vehicles' future true positions.
    """
    #################### 初始化 temporal bundle ####################
    temporal_costmaps: List[np.ndarray] = [current_costmap.astype(np.float32)]
    temporal_frames: List[str] = [frame_name]
    temporal_valid: List[bool] = [True]  # 记录每个 temporal costmap 是否有效，当前帧的costmap默认有效
    temporal_warped: List[bool] = [False]  # 记录这张 costmap 是否经过了 warp, 当前帧的cost_map本来就是在当前帧坐标系下的，所以不算warp
    missing_reasons: List[str] = []  # 记录未来帧costmap缺失的原因，方便后续分析

    # Cumulative temporal debug map:
    # current costmap + all successfully warped future costmaps.
    # This is only for debug visualization, not for scoring.
    cumulative_temporal_costmap = current_costmap.astype(np.float32).copy()
    fused_future_frames: List[str] = []

    # 如果选择禁用未来帧costmap(也就是只使用当前帧costmap,这意味着后续score_rollout只能使用当前帧costmap评分),直接返回当前帧的costmap和相关信息，未来帧部分标记为无效且未启用
    # 做消融实验时会用到这个选项，看看完全不使用未来帧costmap的效果
    if args.disable_future_costmaps:
        return {
            "costmaps": temporal_costmaps,
            "frames": temporal_frames,
            "valid": temporal_valid,
            "warped": temporal_warped,
            "missing_reasons": missing_reasons,
            "enabled": False,
        }

    # 选择 costmap 和 meta 文件夹，如果使用增强数据则选择增强数据的文件夹，否则选择原始数据的文件夹
    costmap_folder = args.costmap_folder_augmented if args.augmented else args.costmap_folder
    meta_folder = args.bev_meta_folder_augmented if args.augmented else args.bev_meta_folder

    # 遍历未来帧
    for offset in range(1, int(args.num_future_waypoints) + 1):
        future_name = offset_frame_name(frame_name, offset, stride=args.future_frame_stride)
        if future_name is None:
            missing_reasons.append(f"offset={offset}: non_numeric_frame_name")
            temporal_costmaps.append(append_missing_temporal_map(temporal_costmaps, current_costmap, args))
            temporal_frames.append(f"missing+{offset}")
            temporal_valid.append(False)
            temporal_warped.append(False)
            continue

        # 根据未来帧编号构建未来帧的costmap、measurement和meta文件路径
        future_costmap_path = route_dir / costmap_folder / f"{future_name}.npy"
        future_measurement_path = route_dir / args.measurements_folder / f"{future_name}.json.gz"
        future_meta_path = route_dir / meta_folder / f"{future_name}.json.gz"
        # 如果未来帧的costmap或measurement文件不存在，记录缺失原因，并在temporal bundle中添加一个无效的占位costmap
        if not future_costmap_path.exists() or not future_measurement_path.exists():
            missing_reasons.append(
                f"offset={offset}, frame={future_name}: missing future costmap or measurement"
            )
            temporal_costmaps.append(append_missing_temporal_map(temporal_costmaps, current_costmap, args))
            temporal_frames.append(future_name)
            temporal_valid.append(False)
            temporal_warped.append(False)
            continue
        # 读取未来帧的costmap、measurement和meta信息，如果meta信息不存在则使用空字典
        future_costmap = load_costmap(future_costmap_path)
        future_measurement = load_json_gz(future_measurement_path)
        future_meta = load_json_gz(future_meta_path) if future_meta_path.exists() else {}

        # 关键步骤：将未来帧的costmap warp到当前帧的坐标系下，以便后续score_rollout可以直接采样比较
        warped = warp_costmap_to_current_frame(
            source_costmap=future_costmap,  # 未来帧的costmap数据
            source_meta=future_meta,  # 未来帧的meta信息
            source_measurement=future_measurement,  # 未来帧的measurement信息
            current_measurement=current_measurement,  # 当前帧的measurement信息
            target_shape=current_costmap.shape,  # 输出的costmap形状和当前帧的costmap一样
            target_ego_center=current_ego_center,  # 输出costmap中自车中心的像素坐标和当前帧一样
            target_meters_per_pixel=current_meters_per_pixel,  # 输出costmap的米/像素和当前帧一样
            args=args,
        )

        if warped is None:
            missing_reasons.append(
                f"offset={offset}, frame={future_name}: missing pos_global/theta for temporal warp"
            )
            temporal_costmaps.append(append_missing_temporal_map(temporal_costmaps, current_costmap, args))
            temporal_frames.append(future_name)
            temporal_valid.append(False)
            temporal_warped.append(False)
            continue

        # 融合当前帧costmap和warped的未来帧costmap
        combined = combine_current_and_future_costmap(
            current_costmap=current_costmap,
            future_costmap_in_current_frame=warped,
            args=args,
        )
        
        # Update cumulative temporal debug costmap.
        # This accumulates all warped future costmaps into one complete map.
        weighted_warped = warped.astype(np.float32) * float(args.future_costmap_weight)
        cumulative_mode = str(args.future_costmap_combine).lower()

        if cumulative_mode == "max":
            cumulative_temporal_costmap = np.maximum(
                cumulative_temporal_costmap,
                weighted_warped,
            )
        elif cumulative_mode == "add":
            cumulative_temporal_costmap = cumulative_temporal_costmap + weighted_warped
        elif cumulative_mode == "future_only":
            # For debug visualization, future_only means accumulate future risk only.
            if len(fused_future_frames) == 0:
                cumulative_temporal_costmap = weighted_warped.copy()
            else:
                cumulative_temporal_costmap = np.maximum(
                    cumulative_temporal_costmap,
                    weighted_warped,
                )
        else:
            raise ValueError(f"Unsupported future_costmap_combine: {args.future_costmap_combine}")

        if float(args.temporal_cost_clip_max) > 0.0:
            cumulative_temporal_costmap = np.clip(
                cumulative_temporal_costmap,
                0.0,
                float(args.temporal_cost_clip_max),
            ).astype(np.float32)

        fused_future_frames.append(future_name)

        temporal_costmaps.append(combined)
        temporal_frames.append(future_name)
        temporal_valid.append(True)
        temporal_warped.append(True)

    # Save one cumulative temporal debug costmap for the current frame.
    # This contains current costmap + all successfully warped future costmaps.
    if args.save_temporal_debug:
        save_cumulative_temporal_costmap_debug(
            route_dir=route_dir,
            frame_name=frame_name,
            cumulative_costmap=cumulative_temporal_costmap,
            current_ego_center=current_ego_center,
            fused_future_frames=fused_future_frames,
            args=args,
        )

    return {
        "costmaps": temporal_costmaps,
        "frames": temporal_frames,
        "valid": temporal_valid,
        "warped": temporal_warped,
        "missing_reasons": missing_reasons,
        "enabled": True,
    }

def temporal_index_for_dense_step(step_idx: int, num_temporal_maps: int, args) -> int:
    """
    Map a dense bicycle-model step to a temporal costmap index.

    The rollout is integrated at model_fps. Future costmaps are assumed to be
    stored at future_fps with frame stride args.future_frame_stride. With the
    default model_fps=20 and future_fps=4, every 5 integration steps correspond
    to one output waypoint / one future frame.
    """
    if num_temporal_maps <= 1:
        return 0

    t = float(step_idx + 1) / float(args.model_fps)
    raw = t * float(args.future_fps) / max(float(args.future_frame_stride), 1.0)
    mode = str(args.temporal_index_mode).lower()

    if mode == "floor":
        idx = int(math.floor(raw))
    elif mode == "ceil":
        idx = int(math.ceil(raw))
    elif mode == "round":
        idx = int(math.floor(raw + 0.5))
    else:
        raise ValueError(f"Unsupported temporal_index_mode: {args.temporal_index_mode}")

    return int(np.clip(idx, 0, num_temporal_maps - 1))
