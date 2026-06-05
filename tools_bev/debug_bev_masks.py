import argparse
from pathlib import Path

import cv2
import numpy as np


"""
python tools_bev/debug_bev_masks.py --input data/town01_original/routes_town01_xxx
"""


# OpenCV uses BGR color order.
COLOR = {
    "black": (0, 0, 0),
    "road": (46, 52, 54),          # dark gray
    "sidewalk": (128, 128, 128),   # gray
    "lane_all": (255, 0, 255),     # magenta
    "lane_broken": (255, 140, 255),
    "lane_solid": (180, 0, 180),

    "vehicle": (255, 0, 0),        # blue in BGR
    "walker": (255, 255, 0),       # cyan in BGR
    "actor": (255, 0, 0),

    "stop": (0, 180, 255),         # orange/yellow
    "tl_green": (0, 255, 0),
    "tl_yellow": (0, 255, 255),
    "tl_red": (0, 0, 255),

    "white": (255, 255, 255),
}


def load_npz(path: Path) -> dict:
    if path is None or not path.exists():
        return {}

    data = np.load(str(path))
    return {k: data[k] for k in data.files}


def as_bool(mask):
    return mask.astype(np.uint8) > 0


def infer_shape(*dicts):
    for d in dicts:
        for v in d.values():
            return v.shape
    raise ValueError("Cannot infer mask shape because all mask dicts are empty.")


def save_binary_masks(mask_dict: dict, save_dir: Path, prefix: str):
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, mask in mask_dict.items():
        out = (as_bool(mask).astype(np.uint8) * 255)
        cv2.imwrite(str(save_dir / f"{prefix}_{name}.png"), out)


def colorize_static(static_masks: dict, shape):
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

    road = as_bool(static_masks["road"]) if "road" in static_masks else None
    sidewalk = as_bool(static_masks["sidewalk"]) if "sidewalk" in static_masks else None
    lane_all = as_bool(static_masks["lane_all"]) if "lane_all" in static_masks else None
    lane_broken = as_bool(static_masks["lane_broken"]) if "lane_broken" in static_masks else None
    lane_solid = as_bool(static_masks["lane_solid"]) if "lane_solid" in static_masks else None

    if road is not None:
        image[road] = COLOR["road"]

    if sidewalk is not None:
        image[sidewalk] = COLOR["sidewalk"]

    if lane_all is not None:
        image[lane_all] = COLOR["lane_all"]

    if lane_solid is not None:
        image[lane_solid] = COLOR["lane_solid"]

    if lane_broken is not None:
        image[lane_broken] = COLOR["lane_broken"]

    return image


def colorize_dynamic(dynamic_masks: dict, shape):
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

    vehicle = as_bool(dynamic_masks["vehicle"]) if "vehicle" in dynamic_masks else None
    walker = as_bool(dynamic_masks["walker"]) if "walker" in dynamic_masks else None
    actor = as_bool(dynamic_masks["actor"]) if "actor" in dynamic_masks else None

    if actor is not None:
        image[actor] = COLOR["actor"]

    if vehicle is not None:
        image[vehicle] = COLOR["vehicle"]

    if walker is not None:
        image[walker] = COLOR["walker"]

    return image


def colorize_traffic(traffic_masks: dict, shape):
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

    stop = as_bool(traffic_masks["stop"]) if "stop" in traffic_masks else None
    tl_green = as_bool(traffic_masks["tl_green"]) if "tl_green" in traffic_masks else None
    tl_yellow = as_bool(traffic_masks["tl_yellow"]) if "tl_yellow" in traffic_masks else None
    tl_red = as_bool(traffic_masks["tl_red"]) if "tl_red" in traffic_masks else None

    if stop is not None:
        image[stop] = COLOR["stop"]

    if tl_green is not None:
        image[tl_green] = COLOR["tl_green"]

    if tl_yellow is not None:
        image[tl_yellow] = COLOR["tl_yellow"]

    if tl_red is not None:
        image[tl_red] = COLOR["tl_red"]

    return image


def colorize_combined(static_masks: dict, dynamic_masks: dict, traffic_masks: dict, shape):
    image = colorize_static(static_masks, shape)

    # Traffic masks overwrite static masks.
    traffic_img = colorize_traffic(traffic_masks, shape)
    traffic_nonzero = np.any(traffic_img > 0, axis=-1)
    image[traffic_nonzero] = traffic_img[traffic_nonzero]

    # Dynamic masks overwrite everything else.
    dynamic_img = colorize_dynamic(dynamic_masks, shape)
    dynamic_nonzero = np.any(dynamic_img > 0, axis=-1)
    image[dynamic_nonzero] = dynamic_img[dynamic_nonzero]

    return image


def maybe_rotate(image, rotate):
    if rotate == "none":
        return image
    if rotate == "cw":
        return np.rot90(image, k=-1)
    if rotate == "ccw":
        return np.rot90(image, k=1)
    if rotate == "180":
        return np.rot90(image, k=2)
    raise ValueError(f"Unknown rotate mode: {rotate}")


def process_one_frame(route_dir: Path, out_dir: Path, frame_name: str, augmented=False,
                      save_individual=False, rotate="none"):
    suffix = "_augmented" if augmented else ""

    static_path = route_dir / f"bev_static_masks{suffix}" / f"{frame_name}.npz"
    dynamic_path = route_dir / f"bev_dynamic_masks{suffix}" / f"{frame_name}.npz"
    traffic_path = route_dir / f"bev_traffic_masks{suffix}" / f"{frame_name}.npz"

    static_masks = load_npz(static_path)
    dynamic_masks = load_npz(dynamic_path)
    traffic_masks = load_npz(traffic_path)

    if not static_masks and not dynamic_masks and not traffic_masks:
        print(f"[Skip] No masks found for frame {frame_name} in {route_dir}")
        return

    shape = infer_shape(static_masks, dynamic_masks, traffic_masks)

    static_img = colorize_static(static_masks, shape)
    dynamic_img = colorize_dynamic(dynamic_masks, shape)
    traffic_img = colorize_traffic(traffic_masks, shape)
    combined_img = colorize_combined(static_masks, dynamic_masks, traffic_masks, shape)

    static_img = maybe_rotate(static_img, rotate)
    dynamic_img = maybe_rotate(dynamic_img, rotate)
    traffic_img = maybe_rotate(traffic_img, rotate)
    combined_img = maybe_rotate(combined_img, rotate)

    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / f"{frame_name}_static.png"), static_img)
    cv2.imwrite(str(out_dir / f"{frame_name}_dynamic.png"), dynamic_img)
    cv2.imwrite(str(out_dir / f"{frame_name}_traffic.png"), traffic_img)
    cv2.imwrite(str(out_dir / f"{frame_name}_combined.png"), combined_img)

    if save_individual:
        individual_dir = out_dir / f"{frame_name}_individual"
        save_binary_masks(static_masks, individual_dir, "static")
        save_binary_masks(dynamic_masks, individual_dir, "dynamic")
        save_binary_masks(traffic_masks, individual_dir, "traffic")

    print(f"[OK] Saved debug PNGs for {route_dir.name}/{frame_name} -> {out_dir}")


def find_route_dirs(root: Path, augmented=False):
    suffix = "_augmented" if augmented else ""

    route_dirs = []
    for p in root.rglob(f"bev_static_masks{suffix}"):
        if p.is_dir():
            route_dirs.append(p.parent)

    return sorted(set(route_dirs))


def process_route_dir(route_dir: Path, out_root: Path, frame=None, augmented=False,
                      save_individual=False, rotate="none"):
    suffix = "_augmented" if augmented else ""
    static_dir = route_dir / f"bev_static_masks{suffix}"

    if not static_dir.exists():
        print(f"[Skip] {static_dir} does not exist.")
        return

    if frame is not None:
        frame_names = [frame.replace(".npz", "")]
    else:
        frame_names = sorted([p.stem for p in static_dir.glob("*.npz")])

    if len(frame_names) == 0:
        print(f"[Skip] No npz files found in {static_dir}")
        return

    out_dir = out_root / route_dir.name / ("augmented" if augmented else "normal")

    for frame_name in frame_names:
        process_one_frame(
            route_dir=route_dir,
            out_dir=out_dir,
            frame_name=frame_name,
            augmented=augmented,
            save_individual=save_individual,
            rotate=rotate,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Route directory or data root. Example: data/town01_original/routes_xxx",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Default: <input>/mask_debug_png if input is a route dir, otherwise ./mask_debug_png",
    )
    parser.add_argument(
        "--frame",
        type=str,
        default=None,
        help="Only process one frame, e.g. 0000 or 0000.npz",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively find all route folders containing bev_static_masks.",
    )
    parser.add_argument(
        "--augmented",
        action="store_true",
        help="Use bev_static_masks_augmented / bev_dynamic_masks_augmented / bev_traffic_masks_augmented.",
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Also save each binary mask as a separate PNG.",
    )
    parser.add_argument(
        "--rotate",
        type=str,
        default="none",
        choices=["none", "cw", "ccw", "180"],
        help="Rotate output PNG for visualization only. Default: none.",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()

    if args.output is None:
        if args.recursive:
            out_root = Path("mask_debug_png").resolve()
        else:
            out_root = input_path / "mask_debug_png"
    else:
        out_root = Path(args.output).resolve()

    if args.recursive:
        route_dirs = find_route_dirs(input_path, augmented=args.augmented)
        print(f"[Info] Found {len(route_dirs)} route dirs.")

        for route_dir in route_dirs:
            process_route_dir(
                route_dir=route_dir,
                out_root=out_root,
                frame=args.frame,
                augmented=args.augmented,
                save_individual=args.save_individual,
                rotate=args.rotate,
            )
    else:
        process_route_dir(
            route_dir=input_path,
            out_root=out_root,
            frame=args.frame,
            augmented=args.augmented,
            save_individual=args.save_individual,
            rotate=args.rotate,
        )


if __name__ == "__main__":
    main()