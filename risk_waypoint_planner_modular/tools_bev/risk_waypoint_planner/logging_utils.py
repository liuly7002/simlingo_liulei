# -*- coding: utf-8 -*-

from .common import *

def setup_logging(args, input_path: Path) -> Path:
    """
    Setup file + optional console logging.

    If --log_file is not specified, the log will be saved to:
        input_path / log_folder / plan_waypoints_bicycle_YYYYMMDD_HHMMSS.log
    """
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.propagate = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.log_file is None:
        log_dir = input_path / args.log_folder
        log_path = log_dir / f"plan_waypoints_bicycle_{timestamp}.log"
    else:
        log_path = Path(args.log_file).expanduser().resolve()

    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    if not args.quiet_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(console_handler)

    LOGGER.info(f"[Log] log_file={log_path}")
    return log_path
