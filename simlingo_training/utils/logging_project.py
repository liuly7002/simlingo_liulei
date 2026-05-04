import json
import os
import glob
import subprocess
import argparse
import logging

from git import Repo
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime

from hydra.utils import get_original_cwd, to_absolute_path


def setup_logging(cfg, save_folder=None):
    
    if save_folder is None:
        working_dir = get_original_cwd()
        save_folder = 'log'
    else:
        # get working dir
        working_dir = os.getcwd()
        save_folder = save_folder + '/log'
    # Log args
    # Path(save_folder).mkdir(parents=True, exist_ok=True)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    arg_dict = OmegaConf.to_container(cfg, resolve=True)
    args = argparse.Namespace(**arg_dict)
    # with open(os.path.join(save_folder, "args.txt"), "w") as f:
    with open(os.path.join(save_folder, "args.txt"), "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=2)

    # Log git
    # sha = (
    #     subprocess.check_output(
    #         ["git", "-C", f"{working_dir}", "rev-parse", "HEAD"]
    #     )
    #     .decode("ascii")
    #     .strip()
    # )
    # commit = (
    #     subprocess.check_output(["git", "-C", f"{working_dir}", "log", "-1"])
    #     .decode("ascii")
    #     .strip()
    # )
    # branch = (
    #     subprocess.check_output(["git", "-C", f"{working_dir}", "branch"])
    #     .decode("ascii")
    #     .strip()
    # )
    # repo = Repo(working_dir)

    # Log git
    try:
        sha = (
            subprocess.check_output(
                ["git", "-C", f"{working_dir}", "rev-parse", "HEAD"],
                stderr=subprocess.STDOUT
            )
            .decode("utf-8", errors="replace")
            .strip()
        )

        commit = (
            subprocess.check_output(
                ["git", "-C", f"{working_dir}", "log", "-1"],
                stderr=subprocess.STDOUT
            )
            .decode("utf-8", errors="replace")
            .strip()
        )

        branch = (
            subprocess.check_output(
                ["git", "-C", f"{working_dir}", "branch"],
                stderr=subprocess.STDOUT
            )
            .decode("utf-8", errors="replace")
            .strip()
        )

        repo = Repo(working_dir)
        git_diff = repo.git.diff("HEAD")

    except Exception as e:
        sha = "Failed to read git sha"
        commit = f"Failed to read git commit: {repr(e)}"
        branch = "Failed to read git branch"
        git_diff = ""

    # with open(os.path.join(save_folder, "git_info.txt"), "w") as f:
    with open(os.path.join(save_folder, "git_info.txt"), "w", encoding="utf-8") as f:
        # write current date and time
        f.write(
            f"Run started at: {str(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))}\n"
        )
        f.write(f"Git state: {sha}\n")
        f.write(f"Git commit: {commit}\n")
        f.write(f"Git branch: {branch}\n\n")
        # f.write(f"{repo.git.diff('HEAD')}")
        f.write(f"{git_diff}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def sync_wandb(cfg):
    # TODO: sync wandb - still not working correctly
    wandb_files = glob.glob(f"./wandb/offline*/*.wandb")
    os.environ["TMPDIR"] = "/home/geiger/krenz73/tmp"
    for wandb_file in wandb_files:
        if os.path.getsize(wandb_file) > 5000000:
            os.system(f"wandb sync {wandb_file}")