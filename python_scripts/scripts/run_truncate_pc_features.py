import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import yaml


# Keep the leading components of stored PC features, e.g. dataset PC500 from
# the PC1000 projection. Example from the project root:
# .venv/bin/python python_scripts/scripts/run_truncate_pc_features.py \
#     --model_name ijepa_vith14_1k --source_pooling PC1000 --n_keep 500


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
# end with open

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from image_processing.frame_pca import truncate_pc_feature_file
from image_processing.video_feature_extraction import list_video_feature_files


@dataclass
class Cfg:
    model_name: str = "ijepa_vith14_1k"
    dataset_name: str = "static_dynamic"
    source_pooling: str = "PC1000"
    n_keep: int = 500
# EOF


"""
main
Truncate every layer file of one model; the new tag is "<source>first<k>".
"""
def main() -> None:
    parser = argparse.ArgumentParser(description="Truncate PC feature files.")
    parser.add_argument("--model_name", default=Cfg.model_name)
    parser.add_argument("--dataset_name", default=Cfg.dataset_name)
    parser.add_argument("--source_pooling", default=Cfg.source_pooling)
    parser.add_argument("--n_keep", type=int, default=Cfg.n_keep)
    cfg = Cfg(**vars(parser.parse_args()))

    models_dir = Path(paths["data_path"]) / "models"
    target_pooling = f"{cfg.source_pooling}first{cfg.n_keep}"
    for source_path in list_video_feature_files(
            models_dir, cfg.model_name, cfg.dataset_name, cfg.source_pooling):
        # Only the pooling tag changes in the filename.
        target_path = source_path.with_name(source_path.name.replace(
            f"_{cfg.source_pooling}pool.h5", f"_{target_pooling}pool.h5",
        ))
        truncate_pc_feature_file(source_path, target_path, cfg.n_keep, target_pooling)
        print(f"wrote {target_path.name}", flush=True)
    # end for source_path
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
