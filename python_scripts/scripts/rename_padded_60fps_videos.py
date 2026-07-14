import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cfg:
    padded_dir: Path = Path(
        "/Users/tizianocausin/sd_local/possible_vids/final_video_selection/yes/"
        "standardized_60fps/padded_60fps"
    )
    mode: str = "apply"
    mapping_file: Path | None = None


"""
parse_args
Parse command line arguments into a config object.

OUTPUT:
    - cfg: Cfg -> script configuration
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename padded 60 fps videos and matching image-videos with a reversible CSV mapping."
    )
    parser.add_argument("--padded_dir", type=Path, default=Cfg.padded_dir)
    parser.add_argument("--mapping_file", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["apply", "restore"],
        default=Cfg.mode,
        help="apply short names, or restore original names from the mapping CSV.",
    )
    args = parser.parse_args()

    return Cfg(
        padded_dir=args.padded_dir,
        mapping_file=args.mapping_file,
        mode=args.mode,
    )


def base_name(path):
    stem = path.stem
    stem = re.sub(r"__start_.*$", "", stem)
    stem = re.sub(r"__crop_.*$", "", stem)
    stem = re.sub(r"_(?:\d{2}_\d{2}_\d{2}(?:_\d+)?)$", "", stem)
    return stem


def read_mapping(mapping_file):
    with mapping_file.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    mapping = [(row["old_name"], row["new_name"]) for row in rows]
    if not mapping:
        raise RuntimeError(f"No mapping rows found in {mapping_file}")
    return mapping


def write_mapping(mapping_file, mapping):
    with mapping_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["old_name", "new_name"])
        writer.writerows(mapping)


"""
build_mapping
Build old-to-new short filename mapping from current long padded video names.

INPUT:
    - padded_dir: Path -> folder with padded videos

OUTPUT:
    - mapping: list[tuple[str, str]] -> old and new filenames
"""
def build_mapping(padded_dir):
    paths = sorted(
        path for path in padded_dir.glob("*.mov")
        if not path.name.startswith(".") and path.is_file()
    )
    if not paths:
        raise RuntimeError(f"No .mov videos found in {padded_dir}")

    groups = defaultdict(list)
    for path in paths:
        groups[base_name(path)].append(path)

    mapping = []
    for base in sorted(groups):
        for idx, path in enumerate(sorted(groups[base], key=lambda p: p.name), start=1):
            mapping.append((path.name, f"{base}{idx}.mov"))

    new_names = [new_name for _, new_name in mapping]
    if len(new_names) != len(set(new_names)):
        raise RuntimeError("Renaming would create duplicate destination names")

    return mapping


def check_matching_image_videos(padded_dir, images_dir, current_names):
    image_names = {path.name for path in images_dir.glob("*.mov") if path.is_file()}
    missing = sorted(current_names - image_names)
    if missing:
        raise RuntimeError(f"Missing matching image-videos: {missing[:5]}")


def apply_mapping_to_folder(folder, mapping):
    current_names = {path.name for path in folder.glob("*.mov") if path.is_file()}
    source_names = {old_name for old_name, _ in mapping}
    destination_names = {new_name for _, new_name in mapping}

    missing = sorted(source_names - current_names)
    if missing:
        raise RuntimeError(f"Missing source files in {folder}: {missing[:5]}")

    blocked = sorted((destination_names - source_names) & current_names)
    if blocked:
        raise RuntimeError(f"Destination files already exist in {folder}: {blocked[:5]}")

    for old_name, _ in mapping:
        tmp_name = f".renaming_tmp__{old_name}"
        (folder / old_name).rename(folder / tmp_name)

    for old_name, new_name in mapping:
        tmp_name = f".renaming_tmp__{old_name}"
        (folder / tmp_name).rename(folder / new_name)


def main():
    cfg = parse_args()
    cfg.padded_dir = cfg.padded_dir.expanduser().resolve()
    images_dir = cfg.padded_dir / "images"
    cfg.mapping_file = (cfg.mapping_file or cfg.padded_dir / "rename_mapping.csv").expanduser().resolve()

    if not images_dir.exists():
        raise RuntimeError(f"Missing images folder: {images_dir}")

    if cfg.mode == "apply":
        mapping = build_mapping(cfg.padded_dir)
        current_names = {old_name for old_name, _ in mapping}
        check_matching_image_videos(cfg.padded_dir, images_dir, current_names)
        apply_mapping_to_folder(cfg.padded_dir, mapping)
        apply_mapping_to_folder(images_dir, mapping)
        write_mapping(cfg.mapping_file, mapping)

    if cfg.mode == "restore":
        mapping = read_mapping(cfg.mapping_file)
        reverse_mapping = [(new_name, old_name) for old_name, new_name in mapping]
        current_names = {old_name for old_name, _ in reverse_mapping}
        check_matching_image_videos(cfg.padded_dir, images_dir, current_names)
        apply_mapping_to_folder(cfg.padded_dir, reverse_mapping)
        apply_mapping_to_folder(images_dir, reverse_mapping)

    print(f"{cfg.mode} complete for {cfg.padded_dir}")
    print(f"Mapping file: {cfg.mapping_file}")


if __name__ == "__main__":
    main()


# EOF
