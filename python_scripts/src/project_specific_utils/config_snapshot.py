from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


"""
yaml_safe
Convert a value to plain YAML types: tuples become lists, paths strings, and
numpy scalars/arrays Python numbers/lists; dicts and lists are converted
recursively.

INPUT:
    - value: object -> value to convert

OUTPUT:
    - converted: object -> value made of dict, list, str, int, float, bool, None
"""
def yaml_safe(value):
    if isinstance(value, dict):
        return {str(key): yaml_safe(item) for key, item in value.items()}
    # end if dict
    if isinstance(value, (list, tuple)):
        return [yaml_safe(item) for item in value]
    # end if sequence
    if isinstance(value, Path):
        return str(value)
    # end if path
    if isinstance(value, np.ndarray):
        return yaml_safe(value.tolist())
    # end if array
    if isinstance(value, np.generic):
        return value.item()
    # end if numpy scalar
    return value
# EOF


"""
save_config_snapshot
Save the parameters of one run as config_<YYYYmmdd_HHMMSS>.yaml in output_dir,
with the date and time, a free-text description of the major changes, and the
parameters that differ from the most recent earlier snapshot in the same folder
(previous and current value of each).

INPUT:
    - config: dict -> parameters of the run (e.g. dataclasses.asdict(cfg))
    - output_dir: str | Path -> folder receiving the snapshot (created if needed)
    - description: str -> brief description of the major changes
    - extra: dict | None -> further settings to store (e.g. session tables)

OUTPUT:
    - snapshot_path: Path -> written YAML file
"""
def save_config_snapshot(config, output_dir, description="", extra=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    current = yaml_safe({**config, **(extra or {})})

    # The timestamped names sort chronologically; the last one is the previous run.
    previous_paths = sorted(output_dir.glob("config_*.yaml"))
    if previous_paths:
        with open(previous_paths[-1], "r") as file:
            previous = yaml.safe_load(file).get("parameters", {})
        # end with open
        changes = {
            name: {"previous": previous.get(name), "current": value}
            for name, value in current.items()
            if previous.get(name) != value
        }
        # Parameters that existed before but are gone now.
        changes.update({
            name: {"previous": value, "current": None}
            for name, value in previous.items() if name not in current
        })
        changes_since = previous_paths[-1].name
    else:
        changes, changes_since = {}, None
    # end if previous snapshot exists

    snapshot = {
        "saved_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "description": description,
        "changes_since": changes_since,
        "changed_parameters": changes,
        "parameters": current,
    }
    snapshot_path = output_dir / f"config_{now.strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(snapshot_path, "w") as file:
        yaml.safe_dump(snapshot, file, sort_keys=False, allow_unicode=True)
    # end with open
    return snapshot_path
# EOF
