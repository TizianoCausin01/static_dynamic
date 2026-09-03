import argparse
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
# end with open

if ENV not in config:
    raise KeyError(f"MY_ENV={ENV!r} is not defined in {PROJECT_ROOT / 'config.yaml'}.")
# end if ENV not in config

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from image_processing.video_feature_extraction import (
    list_video_feature_files,
    match_feature_stimulus_names,
)
from project_specific_utils import (
    build_neural_model_rsa_filename,
    compute_layer_neural_model_rsa,
    compute_neural_rdm_timeseries,
    load_natraster,
    min_max_normalization,
    normalize_rsa_metric,
    save_layer_neural_model_rsa,
    select_stimulus_rasters,
)
from useful_stuff.general_utils import TimeSeries, print_wise


@dataclass
class Cfg:
    monkey_name: str = "red"
    static_experiment_name: str = "20260726to27"
    dynamic_experiment_name: str = "20260720to24"
    experiment_name: str | None = None
    static_path: str | None = None
    dynamic_path: str | None = None
    output_dir: str | None = None

    # MATLAB channel numbers are one-based and both endpoints are inclusive.
    good_channels: tuple[int, int] | None = None
    reliable_channels_config: str | None = None
    reliable_channels_key: str | None = None
    source_fs: float = 1000
    new_fs: float = 100
    static_crop_ms: float | None = 1000
    dynamic_crop_ms: float | None = None
    normalization: str | None = None

    model_name: str = "dino_v3_l"
    model_dataset_name: str = "static_dynamic"
    model_pooling: str | None = "mean"
    model_features_dir: str | None = None
    model_frame_index: int = -1

    signal_rdm_metric: str = "cosine_cnt"
    model_rdm_metric: str = "cosine_cnt"
    rsa_metric: str = "pearson"

    parallel: bool = False
    overwrite: bool = False
    max_layers: int | None = None
# EOF


@dataclass
class AnalysisState:
    static_neural_rdms: np.ndarray
    dynamic_neural_rdms: np.ndarray
    static_feature_names: list[str]
    dynamic_feature_names: list[str]
    static_stimulus_names: list[str]
    dynamic_stimulus_names: list[str]
    selected_channel_numbers: np.ndarray
    static_time_ms: np.ndarray
    dynamic_time_ms: np.ndarray
# EOF


"""
parse_args
Parse and validate the static/dynamic neural-to-model RSA configuration.

OUTPUT:
    - cfg: Cfg -> validated analysis configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Compute static final-frame RSA and dynamic cross-temporal RSA "
            "between neural responses and every saved network layer."
        )
    )
    parser.add_argument("--monkey_name", default=Cfg.monkey_name)
    parser.add_argument(
        "--static_experiment_name", default=Cfg.static_experiment_name,
        help="Static session suffix, or a full label beginning with monkey_name.",
    )
    parser.add_argument(
        "--dynamic_experiment_name", default=Cfg.dynamic_experiment_name,
        help="Dynamic session suffix, or a full label beginning with monkey_name.",
    )
    parser.add_argument(
        "--experiment_name",
        help="Result label; defaults to DYNAMIC_LABEL_vs_STATIC_LABEL.",
    )
    parser.add_argument("--static_path")
    parser.add_argument("--dynamic_path")
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--good_channels", nargs=2, type=int, metavar=("FIRST", "LAST"),
        help="Inclusive one-based MATLAB channel range; default uses all channels.",
    )
    parser.add_argument(
        "--reliable_channels_config",
        help=(
            "YAML reliability file. Its non-contiguous channel list is "
            "intersected with --good_channels when both are supplied."
        ),
    )
    parser.add_argument(
        "--reliable_channels_key",
        help=(
            "Dataset key in the reliability YAML; defaults to the complete "
            "static session label."
        ),
    )
    parser.add_argument("--source_fs", type=float, default=Cfg.source_fs)
    parser.add_argument("--new_fs", type=float, default=Cfg.new_fs)
    parser.add_argument(
        "--static_crop_ms", type=float, default=Cfg.static_crop_ms,
        help="Static duration from onset; use --no_static_crop for the full raster.",
    )
    parser.add_argument("--no_static_crop", action="store_true")
    parser.add_argument("--dynamic_crop_ms", type=float)
    parser.add_argument(
        "--normalization", choices=("min_max",), default=Cfg.normalization,
        help="Optional within-condition, per-channel neural normalization.",
    )
    parser.add_argument("--model_name", default=Cfg.model_name)
    parser.add_argument(
        "--model_dataset_name", default=Cfg.model_dataset_name,
    )
    parser.add_argument(
        "--model_pooling", default=Cfg.model_pooling,
        help="Pooling label used in feature filenames; pass 'none' for no pooling.",
    )
    parser.add_argument("--model_features_dir")
    parser.add_argument(
        "--model_frame_index", type=int, default=Cfg.model_frame_index,
        help="Model video frame used for static RSA; -1 selects the final frame.",
    )
    parser.add_argument(
        "--signal_rdm_metric", default=Cfg.signal_rdm_metric,
        help="Dissimilarity metric for neural response RDMs.",
    )
    parser.add_argument(
        "--model_rdm_metric", default=Cfg.model_rdm_metric,
        help="Dissimilarity metric for network-feature RDMs.",
    )
    parser.add_argument(
        "--rsa_metric", choices=("pearson", "correlation", "spearman"),
        default=Cfg.rsa_metric,
        help="Correlation between vectorized neural and model RDMs.",
    )
    parser.add_argument(
        "--parallel", action=argparse.BooleanOptionalAction,
        default=Cfg.parallel,
        help="Distribute layers with useful_stuff's MPI master/worker queue.",
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction,
        default=Cfg.overwrite,
    )
    parser.add_argument(
        "--max_layers", type=int,
        help="Optional natural-order layer cap for smoke tests or partial runs.",
    )
    args = parser.parse_args()

    if args.source_fs <= 0 or args.new_fs <= 0:
        parser.error("--source_fs and --new_fs must be positive.")
    # end if invalid sampling frequency
    for argument_name in ("static_crop_ms", "dynamic_crop_ms"):
        value = getattr(args, argument_name)
        if value is not None and value <= 0:
            parser.error(f"--{argument_name} must be positive.")
        # end if invalid crop
    # end for argument_name
    if args.good_channels is not None:
        first_channel, last_channel = args.good_channels
        if first_channel < 1 or last_channel < first_channel:
            parser.error("--good_channels must be an increasing positive range.")
        # end if invalid channel range
    # end if args.good_channels is not None
    if args.max_layers is not None and args.max_layers < 1:
        parser.error("--max_layers must be positive.")
    # end if invalid max_layers

    args.good_channels = (
        None if args.good_channels is None else tuple(args.good_channels)
    )
    if args.no_static_crop:
        args.static_crop_ms = None
    # end if args.no_static_crop
    del args.no_static_crop
    if args.model_pooling.lower() == "none":
        args.model_pooling = None
    # end if model_pooling is none
    normalize_rsa_metric(args.rsa_metric)
    return Cfg(**vars(args))
# EOF


"""
session_label
Combine monkey and session names while accepting an already complete label.

INPUT:
    - monkey_name: str -> monkey identifier
    - experiment_name: str -> date/session suffix or complete session label

OUTPUT:
    - label: str -> complete session label
"""
def session_label(monkey_name: str, experiment_name: str) -> str:
    if experiment_name.startswith(f"{monkey_name}_"):
        return experiment_name
    # end if complete experiment label
    return f"{monkey_name}_{experiment_name}"
# EOF


"""
channel_selection_name
Build the channel-selection token registered in every result filename.

INPUT:
    - cfg: Cfg -> channel-range and reliability configuration
    - selected_channel_numbers: np.ndarray -> retained one-based channels

OUTPUT:
    - name: str -> range or reliability selection name
"""
def channel_selection_name(
        cfg: Cfg,
        selected_channel_numbers: np.ndarray,
        ) -> str:
    if cfg.reliable_channels_config is not None:
        # The digest distinguishes YAML revisions that retain the same count.
        channel_text = ",".join(
            str(channel) for channel in selected_channel_numbers
        )
        channel_digest = hashlib.sha256(channel_text.encode()).hexdigest()[:8]
        reliability_name = (
            f"reliable-{cfg.reliable_channels_key}"
            f"-n{len(selected_channel_numbers)}-{channel_digest}"
        )
        if cfg.good_channels is not None:
            reliability_name += (
                f"-within-{cfg.good_channels[0]}to{cfg.good_channels[1]}"
            )
        # end if reliability intersected with channel range
        return reliability_name
    # end if reliability filtering
    if cfg.good_channels is None:
        return "all_channels"
    # end if cfg.good_channels is None
    return f"channels_{cfg.good_channels[0]}to{cfg.good_channels[1]}"
# EOF


"""
resolve_cfg_paths
Resolve data, feature, output, and experiment labels from config.yaml defaults.

INPUT:
    - cfg: Cfg -> user configuration

OUTPUT:
    - cfg: Cfg -> configuration containing explicit resolved paths and label
"""
def resolve_cfg_paths(cfg: Cfg) -> Cfg:
    data_dir = Path(paths["data_path"]) / "data"
    static_label = session_label(cfg.monkey_name, cfg.static_experiment_name)
    dynamic_label = session_label(cfg.monkey_name, cfg.dynamic_experiment_name)
    cfg.experiment_name = (
        cfg.experiment_name or f"{dynamic_label}_vs_{static_label}"
    )
    cfg.static_path = str(Path(
        cfg.static_path or data_dir / f"{static_label}_natraster_img.mat"
    ).expanduser())
    cfg.dynamic_path = str(Path(
        cfg.dynamic_path or data_dir / f"{dynamic_label}_natraster_vid.mat"
    ).expanduser())
    cfg.model_features_dir = str(Path(
        cfg.model_features_dir or Path(paths["data_path"]) / "models"
    ).expanduser())
    cfg.output_dir = str(Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / "static_dynamic_neural_model_rsa"
    ).expanduser())

    if cfg.reliable_channels_config is not None:
        cfg.reliable_channels_config = str(Path(
            cfg.reliable_channels_config
        ).expanduser().resolve())
        cfg.reliable_channels_key = (
            cfg.reliable_channels_key or static_label
        )
    elif cfg.reliable_channels_key is not None:
        raise ValueError(
            "--reliable_channels_key requires --reliable_channels_config."
        )
    # end if reliability configuration

    for path_name in ("static_path", "dynamic_path", "model_features_dir"):
        target_path = Path(getattr(cfg, path_name))
        if not target_path.exists():
            raise FileNotFoundError(f"{path_name} does not exist: {target_path}")
        # end if missing path
    # end for path_name
    if (
            cfg.reliable_channels_config is not None
            and not Path(cfg.reliable_channels_config).is_file()
            ):
        raise FileNotFoundError(
            "reliable_channels_config does not exist: "
            f"{cfg.reliable_channels_config}"
        )
    # end if missing reliability config
    return cfg
# EOF


"""
prepare_neural_timeseries
Load one modality, select channels/stimuli, crop, normalize, and resample it.

INPUT:
    - neural_path: str | Path -> natraster file
    - stimulus_prefix: str -> img_ or vid_ modality prefix
    - crop_ms: float | None -> duration retained from onset
    - cfg: Cfg -> channel, sampling, and normalization parameters

OUTPUT:
    - neural_ts: TimeSeries -> channels x analysis time x stimuli
    - stimulus_names: list[str] -> retained neural stimulus names
    - channel_numbers: np.ndarray -> retained one-based MATLAB channels
"""
def prepare_neural_timeseries(
        neural_path: str | Path,
        stimulus_prefix: str,
        crop_ms: float | None,
        cfg: Cfg,
        ) -> tuple[TimeSeries, list[str], np.ndarray]:
    rasters, all_stimulus_names, channel_numbers = load_natraster(
        neural_path,
        good_channels=cfg.good_channels,
        reliable_channels_config=cfg.reliable_channels_config,
        reliable_channels_key=cfg.reliable_channels_key,
        return_channel_numbers=True,
    )
    excluded_prefixes = ()
    if stimulus_prefix == "img_":
        # The image-session file also contains earlier timed static controls.
        excluded_prefixes = ("img_2000ms_", "img_2250ms_")
    # end if static image condition
    rasters, stimulus_names = select_stimulus_rasters(
        rasters,
        all_stimulus_names,
        stimulus_prefix,
        excluded_prefixes=excluded_prefixes,
    )
    if crop_ms is not None:
        stop_sample = int(round(crop_ms * cfg.source_fs / 1000))
        if stop_sample > rasters.shape[1]:
            raise ValueError(
                f"{crop_ms:g} ms crop needs {stop_sample} samples, but "
                f"{Path(neural_path).name} contains {rasters.shape[1]}."
            )
        # end if crop exceeds raster
        rasters = rasters[:, :stop_sample, :]
    # end if crop_ms is not None
    if cfg.normalization == "min_max":
        rasters = min_max_normalization(rasters)
    # end if min_max normalization

    neural_ts = TimeSeries(rasters, fs=cfg.source_fs)
    neural_ts.resample(cfg.new_fs)
    return neural_ts, stimulus_names, channel_numbers
# EOF


"""
prepare_analysis_state
Precompute neural RDMs and feature-name mappings shared by all network layers.

INPUT:
    - cfg: Cfg -> resolved analysis configuration
    - first_feature_path: str | Path -> representative layer feature file

OUTPUT:
    - state: AnalysisState -> layer-independent arrays, names, and coordinates
"""
def prepare_analysis_state(
        cfg: Cfg,
        first_feature_path: str | Path,
        ) -> AnalysisState:
    static_ts, static_stimulus_names, static_channel_numbers = (
        prepare_neural_timeseries(
            cfg.static_path, "img_", cfg.static_crop_ms, cfg,
        )
    )
    dynamic_ts, dynamic_stimulus_names, dynamic_channel_numbers = (
        prepare_neural_timeseries(
            cfg.dynamic_path, "vid_", cfg.dynamic_crop_ms, cfg,
        )
    )
    if not np.array_equal(static_channel_numbers, dynamic_channel_numbers):
        raise ValueError(
            "Static and dynamic recordings retained different channel numbers."
        )
    # end if inconsistent channel selections
    static_feature_names = match_feature_stimulus_names(
        first_feature_path, static_stimulus_names,
    )
    dynamic_feature_names = match_feature_stimulus_names(
        first_feature_path, dynamic_stimulus_names,
    )

    print(
        f"Static neural data: {static_ts.shape()} (channels, time, stimuli)",
        flush=True,
    )
    print(
        f"Dynamic neural data: {dynamic_ts.shape()} (channels, time, stimuli)",
        flush=True,
    )
    print("Computing layer-independent neural RDM time series...", flush=True)
    static_neural_rdms = compute_neural_rdm_timeseries(
        static_ts, cfg.signal_rdm_metric,
    )
    dynamic_neural_rdms = compute_neural_rdm_timeseries(
        dynamic_ts, cfg.signal_rdm_metric,
    )
    return AnalysisState(
        static_neural_rdms=static_neural_rdms,
        dynamic_neural_rdms=dynamic_neural_rdms,
        static_feature_names=static_feature_names,
        dynamic_feature_names=dynamic_feature_names,
        static_stimulus_names=static_stimulus_names,
        dynamic_stimulus_names=dynamic_stimulus_names,
        selected_channel_numbers=static_channel_numbers,
        static_time_ms=np.arange(len(static_ts)) * 1000 / static_ts.get_fs(),
        dynamic_time_ms=np.arange(len(dynamic_ts)) * 1000 / dynamic_ts.get_fs(),
    )
# EOF


"""
run_layer
Compute and save both RSA analyses for one network layer.

INPUT:
    - paths: dict -> config paths required by the MPI queue interface
    - rank: int -> MPI rank or zero for serial execution
    - feature_path: str | Path -> current layer feature file
    - state: AnalysisState -> precomputed layer-independent analysis state
    - cfg: Cfg -> resolved analysis configuration

OUTPUT:
    - output_path: Path -> saved or pre-existing layer result
"""
def run_layer(
        paths: dict,
        rank: int,
        feature_path: str | Path,
        state: AnalysisState,
        cfg: Cfg,
        ) -> Path:
    del paths  # Required only by useful_stuff.master_workers_queue.
    with h5py.File(feature_path, "r") as feature_file:
        layer_name = str(feature_file.attrs["layer_name"])
    # end with h5py.File
    output_name = build_neural_model_rsa_filename(
        experiment_name=cfg.experiment_name,
        model_name=cfg.model_name,
        layer_name=layer_name,
        signal_rdm_metric=cfg.signal_rdm_metric,
        model_rdm_metric=cfg.model_rdm_metric,
        rsa_metric=normalize_rsa_metric(cfg.rsa_metric),
        new_fs=cfg.new_fs,
        model_dataset_name=cfg.model_dataset_name,
        model_pooling=cfg.model_pooling,
        channel_name=channel_selection_name(
            cfg, state.selected_channel_numbers,
        ),
        normalization=cfg.normalization,
        static_crop_ms=cfg.static_crop_ms,
        dynamic_crop_ms=cfg.dynamic_crop_ms,
        model_frame_index=cfg.model_frame_index,
    )
    output_path = Path(cfg.output_dir) / output_name
    if output_path.exists() and not cfg.overwrite:
        print_wise(f"result already exists at {output_path}", rank=rank)
        return output_path
    # end if output exists

    results = compute_layer_neural_model_rsa(
        feature_path=feature_path,
        static_feature_names=state.static_feature_names,
        dynamic_feature_names=state.dynamic_feature_names,
        static_neural_rdms=state.static_neural_rdms,
        dynamic_neural_rdms=state.dynamic_neural_rdms,
        analysis_fs=cfg.new_fs,
        model_rdm_metric=cfg.model_rdm_metric,
        rsa_metric=cfg.rsa_metric,
        model_frame_index=cfg.model_frame_index,
    )

    metadata = asdict(cfg)
    metadata.update({
        "environment": ENV,
        "feature_path": str(Path(feature_path).resolve()),
        "layer_name": results["layer_name"],
        "rsa_metric_computed": normalize_rsa_metric(cfg.rsa_metric),
        "static_rsa_axes": ["neural_time"],
        "dynamic_rsa_axes": ["neural_time", "model_time"],
        "selected_channel_numbers": state.selected_channel_numbers.tolist(),
    })
    save_layer_neural_model_rsa(
        output_path=output_path,
        results=results,
        metadata=metadata,
        static_time_ms=state.static_time_ms,
        dynamic_neural_time_ms=state.dynamic_time_ms,
        selected_channel_numbers=state.selected_channel_numbers,
        static_stimulus_names=state.static_stimulus_names,
        dynamic_stimulus_names=state.dynamic_stimulus_names,
    )
    print_wise(f"saved {output_path}", rank=rank)
    return output_path
# EOF


"""
main
Run all layer analyses serially or through useful_stuff's MPI layer queue.
"""
def main() -> None:
    cfg = resolve_cfg_paths(parse_args())
    feature_paths = list_video_feature_files(
        cfg.model_features_dir,
        cfg.model_name,
        cfg.model_dataset_name,
        cfg.model_pooling,
    )
    if cfg.max_layers is not None:
        feature_paths = feature_paths[:cfg.max_layers]
    # end if max_layers is not None
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    if cfg.parallel:
        # Every worker prepares the shared neural RDMs once, then receives layers.
        from useful_stuff.parallel.parallel_funcs import (
            master_workers_queue,
            parallel_setup,
        )

        _, rank, size = parallel_setup()
        if size < 2:
            raise RuntimeError(
                "--parallel needs at least two MPI ranks; run with, for example, "
                "mpiexec -np 5 python run_static_dynamic_neural_model_rsa.py "
                "--parallel ..."
            )
        # end if size < 2
        state = None
        if 0 < rank <= len(feature_paths):
            state = prepare_analysis_state(cfg, feature_paths[0])
        # end if active worker
        master_workers_queue(feature_paths, paths, run_layer, state, cfg)
    else:
        state = prepare_analysis_state(cfg, feature_paths[0])
        print(f"Computing {len(feature_paths)} model layers serially...", flush=True)
        for feature_path in feature_paths:
            run_layer(paths, 0, feature_path, state, cfg)
        # end for feature_path
        print(f"Results saved to {cfg.output_dir}", flush=True)
    # end if cfg.parallel
# EOF


if __name__ == "__main__":
    main()
# EOF
