from pathlib import Path

import cv2
import h5py
import numpy as np


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


"""
list_movie_paths
Lists movie stimuli in stable filename order.

INPUT:
    - stimuli_dir: str | Path -> directory containing movie stimuli
    - video_patterns: tuple[str, ...] -> glob patterns in preferred-format order;
        later matches with an already selected filename stem are ignored
    - max_videos: int | None -> optional cap for tests or partial extraction

OUTPUT:
    - movie_paths: list[Path] -> sorted unique movie paths
"""
def list_movie_paths(
        stimuli_dir,
        video_patterns=("vid_*.mp4", "vid_*.mov", "vid_*.m4v"),
        max_videos=None,
        ):
    stimuli_dir = Path(stimuli_dir).expanduser()
    if not stimuli_dir.exists():
        raise FileNotFoundError(f"Stimulus directory does not exist: {stimuli_dir}")
    # end if not stimuli_dir.exists()

    # The stimulus folder can contain both .mp4 and .mov versions of each movie.
    # Keep the first pattern's version so every stimulus identity appears once.
    movie_by_stem = {}
    for pattern in video_patterns:
        for path in sorted(stimuli_dir.glob(pattern)):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            # end if unsupported path
            movie_by_stem.setdefault(path.stem, path)
        # end for path
    # end for pattern
    movie_paths = sorted(movie_by_stem.values(), key=lambda path: path.name)
    if max_videos is not None:
        movie_paths = movie_paths[:max_videos]
    # end if max_videos
    if not movie_paths:
        raise FileNotFoundError(
            f"No movies matching {video_patterns!r} found in {stimuli_dir}"
        )
    # end if not movie_paths
    return movie_paths
# EOF


"""
prepare_flow_frame
Converts a decoded BGR frame to the fixed grayscale grid used for optical flow.

INPUT:
    - frame_bgr: np.ndarray -> decoded H x W x 3 BGR frame
    - flow_width: int -> output grid width
    - flow_height: int -> output grid height

OUTPUT:
    - gray_frame: np.ndarray -> flow_height x flow_width uint8 grayscale frame
"""
def prepare_flow_frame(frame_bgr, flow_width, flow_height):
    gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    interpolation = (
        cv2.INTER_AREA
        if gray_frame.shape[1] > flow_width or gray_frame.shape[0] > flow_height
        else cv2.INTER_LINEAR
    )
    gray_frame = cv2.resize(
        gray_frame,
        (flow_width, flow_height),
        interpolation=interpolation,
    )
    return gray_frame
# EOF


"""
compute_farneback_flow
Computes dense displacement from the previous retained frame to the current one.

INPUT:
    - previous_gray: np.ndarray -> previous fixed-grid grayscale frame
    - current_gray: np.ndarray -> current fixed-grid grayscale frame
    - pyr_scale: float -> image-pyramid scale between levels
    - levels: int -> number of pyramid levels
    - winsize: int -> averaging-window size
    - iterations: int -> iterations at every pyramid level
    - poly_n: int -> pixel-neighborhood size for polynomial expansion
    - poly_sigma: float -> Gaussian standard deviation for polynomial expansion
    - flags: int -> OpenCV Farneback flags

OUTPUT:
    - flow: np.ndarray -> flow_height x flow_width x 2 float32 field
        where component 0 is horizontal displacement u and component 1 is
        vertical displacement v, in pixels of the resized flow grid
"""
def compute_farneback_flow(
        previous_gray,
        current_gray,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
        ):
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    )
    return flow.astype(np.float32, copy=False)
# EOF


"""
flatten_flow_field
Flattens a dense flow field into a component-major feature vector.

INPUT:
    - flow: np.ndarray -> height x width x 2 dense flow

OUTPUT:
    - flow_vector: np.ndarray -> vector containing all u values followed by all v values
"""
def flatten_flow_field(flow):
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected H x W x 2 flow, got {flow.shape}")
    # end if flow shape
    return flow.transpose(2, 0, 1).reshape(-1).astype(np.float32, copy=False)
# EOF


"""
interpolate_internal_zero_flow_runs
Linearly interpolates near-zero flow runs bounded by healthy flow on both sides.

The first undefined flow vector and unbounded leading/trailing zero runs are
left unchanged because they do not have two valid interpolation endpoints.

INPUT:
    - flow_vectors: np.ndarray -> time x flattened-flow-feature array
    - zero_flow_threshold: float -> maximum per-frame RMS displacement treated
        as zero flow

OUTPUT:
    - interpolated_vectors: np.ndarray -> repaired float32 flow vectors
    - interpolation_info: dict -> detected and repaired run/frame counts
"""
def interpolate_internal_zero_flow_runs(
        flow_vectors,
        zero_flow_threshold=0.05,
        ):
    flow_vectors = np.asarray(flow_vectors, dtype=np.float32)
    if flow_vectors.ndim != 2:
        raise ValueError(
            f"flow_vectors must have shape time x features; got {flow_vectors.shape}"
        )
    # end if flow_vectors.ndim
    if flow_vectors.shape[0] < 2:
        raise ValueError("Need at least two flow timepoints.")
    # end if flow timepoints
    if zero_flow_threshold < 0:
        raise ValueError("zero_flow_threshold must be non-negative.")
    # end if zero_flow_threshold

    # RMS displacement is independent of feature count and treats small numerical
    # Farneback residuals from duplicated frames as zero when below the threshold.
    flow_rms = np.sqrt(
        np.mean(flow_vectors.astype(np.float64) ** 2, axis=1)
    )
    zero_flow_mask = flow_rms <= zero_flow_threshold
    interpolated_vectors = flow_vectors.copy()
    interpolated_run_count = 0
    interpolated_frame_count = 0
    unbounded_zero_frame_count = 0

    frame_index = 0
    while frame_index < len(zero_flow_mask):
        if not zero_flow_mask[frame_index]:
            frame_index += 1
            continue
        # end if healthy frame

        # Find the half-open [run_start, run_stop) interval of this zero-flow run.
        run_start = frame_index
        while (
            frame_index < len(zero_flow_mask)
            and zero_flow_mask[frame_index]
        ):
            frame_index += 1
        # end while zero-flow run
        run_stop = frame_index

        left_index = run_start - 1
        right_index = run_stop
        bounded_by_healthy_flow = (
            left_index >= 0
            and right_index < len(zero_flow_mask)
            and not zero_flow_mask[left_index]
            and not zero_flow_mask[right_index]
        )
        if not bounded_by_healthy_flow:
            unbounded_zero_frame_count += run_stop - run_start
            continue
        # end if not bounded_by_healthy_flow

        # Interpolate the complete flattened vector field at each missing time.
        # Alpha excludes 0 and 1 because the healthy endpoints are preserved.
        run_indices = np.arange(run_start, run_stop)
        alpha = (
            (run_indices - left_index) / (right_index - left_index)
        ).astype(np.float32)[:, None]
        interpolated_vectors[run_start:run_stop] = (
            (1 - alpha) * flow_vectors[left_index]
            + alpha * flow_vectors[right_index]
        )
        interpolated_run_count += 1
        interpolated_frame_count += run_stop - run_start
    # end while frame_index

    interpolation_info = {
        "zero_flow_threshold": float(zero_flow_threshold),
        "detected_zero_frames": int(zero_flow_mask.sum()),
        "interpolated_zero_runs": interpolated_run_count,
        "interpolated_zero_frames": interpolated_frame_count,
        "unbounded_zero_frames": unbounded_zero_frame_count,
    }
    return interpolated_vectors, interpolation_info
# EOF


"""
repair_hdf5_zero_flow_runs
Repairs one saved movie dataset in place and records interpolation metadata.

INPUT:
    - feature_file: h5py.File -> open optical-flow feature file
    - dataset_name: str -> movie dataset to repair
    - zero_flow_threshold: float -> maximum RMS displacement treated as zero

OUTPUT:
    - interpolation_info: dict -> detected and repaired run/frame counts
"""
def repair_hdf5_zero_flow_runs(
        feature_file,
        dataset_name,
        zero_flow_threshold=0.05,
        ):
    dataset = feature_file[dataset_name]
    repaired_vectors, interpolation_info = interpolate_internal_zero_flow_runs(
        dataset[:],
        zero_flow_threshold=zero_flow_threshold,
    )

    # Rewrite only when an internal run changed, avoiding unnecessary HDF5 writes.
    if interpolation_info["interpolated_zero_frames"] > 0:
        dataset[:] = repaired_vectors
    # end if interpolated frames

    dataset.attrs["zero_flow_policy"] = "linear_internal_interpolation"
    for attr_name, value in interpolation_info.items():
        dataset.attrs[attr_name] = value
    # end for attr_name
    feature_file.flush()
    return interpolation_info
# EOF


"""
append_feature_rows
Appends a batch of optical-flow vectors to an extendable HDF5 dataset.

INPUT:
    - feature_file: h5py.File -> open output file
    - dataset_key: str -> active dataset name
    - feature_rows: list[np.ndarray] -> flow vectors to append
    - feature_size: int -> expected flattened feature count
    - compression: str | None -> optional HDF5 compression filter

OUTPUT:
    - None
"""
def append_feature_rows(
        feature_file,
        dataset_key,
        feature_rows,
        feature_size,
        compression="lzf",
        ):
    if not feature_rows:
        return None
    # end if not feature_rows

    feature_rows = np.stack(feature_rows, axis=0)
    if feature_rows.shape[1] != feature_size:
        raise ValueError(
            f"Expected {feature_size} flow features, got {feature_rows.shape[1]}"
        )
    # end if feature_rows.shape

    if dataset_key not in feature_file:
        feature_file.create_dataset(
            dataset_key,
            shape=(0, feature_size),
            maxshape=(None, feature_size),
            dtype=np.float32,
            chunks=True,
            compression=compression,
        )
    # end if dataset_key

    dataset = feature_file[dataset_key]
    start = dataset.shape[0]
    dataset.resize(start + feature_rows.shape[0], axis=0)
    dataset[start:] = feature_rows
    return None
# EOF


"""
extract_movie_optical_flow
Streams one movie through dense Farneback optical-flow extraction.

The first retained frame receives a zero vector because it has no preceding
retained frame. Every later vector describes motion from the previous retained
frame to the current retained frame.

INPUT:
    - movie_path: str | Path -> movie file
    - feature_file: h5py.File -> open output file
    - flow_width: int -> resized flow-grid width
    - flow_height: int -> resized flow-grid height
    - frame_stride: int -> retain one frame every frame_stride decoded frames
    - max_frames: int | None -> optional cap on retained frames
    - write_batch_size: int -> rows buffered before an HDF5 append
    - compression: str | None -> HDF5 compression filter
    - farneback_kwargs: dict -> OpenCV Farneback parameters

OUTPUT:
    - n_frames: int -> number of retained frames and saved flow vectors
"""
def extract_movie_optical_flow(
        movie_path,
        feature_file,
        flow_width,
        flow_height,
        frame_stride=1,
        max_frames=None,
        write_batch_size=64,
        compression="lzf",
        farneback_kwargs=None,
        ):
    movie_path = Path(movie_path)
    temporary_key = f"__incomplete__{movie_path.name}"
    if temporary_key in feature_file:
        del feature_file[temporary_key]
    # end if temporary_key

    cap = cv2.VideoCapture(str(movie_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open movie: {movie_path}")
    # end if not cap.isOpened()

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
        print(f"{movie_path.name}: invalid FPS metadata; using 30 Hz.")
    # end if source_fps

    farneback_kwargs = {} if farneback_kwargs is None else farneback_kwargs
    feature_size = 2 * flow_height * flow_width
    feature_rows = []
    previous_gray = None
    decoded_frame_index = 0
    retained_frames = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            # end if not ok

            retain_frame = decoded_frame_index % frame_stride == 0
            decoded_frame_index += 1
            if not retain_frame:
                continue
            # end if not retain_frame

            current_gray = prepare_flow_frame(
                frame_bgr, flow_width, flow_height
            )
            if previous_gray is None:
                # Preserve the movie/frame time grid with an explicit undefined-motion zero.
                flow_vector = np.zeros(feature_size, dtype=np.float32)
            else:
                flow = compute_farneback_flow(
                    previous_gray,
                    current_gray,
                    **farneback_kwargs,
                )
                flow_vector = flatten_flow_field(flow)
            # end if previous_gray is None

            feature_rows.append(flow_vector)
            previous_gray = current_gray
            retained_frames += 1

            reached_frame_cap = (
                max_frames is not None and retained_frames >= max_frames
            )
            if len(feature_rows) >= write_batch_size or reached_frame_cap:
                append_feature_rows(
                    feature_file,
                    temporary_key,
                    feature_rows,
                    feature_size,
                    compression=compression,
                )
                feature_rows = []
            # end if write batch
            if reached_frame_cap:
                break
            # end if reached_frame_cap
        # end while

        append_feature_rows(
            feature_file,
            temporary_key,
            feature_rows,
            feature_size,
            compression=compression,
        )
    finally:
        cap.release()
    # end try

    if retained_frames < 2:
        if temporary_key in feature_file:
            del feature_file[temporary_key]
        # end if temporary_key
        raise RuntimeError(
            f"Need at least two retained frames in {movie_path}; got {retained_frames}"
        )
    # end if retained_frames

    dataset = feature_file[temporary_key]
    dataset.attrs["source_path"] = str(movie_path)
    dataset.attrs["source_fps"] = source_fps
    dataset.attrs["effective_fps"] = source_fps / frame_stride
    dataset.attrs["frame_stride"] = frame_stride
    dataset.attrs["n_frames"] = retained_frames
    dataset.attrs["first_vector"] = "zeros_no_previous_frame"
    feature_file.move(temporary_key, movie_path.name)
    feature_file.flush()
    return retained_frames
# EOF


"""
extract_optical_flow_dataset
Extracts dense optical-flow vectors for all requested movies into one HDF5 file.

INPUT:
    - movie_paths: list[Path] -> movies processed in stable order
    - output_path: str | Path -> HDF5 output path
    - flow_width: int -> flow-grid width
    - flow_height: int -> flow-grid height
    - frame_stride: int -> decoded-frame stride
    - max_frames: int | None -> optional retained-frame cap per movie
    - write_batch_size: int -> buffered rows per HDF5 append
    - compression: str | None -> HDF5 compression filter
    - overwrite: bool -> recompute movie datasets that already exist
    - farneback_kwargs: dict -> OpenCV Farneback parameters
    - interpolate_zero_flow: bool -> linearly repair bounded zero-flow runs
    - zero_flow_threshold: float -> maximum per-frame RMS displacement treated
        as zero flow during interpolation

OUTPUT:
    - output_path: Path -> completed HDF5 path
"""
def extract_optical_flow_dataset(
        movie_paths,
        output_path,
        flow_width=64,
        flow_height=64,
        frame_stride=1,
        max_frames=None,
        write_batch_size=64,
        compression="lzf",
        overwrite=False,
        farneback_kwargs=None,
        interpolate_zero_flow=True,
        zero_flow_threshold=0.05,
        ):
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    farneback_kwargs = {} if farneback_kwargs is None else farneback_kwargs

    root_metadata = {
        "algorithm": "farneback",
        "flow_width": flow_width,
        "flow_height": flow_height,
        "feature_size": 2 * flow_height * flow_width,
        "component_order": "u_then_v",
        "vector_units": "pixels_on_resized_flow_grid_per_retained_frame",
        "frame_stride": frame_stride,
    }
    root_metadata.update({
        f"farneback_{name}": value
        for name, value in farneback_kwargs.items()
    })

    with h5py.File(output_path, "a") as feature_file:
        # Prevent accidental mixing of incompatible feature geometries or parameters.
        for attr_name, expected_value in root_metadata.items():
            if (
                attr_name in feature_file.attrs
                and feature_file.attrs[attr_name] != expected_value
            ):
                raise ValueError(
                    f"{output_path.name} has {attr_name}="
                    f"{feature_file.attrs[attr_name]!r}; expected {expected_value!r}."
                )
            # end if stored metadata
            feature_file.attrs[attr_name] = expected_value
        # end for attr_name

        for movie_index, movie_path in enumerate(movie_paths, start=1):
            if movie_path.name in feature_file and not overwrite:
                dataset = feature_file[movie_path.name]
                stored_policy = str(
                    dataset.attrs.get("zero_flow_policy", "none")
                )
                if interpolate_zero_flow and stored_policy == "none":
                    interpolation_info = repair_hdf5_zero_flow_runs(
                        feature_file,
                        movie_path.name,
                        zero_flow_threshold=zero_flow_threshold,
                    )
                    print(
                        f"[{movie_index}/{len(movie_paths)}] Repaired existing "
                        f"{movie_path.name}: "
                        f"{interpolation_info['interpolated_zero_frames']} frames "
                        f"in {interpolation_info['interpolated_zero_runs']} runs"
                    )
                    continue
                # end if raw existing dataset
                if (
                    interpolate_zero_flow
                    and stored_policy != "linear_internal_interpolation"
                ):
                    raise ValueError(
                        f"{movie_path.name} has unsupported zero_flow_policy="
                        f"{stored_policy!r}; use --overwrite to recompute it."
                    )
                # end if stored_policy
                if (
                    interpolate_zero_flow
                    and stored_policy == "linear_internal_interpolation"
                ):
                    stored_threshold = float(
                        dataset.attrs["zero_flow_threshold"]
                    )
                    if not np.isclose(
                        stored_threshold,
                        zero_flow_threshold,
                        rtol=0,
                        atol=0,
                    ):
                        raise ValueError(
                            f"{movie_path.name} was interpolated with threshold "
                            f"{stored_threshold:g}; requested "
                            f"{zero_flow_threshold:g}. Use --overwrite to "
                            "recompute it."
                        )
                    # end if stored_threshold
                # end if interpolated dataset
                if not interpolate_zero_flow and stored_policy != "none":
                    raise ValueError(
                        f"{movie_path.name} is already interpolated; use "
                        "--overwrite with --no-interpolate-zero-flow to recreate "
                        "the raw flow vectors."
                    )
                # end if interpolation disabled
                print(
                    f"[{movie_index}/{len(movie_paths)}] "
                    f"Skipping existing {movie_path.name}"
                )
                continue
            # end if movie exists
            if movie_path.name in feature_file:
                del feature_file[movie_path.name]
            # end if overwrite movie

            print(
                f"[{movie_index}/{len(movie_paths)}] "
                f"Extracting {movie_path.name}"
            )
            n_frames = extract_movie_optical_flow(
                movie_path,
                feature_file,
                flow_width=flow_width,
                flow_height=flow_height,
                frame_stride=frame_stride,
                max_frames=max_frames,
                write_batch_size=write_batch_size,
                compression=compression,
                farneback_kwargs=farneback_kwargs,
            )
            if interpolate_zero_flow:
                interpolation_info = repair_hdf5_zero_flow_runs(
                    feature_file,
                    movie_path.name,
                    zero_flow_threshold=zero_flow_threshold,
                )
            else:
                dataset = feature_file[movie_path.name]
                dataset.attrs["zero_flow_policy"] = "none"
                dataset.attrs["zero_flow_threshold"] = zero_flow_threshold
                dataset.attrs["detected_zero_frames"] = -1
                dataset.attrs["interpolated_zero_runs"] = 0
                dataset.attrs["interpolated_zero_frames"] = 0
                dataset.attrs["unbounded_zero_frames"] = -1
                interpolation_info = {
                    "interpolated_zero_runs": 0,
                    "interpolated_zero_frames": 0,
                }
                feature_file.flush()
            # end if interpolate_zero_flow
            print(
                f"Saved {n_frames} flow vectors for {movie_path.name}; "
                f"interpolated "
                f"{interpolation_info['interpolated_zero_frames']} frames in "
                f"{interpolation_info['interpolated_zero_runs']} runs"
            )
        # end for movie_path
    # end with h5py.File
    return output_path
# EOF


"""
load_aligned_optical_flow_features
Loads flow features in a requested movie order for time-resolved analysis.

INPUT:
    - feature_path: str | Path -> optical-flow HDF5 file
    - movie_names: list[str] -> exact HDF5 dataset names in stimulus order
    - drop_first_frame: bool -> remove the all-zero undefined first flow vector

OUTPUT:
    - features: np.ndarray -> features x time x stimuli
    - effective_fps: float -> retained-frame sampling frequency
    - metadata: dict -> flow geometry and model-time offset information
"""
def load_aligned_optical_flow_features(
        feature_path,
        movie_names,
        drop_first_frame=True,
        ):
    feature_path = Path(feature_path)
    with h5py.File(feature_path, "r") as feature_file:
        missing_names = [name for name in movie_names if name not in feature_file]
        if missing_names:
            raise KeyError(
                f"{feature_path.name} is missing {len(missing_names)} movies: "
                f"{missing_names[:5]}"
            )
        # end if missing_names

        frame_counts = {feature_file[name].shape[0] for name in movie_names}
        feature_sizes = {feature_file[name].shape[1] for name in movie_names}
        effective_fps_values = {
            float(feature_file[name].attrs["effective_fps"])
            for name in movie_names
        }
        if len(frame_counts) != 1:
            raise ValueError(f"Movies have different flow frame counts: {frame_counts}")
        # end if frame_counts
        if len(feature_sizes) != 1:
            raise ValueError(f"Movies have different flow feature sizes: {feature_sizes}")
        # end if feature_sizes
        if len(effective_fps_values) != 1:
            raise ValueError(
                f"Movies have different effective frame rates: {effective_fps_values}"
            )
        # end if effective_fps_values

        # Stack as stimuli x time x features, then expose the project convention
        # features x time x stimuli used by TimeSeries and dRSA.
        features = np.stack(
            [feature_file[name][:] for name in movie_names],
            axis=0,
        ).transpose(2, 1, 0)

        effective_fps = effective_fps_values.pop()
        first_valid_flow_time_s = 0.0
        if drop_first_frame:
            features = features[:, 1:, :]
            first_valid_flow_time_s = 1.0 / effective_fps
        # end if drop_first_frame

        metadata = {
            "algorithm": str(feature_file.attrs["algorithm"]),
            "flow_width": int(feature_file.attrs["flow_width"]),
            "flow_height": int(feature_file.attrs["flow_height"]),
            "component_order": str(feature_file.attrs["component_order"]),
            "vector_units": str(feature_file.attrs["vector_units"]),
            "drop_first_frame": drop_first_frame,
            "first_valid_flow_time_s": first_valid_flow_time_s,
        }

        zero_flow_policies = {
            str(feature_file[name].attrs.get("zero_flow_policy", "none"))
            for name in movie_names
        }
        zero_flow_thresholds = {
            (
                float(feature_file[name].attrs["zero_flow_threshold"])
                if "zero_flow_threshold" in feature_file[name].attrs
                else None
            )
            for name in movie_names
        }
        if len(zero_flow_policies) != 1:
            raise ValueError(
                f"Movies use different zero-flow policies: {zero_flow_policies}"
            )
        # end if zero_flow_policies
        if len(zero_flow_thresholds) != 1:
            raise ValueError(
                f"Movies use different zero-flow thresholds: {zero_flow_thresholds}"
            )
        # end if zero_flow_thresholds

        interpolated_frames_by_movie = {
            name: int(
                feature_file[name].attrs.get("interpolated_zero_frames", 0)
            )
            for name in movie_names
        }
        interpolated_runs_by_movie = {
            name: int(
                feature_file[name].attrs.get("interpolated_zero_runs", 0)
            )
            for name in movie_names
        }
        metadata.update({
            "zero_flow_policy": zero_flow_policies.pop(),
            "zero_flow_threshold": zero_flow_thresholds.pop(),
            "interpolated_zero_frames_by_movie": interpolated_frames_by_movie,
            "interpolated_zero_runs_by_movie": interpolated_runs_by_movie,
            "interpolated_zero_frames_total": sum(
                interpolated_frames_by_movie.values()
            ),
            "interpolated_zero_runs_total": sum(
                interpolated_runs_by_movie.values()
            ),
        })
    # end with h5py.File
    return features, effective_fps, metadata
# EOF
