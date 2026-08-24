from pathlib import Path

import cv2
import h5py
import numpy as np


"""
pixel_sample_indices
Returns the flattened spatial pixel indices retained at a fixed step.

INPUT:
    - image_width: int -> width of the fixed RGB frame grid
    - image_height: int -> height of the fixed RGB frame grid
    - pixel_step: int -> retain every pixel_step-th spatial pixel

OUTPUT:
    - sample_indices: np.ndarray -> retained row-major spatial indices
"""
def pixel_sample_indices(image_width, image_height, pixel_step=50):
    if image_width < 1 or image_height < 1:
        raise ValueError("image_width and image_height must be positive.")
    # end if image geometry
    if pixel_step < 1:
        raise ValueError("pixel_step must be positive.")
    # end if pixel_step

    return np.arange(0, image_width * image_height, pixel_step)
# EOF


"""
sample_rgb_pixels
Resizes a decoded frame and retains every pixel_step-th spatial RGB pixel.

RGB channels are kept together, so the returned vector has pixel-major order:
[R0, G0, B0, R1, G1, B1, ...].

INPUT:
    - frame_bgr: np.ndarray -> decoded H x W x 3 OpenCV BGR frame
    - image_width: int -> output frame-grid width
    - image_height: int -> output frame-grid height
    - pixel_step: int -> retain every pixel_step-th spatial pixel

OUTPUT:
    - pixel_vector: np.ndarray -> flattened sampled RGB values as uint8
"""
def sample_rgb_pixels(
        frame_bgr,
        image_width,
        image_height,
        pixel_step=50,
        ):
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected H x W x 3 frame, got {frame_bgr.shape}.")
    # end if frame shape

    interpolation = (
        cv2.INTER_AREA
        if frame_bgr.shape[1] > image_width
        or frame_bgr.shape[0] > image_height
        else cv2.INTER_LINEAR
    )
    resized_bgr = cv2.resize(
        frame_bgr,
        (image_width, image_height),
        interpolation=interpolation,
    )
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    # Subsample spatial pixels, not scalar channel entries, so each retained
    # location contributes all three color channels.
    sampled_rgb = resized_rgb.reshape(-1, 3)[::pixel_step]
    return sampled_rgb.reshape(-1).astype(np.uint8, copy=False)
# EOF


"""
append_pixel_rows
Appends buffered RGB pixel vectors to an extendable HDF5 dataset.

INPUT:
    - feature_file: h5py.File -> open output feature file
    - dataset_key: str -> active temporary dataset name
    - feature_rows: list[np.ndarray] -> sampled RGB vectors to append
    - feature_size: int -> expected vector length
    - compression: str | None -> optional HDF5 compression filter

OUTPUT:
    - None
"""
def append_pixel_rows(
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
            f"Expected {feature_size} RGB features, got {feature_rows.shape[1]}."
        )
    # end if feature size

    if dataset_key not in feature_file:
        feature_file.create_dataset(
            dataset_key,
            shape=(0, feature_size),
            maxshape=(None, feature_size),
            dtype=np.uint8,
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
extract_movie_pixel_values
Streams one movie into subsampled RGB pixel vectors.

INPUT:
    - movie_path: str | Path -> input movie file
    - feature_file: h5py.File -> open output feature file
    - image_width: int -> fixed RGB frame width
    - image_height: int -> fixed RGB frame height
    - pixel_step: int -> retain every pixel_step-th spatial pixel
    - frame_stride: int -> retain one in every frame_stride decoded frames
    - max_frames: int | None -> optional retained-frame cap
    - write_batch_size: int -> rows buffered before each HDF5 append
    - compression: str | None -> HDF5 compression filter

OUTPUT:
    - retained_frames: int -> number of vectors written for the movie
"""
def extract_movie_pixel_values(
        movie_path,
        feature_file,
        image_width,
        image_height,
        pixel_step=50,
        frame_stride=1,
        max_frames=None,
        write_batch_size=64,
        compression="lzf",
        ):
    movie_path = Path(movie_path)
    temporary_key = f"__incomplete__{movie_path.name}"
    if temporary_key in feature_file:
        del feature_file[temporary_key]
    # end if temporary_key

    cap = cv2.VideoCapture(str(movie_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open movie: {movie_path}")
    # end if not cap.isOpened

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
        print(f"{movie_path.name}: invalid FPS metadata; using 30 Hz.")
    # end if source_fps

    n_sampled_pixels = len(
        pixel_sample_indices(image_width, image_height, pixel_step)
    )
    feature_size = 3 * n_sampled_pixels
    feature_rows = []
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

            pixel_vector = sample_rgb_pixels(
                frame_bgr,
                image_width=image_width,
                image_height=image_height,
                pixel_step=pixel_step,
            )
            feature_rows.append(pixel_vector)
            retained_frames += 1

            reached_frame_cap = (
                max_frames is not None and retained_frames >= max_frames
            )
            if len(feature_rows) >= write_batch_size or reached_frame_cap:
                append_pixel_rows(
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

        append_pixel_rows(
            feature_file,
            temporary_key,
            feature_rows,
            feature_size,
            compression=compression,
        )
    finally:
        cap.release()
    # end try

    if retained_frames < 1:
        if temporary_key in feature_file:
            del feature_file[temporary_key]
        # end if temporary_key
        raise RuntimeError(f"No retained frames decoded from {movie_path}.")
    # end if retained_frames

    dataset = feature_file[temporary_key]
    dataset.attrs["source_path"] = str(movie_path)
    dataset.attrs["source_fps"] = source_fps
    dataset.attrs["effective_fps"] = source_fps / frame_stride
    dataset.attrs["frame_stride"] = frame_stride
    dataset.attrs["n_frames"] = retained_frames
    feature_file.move(temporary_key, movie_path.name)
    feature_file.flush()
    return retained_frames
# EOF


"""
extract_pixel_value_dataset
Extracts subsampled RGB vectors for a stable ordered set of movie stimuli.

INPUT:
    - movie_paths: list[Path] -> movies to process
    - output_path: str | Path -> output HDF5 path
    - image_width: int -> fixed RGB frame width
    - image_height: int -> fixed RGB frame height
    - pixel_step: int -> retain every pixel_step-th spatial pixel
    - frame_stride: int -> decoded-frame stride
    - max_frames: int | None -> optional retained-frame cap per movie
    - write_batch_size: int -> buffered rows per HDF5 append
    - compression: str | None -> HDF5 compression filter
    - overwrite: bool -> recompute datasets that already exist

OUTPUT:
    - output_path: Path -> completed HDF5 path
"""
def extract_pixel_value_dataset(
        movie_paths,
        output_path,
        image_width=500,
        image_height=500,
        pixel_step=50,
        frame_stride=1,
        max_frames=None,
        write_batch_size=64,
        compression="lzf",
        overwrite=False,
        ):
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_sampled_pixels = len(
        pixel_sample_indices(image_width, image_height, pixel_step)
    )
    root_metadata = {
        "feature_type": "subsampled_rgb_pixels",
        "image_width": image_width,
        "image_height": image_height,
        "pixel_step": pixel_step,
        "n_sampled_pixels": n_sampled_pixels,
        "feature_size": 3 * n_sampled_pixels,
        "channel_order": "RGB",
        "flatten_order": "pixel_major_RGB",
        "frame_stride": frame_stride,
        "value_dtype": "uint8",
        "value_range": "0_to_255",
    }

    with h5py.File(output_path, "a") as feature_file:
        # Refuse to mix incompatible geometries or sampling settings.
        for attr_name, expected_value in root_metadata.items():
            if (
                attr_name in feature_file.attrs
                and feature_file.attrs[attr_name] != expected_value
            ):
                raise ValueError(
                    f"{output_path.name} has {attr_name}="
                    f"{feature_file.attrs[attr_name]!r}; "
                    f"expected {expected_value!r}."
                )
            # end if stored metadata
            feature_file.attrs[attr_name] = expected_value
        # end for attr_name

        for movie_index, movie_path in enumerate(movie_paths, start=1):
            movie_path = Path(movie_path)
            if movie_path.name in feature_file and not overwrite:
                print(
                    f"[{movie_index}/{len(movie_paths)}] "
                    f"Skipping existing {movie_path.name}"
                )
                continue
            # end if existing movie
            if movie_path.name in feature_file:
                del feature_file[movie_path.name]
            # end if overwrite movie

            print(
                f"[{movie_index}/{len(movie_paths)}] "
                f"Extracting {movie_path.name}"
            )
            n_frames = extract_movie_pixel_values(
                movie_path,
                feature_file,
                image_width=image_width,
                image_height=image_height,
                pixel_step=pixel_step,
                frame_stride=frame_stride,
                max_frames=max_frames,
                write_batch_size=write_batch_size,
                compression=compression,
            )
            print(
                f"Saved {n_frames} RGB pixel vectors for {movie_path.name}"
            )
        # end for movie_path
    # end with h5py.File
    return output_path
# EOF


"""
load_aligned_pixel_value_features
Loads subsampled RGB features in a requested movie order.

INPUT:
    - feature_path: str | Path -> pixel-value HDF5 file
    - movie_names: list[str] -> exact HDF5 dataset names in stimulus order

OUTPUT:
    - features: np.ndarray -> sampled RGB features x time x stimuli
    - effective_fps: float -> retained-frame sampling frequency
    - metadata: dict -> spatial geometry and vector-order information
"""
def load_aligned_pixel_value_features(feature_path, movie_names):
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
            raise ValueError(
                f"Movies have different RGB frame counts: {frame_counts}."
            )
        # end if frame_counts
        if len(feature_sizes) != 1:
            raise ValueError(
                f"Movies have different RGB feature sizes: {feature_sizes}."
            )
        # end if feature_sizes
        if len(effective_fps_values) != 1:
            raise ValueError(
                "Movies have different effective frame rates: "
                f"{effective_fps_values}."
            )
        # end if effective_fps_values

        # Stack stimuli x time x features, then expose the project convention
        # features x time x stimuli used by TimeSeries and dRSA.
        features = np.stack(
            [feature_file[name][:] for name in movie_names],
            axis=0,
        ).transpose(2, 1, 0)
        effective_fps = effective_fps_values.pop()
        metadata = {
            "feature_type": str(feature_file.attrs["feature_type"]),
            "image_width": int(feature_file.attrs["image_width"]),
            "image_height": int(feature_file.attrs["image_height"]),
            "pixel_step": int(feature_file.attrs["pixel_step"]),
            "n_sampled_pixels": int(
                feature_file.attrs["n_sampled_pixels"]
            ),
            "feature_size": int(feature_file.attrs["feature_size"]),
            "channel_order": str(feature_file.attrs["channel_order"]),
            "flatten_order": str(feature_file.attrs["flatten_order"]),
            "frame_stride": int(feature_file.attrs["frame_stride"]),
            "value_dtype": str(feature_file.attrs["value_dtype"]),
            "value_range": str(feature_file.attrs["value_range"]),
        }
    # end with h5py.File
    return features, effective_fps, metadata
# EOF
