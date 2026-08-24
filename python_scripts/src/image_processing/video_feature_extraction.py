from pathlib import Path
import re

import cv2
import h5py
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoVideoProcessor


DEFAULT_REPO_URLS = {
    "dino_v3_l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "ijepa_vith14_1k": "facebook/ijepa_vith14_1k",
    "vjepa2_vitl_fpc64_256": "facebook/vjepa2-vitl-fpc64-256",
}


"""
list_video_feature_files
Lists layer-specific feature files in natural layer order.

INPUT:
    - output_dir: str | Path -> directory containing extracted HDF5 feature files
    - model_name: str -> model prefix used during extraction
    - dataset_name: str -> dataset label used during extraction
    - pooling: str | None -> feature pooling used during extraction

OUTPUT:
    - feature_paths: list[Path] -> matching HDF5 files ordered by layer number
"""
def list_video_feature_files(
        output_dir, model_name, dataset_name, pooling="mean",
        ):
    pooling_name = "none" if pooling is None else pooling
    pattern = f"{model_name}_*_{dataset_name}_{pooling_name}pool.h5"
    feature_paths = list(Path(output_dir).glob(pattern))

    def natural_sort_key(path):
        parts = re.split(r"(\d+)", path.name)
        return [int(part) if part.isdigit() else part.lower() for part in parts]
    # EOF

    feature_paths.sort(key=natural_sort_key)
    if not feature_paths:
        raise FileNotFoundError(
            f"No video feature files matching {pattern!r} in {output_dir}"
        )
    # end if not feature_paths
    return feature_paths
# EOF


"""
stimulus_key
Returns a modality-independent stimulus identity from a stimulus filename.

INPUT:
    - stimulus_name: str | Path -> stimulus filename or HDF5 dataset name

OUTPUT:
    - key: str -> filename stem without an img_ or vid_ prefix
"""
def stimulus_key(stimulus_name):
    key = Path(stimulus_name).stem
    for prefix in ("img_", "vid_"):
        if key.startswith(prefix):
            return key[len(prefix):]
        # end if key.startswith(prefix)
    # end for prefix
    return key
# EOF


"""
match_feature_stimulus_names
Matches neural stimulus names to HDF5 feature datasets by identity while preserving
the neural/MATLAB order. Modality prefixes and filename extensions may differ.

INPUT:
    - feature_path: str | Path -> one layer-specific HDF5 feature file
    - stimulus_names: list[str] -> neural stimulus names in the required order

OUTPUT:
    - feature_names: list[str] -> exact HDF5 dataset names in neural stimulus order
"""
def match_feature_stimulus_names(feature_path, stimulus_names):
    with h5py.File(feature_path, "r") as feature_file:
        dataset_names = [
            name for name, value in feature_file.items()
            if isinstance(value, h5py.Dataset)
        ]
    # end with h5py.File

    names_by_key = {}
    for dataset_name in dataset_names:
        key = stimulus_key(dataset_name)
        if key in names_by_key:
            raise ValueError(
                f"Feature file {Path(feature_path).name} contains multiple datasets "
                f"for stimulus identity {key!r}."
            )
        # end if key in names_by_key
        names_by_key[key] = dataset_name
    # end for dataset_name

    stimulus_keys = [stimulus_key(name) for name in stimulus_names]
    if len(stimulus_keys) != len(set(stimulus_keys)):
        raise ValueError("Neural stimulus names contain duplicate stimulus identities.")
    # end if len(stimulus_keys)

    missing_keys = [key for key in stimulus_keys if key not in names_by_key]
    if missing_keys:
        raise KeyError(
            f"{Path(feature_path).name} is missing {len(missing_keys)} neural stimuli: "
            f"{missing_keys[:5]}"
        )
    # end if missing_keys
    return [names_by_key[key] for key in stimulus_keys]
# EOF


"""
load_aligned_video_features
Loads one layer in a requested stimulus order, either for all frames or one frame.

INPUT:
    - feature_path: str | Path -> layer-specific HDF5 feature file
    - video_names: list[str] -> dataset names in the required stimulus order
    - frame_index: int | None -> selected frame, or None to load every frame

OUTPUT:
    - features: np.ndarray -> features x stimuli for one frame, otherwise
        features x frames x stimuli
    - layer_name: str -> layer name stored in the HDF5 metadata
    - source_fps: float -> decoded video frame rate shared by the stimuli
"""
def load_aligned_video_features(feature_path, video_names, frame_index=None):
    with h5py.File(feature_path, "r") as feature_file:
        missing_names = [name for name in video_names if name not in feature_file]
        if missing_names:
            raise KeyError(
                f"{Path(feature_path).name} is missing {len(missing_names)} stimuli: "
                f"{missing_names[:5]}"
            )
        # end if missing_names

        layer_name = str(feature_file.attrs["layer_name"])
        source_fps_values = {
            float(feature_file[name].attrs["source_fps"]) for name in video_names
        }
        if len(source_fps_values) != 1:
            raise ValueError(
                f"Stimuli in {Path(feature_path).name} have different frame rates: "
                f"{sorted(source_fps_values)}"
            )
        # end if len(source_fps_values)
        source_fps = source_fps_values.pop()

        if frame_index is None:
            frame_counts = {feature_file[name].shape[0] for name in video_names}
            if len(frame_counts) != 1:
                raise ValueError(
                    f"Stimuli in {Path(feature_path).name} have different frame counts: "
                    f"{sorted(frame_counts)}"
                )
            # end if len(frame_counts)
            features = np.stack(
                [feature_file[name][:] for name in video_names], axis=0
            ).transpose(2, 1, 0)
        else:
            features = np.stack(
                [feature_file[name][frame_index] for name in video_names], axis=1
            )
        # end if frame_index is None
    # end with h5py.File
    return features, layer_name, source_fps
# EOF


"""
get_model_source
Returns the explicit Hugging Face repository or the project default for a supported model.

INPUT:
    - model_name: str -> project model name passed to imgANN
    - repo_url: str | None -> optional explicit model repository

OUTPUT:
    - model_source: str -> model source used to load the processor and model
"""
def get_model_source(model_name, repo_url=None):
    if repo_url is not None:
        return repo_url
    if model_name in DEFAULT_REPO_URLS:
        return DEFAULT_REPO_URLS[model_name]
    return model_name
# EOF


"""
list_video_paths
Lists the sorted videos selected for sequential feature extraction.

INPUT:
    - stimuli_dir: str | Path -> folder containing the stimulus videos
    - video_pattern: str -> glob pattern used to select videos
    - max_videos: int | None -> optional cap used for testing or partial runs

OUTPUT:
    - video_paths: list[Path] -> sorted paths matching video_pattern
"""
def list_video_paths(stimuli_dir, video_pattern="vid_*.mp4", max_videos=None):
    stimuli_dir = Path(stimuli_dir)
    if not stimuli_dir.exists():
        raise FileNotFoundError(f"Stimuli directory does not exist: {stimuli_dir}")

    video_paths = sorted(path for path in stimuli_dir.glob(video_pattern) if path.is_file())
    if max_videos is not None:
        video_paths = video_paths[:max_videos]
    if not video_paths:
        raise FileNotFoundError(f"No videos matching {video_pattern!r} found in {stimuli_dir}")
    return video_paths
# EOF


"""
build_frame_preprocessor
Builds the model-specific frame preprocessor used before imgANN inference.

INPUT:
    - pkg: str -> model package used by imgANN
    - model_source: str -> Hugging Face repository or model identifier
    - img_size: int -> square model input size

OUTPUT:
    - preprocessor: AutoImageProcessor | torchvision.transforms.Compose -> RGB frame preprocessor
"""
def build_frame_preprocessor(pkg, model_source, img_size):
    if pkg == "hf":
        preprocessor = AutoImageProcessor.from_pretrained(model_source)
        preprocessor.size = {"height": img_size, "width": img_size}
        return preprocessor

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
# EOF


"""
preprocess_frames
Converts a list of decoded RGB frames into a model input tensor.

INPUT:
    - frames: list[np.ndarray] -> decoded RGB frames
    - preprocessor: AutoImageProcessor | torchvision.transforms.Compose -> frame preprocessor
    - pkg: str -> model package used by imgANN
    - device: str | torch.device -> inference device
    - dtype: torch.dtype -> model input dtype

OUTPUT:
    - pixel_values: torch.Tensor -> preprocessed frame batch
"""
def preprocess_frames(frames, preprocessor, pkg, device, dtype):
    if pkg == "hf":
        pixel_values = preprocessor(images=frames, return_tensors="pt")["pixel_values"]
    else:
        pixel_values = torch.stack([preprocessor(frame) for frame in frames])
    return pixel_values.to(device=device, dtype=dtype)
# EOF


"""
build_video_preprocessor
Builds the model-specific preprocessor used before vidANN inference.

INPUT:
    - pkg: str -> model package used by vidANN
    - model_source: str -> Hugging Face repository or model identifier
    - img_size: int -> square model input size

OUTPUT:
    - preprocessor: AutoVideoProcessor | torchvision.transforms.Compose ->
        video-frame preprocessor
"""
def build_video_preprocessor(pkg, model_source, img_size):
    if pkg == "hf":
        return AutoVideoProcessor.from_pretrained(
            model_source, crop_size=img_size,
        )
    # end if Hugging Face

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
# EOF


"""
preprocess_video_frames
Preprocesses decoded RGB frames once and concatenates them in the vidANN input layout.

INPUT:
    - frames: list[np.ndarray] -> decoded RGB frames in temporal order
    - preprocessor: AutoVideoProcessor | torchvision.transforms.Compose ->
        video-frame preprocessor
    - pkg: str -> model package used by vidANN
    - architecture: str -> "transformer" or "cnn"
    - chunk_size: int -> number of frames transformed per CPU chunk
    - dtype: torch.dtype -> stored tensor dtype

OUTPUT:
    - pixel_values: torch.Tensor -> CPU tensor B x T x C x H x W for
        transformers or B x C x T x H x W for CNNs
"""
def preprocess_video_frames(
        frames, preprocessor, pkg, architecture,
        chunk_size=64, dtype=torch.float32,
        ):
    architecture = architecture.lower()
    if architecture not in ("transformer", "cnn"):
        raise ValueError("architecture must be 'transformer' or 'cnn'.")
    # end if invalid architecture
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    # end if invalid chunk_size
    if not frames:
        raise ValueError("frames cannot be empty.")
    # end if frames is empty

    processed_chunks = []
    for chunk_start in range(0, len(frames), chunk_size):
        chunk_stop = min(chunk_start + chunk_size, len(frames))
        frame_chunk = frames[chunk_start:chunk_stop]
        expected_frames = chunk_stop - chunk_start

        if pkg == "hf":
            processor_output = preprocessor(
                frame_chunk, return_tensors="pt",
            )
            if "pixel_values_videos" in processor_output:
                chunk_pixels = processor_output["pixel_values_videos"]
            elif "pixel_values" in processor_output:
                chunk_pixels = processor_output["pixel_values"]
            else:
                raise KeyError(
                    "The video processor returned neither pixel_values_videos "
                    "nor pixel_values."
                )
            # end if processor output key
        else:
            chunk_pixels = torch.stack([
                preprocessor(frame) for frame in frame_chunk
            ]).unsqueeze(0)
        # end if Hugging Face

        if chunk_pixels.ndim != 5:
            raise RuntimeError(
                "Video preprocessing must return a five-dimensional tensor, "
                f"got {tuple(chunk_pixels.shape)}."
            )
        # end if invalid video tensor

        # Normalize each processor's output to the layout required by vidANN.
        if architecture == "transformer":
            if chunk_pixels.shape[1] == expected_frames:
                normalized_chunk = chunk_pixels
            elif chunk_pixels.shape[2] == expected_frames:
                normalized_chunk = chunk_pixels.permute(0, 2, 1, 3, 4)
            else:
                raise RuntimeError(
                    "Could not identify the temporal axis in transformer "
                    f"preprocessor output {tuple(chunk_pixels.shape)}."
                )
            # end if transformer temporal axis
            temporal_axis = 1
        else:
            if chunk_pixels.shape[2] == expected_frames:
                normalized_chunk = chunk_pixels
            elif chunk_pixels.shape[1] == expected_frames:
                normalized_chunk = chunk_pixels.permute(0, 2, 1, 3, 4)
            else:
                raise RuntimeError(
                    "Could not identify the temporal axis in CNN "
                    f"preprocessor output {tuple(chunk_pixels.shape)}."
                )
            # end if CNN temporal axis
            temporal_axis = 2
        # end if transformer

        processed_chunks.append(
            normalized_chunk.to(device="cpu", dtype=dtype)
        )
    # end for chunk_start
    return torch.cat(processed_chunks, dim=temporal_axis)
# EOF


"""
decode_selected_video_frames
Decodes temporally ordered RGB frames with an optional stride and retained-frame cap.

INPUT:
    - video_path: str | Path -> local video file
    - frame_stride: int -> retain one decoded frame every frame_stride frames
    - max_frames: int | None -> optional cap on retained frames

OUTPUT:
    - frames_rgb: list[np.ndarray] -> retained RGB frames
    - source_fps: float -> source video frame rate
    - decoded_indices: np.ndarray -> source indices of retained frames
"""
def decode_selected_video_frames(
        video_path, frame_stride=1, max_frames=None,
        ):
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive.")
    # end if invalid frame_stride
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive when provided.")
    # end if invalid max_frames

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    # end if not cap.isOpened()

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames_rgb = []
    decoded_indices = []
    decoded_frame_index = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            # end if not ok

            if decoded_frame_index % frame_stride == 0:
                frames_rgb.append(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                )
                decoded_indices.append(decoded_frame_index)
                if max_frames is not None and len(frames_rgb) >= max_frames:
                    break
                # end if max_frames reached
            # end if retained frame
            decoded_frame_index += 1
        # end while
    finally:
        cap.release()
    # end try

    if not frames_rgb:
        raise RuntimeError(f"No frames decoded from {video_path}")
    # end if not frames_rgb
    return (
        frames_rgb,
        source_fps,
        np.asarray(decoded_indices, dtype=int),
    )
# EOF


"""
sliding_window_source_indices
Builds one fixed window of retained-frame indices ending at a target frame.

INPUT:
    - right_frame_index: int -> retained frame represented by the output
    - total_frames: int -> number of retained video frames
    - window_size: int -> fixed window length

OUTPUT:
    - window_indices: np.ndarray -> retained indices including left padding
    - num_left_padding: int -> number of repeated first-frame entries
"""
def sliding_window_source_indices(
        right_frame_index, total_frames, window_size,
        ):
    if not 0 <= right_frame_index < total_frames:
        raise IndexError(
            f"right_frame_index={right_frame_index} is outside "
            f"[0, {total_frames})."
        )
    # end if invalid right_frame_index
    if window_size < 1:
        raise ValueError("window_size must be positive.")
    # end if invalid window_size

    first_real_index = max(0, right_frame_index - window_size + 1)
    real_indices = np.arange(
        first_real_index, right_frame_index + 1, dtype=int,
    )
    num_left_padding = window_size - len(real_indices)
    window_indices = np.concatenate([
        np.zeros(num_left_padding, dtype=int),
        real_indices,
    ])
    return window_indices, num_left_padding
# EOF


"""
build_left_padded_video_window
Slices one fixed window and left-pads it by repeating the video's first frame.

INPUT:
    - video_tensor: torch.Tensor -> complete preprocessed video
    - right_frame_index: int -> retained frame at the window's right edge
    - window_size: int -> fixed window length
    - architecture: str -> "transformer" or "cnn"

OUTPUT:
    - window: torch.Tensor -> fixed video window in the architecture input layout
    - window_indices: np.ndarray -> retained source index at every window position
    - num_left_padding: int -> number of repeated first frames
"""
def build_left_padded_video_window(
        video_tensor, right_frame_index, window_size, architecture,
        ):
    architecture = architecture.lower()
    if architecture not in ("transformer", "cnn"):
        raise ValueError("architecture must be 'transformer' or 'cnn'.")
    # end if invalid architecture
    temporal_axis = 1 if architecture == "transformer" else 2
    total_frames = video_tensor.shape[temporal_axis]
    window_indices, num_left_padding = sliding_window_source_indices(
        right_frame_index, total_frames, window_size,
    )
    first_real_index = int(window_indices[num_left_padding])
    num_real_frames = right_frame_index - first_real_index + 1

    real_frames = video_tensor.narrow(
        temporal_axis, first_real_index, num_real_frames,
    )
    first_frame = video_tensor.narrow(temporal_axis, 0, 1)
    repeat_shape = [1] * video_tensor.ndim
    repeat_shape[temporal_axis] = num_left_padding
    left_padding = first_frame.repeat(*repeat_shape)
    window = torch.cat([left_padding, real_frames], dim=temporal_axis)

    if window.shape[temporal_axis] != window_size:
        raise RuntimeError(
            f"Expected a {window_size}-frame window, got "
            f"{window.shape[temporal_axis]} frames."
        )
    # end if incorrect window size
    return window, window_indices, num_left_padding
# EOF


"""
forward_imgann
Runs one preprocessed frame batch through the imgANN model.

INPUT:
    - ann: imgANN -> initialized model wrapper with active forward hooks
    - pixel_values: torch.Tensor -> preprocessed frame batch

OUTPUT:
    - None
"""
def forward_imgann(ann, pixel_values):
    if ann.pkg == "hf":
        ann.model(pixel_values=pixel_values)
    else:
        ann.model(pixel_values)
    return None
# EOF


"""
clear_model_cache
Releases cached accelerator memory after an inference batch.

INPUT:
    - device: str | torch.device -> active inference device

OUTPUT:
    - None
"""
def clear_model_cache(device):
    device = str(device)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    return None
# EOF


"""
feature_file_path
Builds the layer-specific HDF5 path for a video-feature dataset.

INPUT:
    - output_dir: str | Path -> directory receiving feature files
    - model_name: str -> project model name
    - layer_name: str -> hooked model layer name
    - dataset_name: str -> dataset label used in the filename
    - pooling: str | None -> imgANN pooling strategy

OUTPUT:
    - output_path: Path -> layer-specific HDF5 path
"""
def feature_file_path(output_dir, model_name, layer_name, dataset_name, pooling):
    pooling_name = "none" if pooling is None else pooling
    file_name = f"{model_name}_{layer_name}_{dataset_name}_{pooling_name}pool.h5"
    return Path(output_dir) / file_name
# EOF


"""
open_feature_files
Opens one output HDF5 file per layer and records extraction metadata.

INPUT:
    - output_dir: str | Path -> directory receiving feature files
    - ann: imgANN -> initialized model wrapper
    - layers: list[str] -> hooked model layers
    - dataset_name: str -> dataset label used in filenames
    - model_source: str -> model repository or identifier
    - frame_stride: int -> decoded-frame sampling stride
    - extra_metadata: dict | None -> additional extraction settings stored in every file

OUTPUT:
    - feature_files: dict[str, h5py.File] -> open files keyed by layer name
"""
def open_feature_files(
        output_dir, ann, layers, dataset_name, model_source, frame_stride,
        extra_metadata=None,
        ):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_files = {}
    common_metadata = {
        "model_name": ann.model_name,
        "model_source": model_source,
        "dataset_name": dataset_name,
        "pooling": "none" if ann.pooling is None else ann.pooling,
        "img_size": ann.img_size,
        "frame_stride": frame_stride,
    }
    if extra_metadata is not None:
        common_metadata.update(extra_metadata)
    # end if extra_metadata

    try:
        for layer in layers:
            output_path = feature_file_path(
                output_dir, ann.model_name, layer, dataset_name, ann.pooling
            )
            feature_file = h5py.File(output_path, "a")
            expected_metadata = {**common_metadata, "layer_name": layer}

            for attr_name, expected_value in expected_metadata.items():
                if (
                    attr_name in feature_file.attrs
                    and feature_file.attrs[attr_name] != expected_value
                ):
                    stored_value = feature_file.attrs[attr_name]
                    feature_file.close()
                    raise ValueError(
                        f"Existing file {output_path} has {attr_name}={stored_value!r}; "
                        f"expected {expected_value!r}."
                    )
                feature_file.attrs[attr_name] = expected_value
            # end for attr_name
            feature_files[layer] = feature_file
        # end for layer
    except Exception:
        for feature_file in feature_files.values():
            feature_file.close()
        raise
    # end try
    return feature_files
# EOF


"""
append_features
Appends a feature batch to an extendable HDF5 dataset.

INPUT:
    - feature_file: h5py.File -> open layer-specific HDF5 file
    - dataset_key: str -> temporary dataset key for the active video
    - features: np.ndarray -> batch features with frames on axis 0
    - compression: str | None -> optional HDF5 compression filter

OUTPUT:
    - None
"""
def append_features(feature_file, dataset_key, features, compression=None):
    if dataset_key not in feature_file:
        feature_file.create_dataset(
            dataset_key,
            shape=(0, *features.shape[1:]),
            maxshape=(None, *features.shape[1:]),
            dtype=features.dtype,
            chunks=True,
            compression=compression,
        )

    dataset = feature_file[dataset_key]
    start = dataset.shape[0]
    dataset.resize(start + features.shape[0], axis=0)
    dataset[start:] = features
    return None
# EOF


"""
extract_video_features
Decodes one video sequentially and streams hooked imgANN features into HDF5 files.

INPUT:
    - video_path: str | Path -> video whose frames are processed
    - ann: imgANN -> initialized model wrapper
    - preprocessor: AutoImageProcessor | torchvision.transforms.Compose -> frame preprocessor
    - feature_files: dict[str, h5py.File] -> layer-specific HDF5 files
    - layers: list[str] -> layers still missing this video
    - batch_size: int -> number of frames per forward call
    - frame_stride: int -> keep one frame every frame_stride decoded frames
    - max_frames: int | None -> optional cap on retained frames
    - dtype: torch.dtype -> inference dtype
    - compression: str | None -> optional HDF5 compression filter

OUTPUT:
    - n_frames: int -> number of extracted video frames
"""
def extract_video_features(
    video_path,
    ann,
    preprocessor,
    feature_files,
    layers,
    batch_size,
    frame_stride=1,
    max_frames=None,
    dtype=torch.float32,
    compression=None,
):
    video_path = Path(video_path)
    temporary_key = f"__incomplete__{video_path.name}"
    ann.create_forward_hook(layer_names=layers)

    for layer in layers:
        if temporary_key in feature_files[layer]:
            del feature_files[layer][temporary_key]
        # end if temporary_key
    # end for layer

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    decoded_frame_idx = 0
    n_frames = 0

    try:
        with torch.inference_mode():
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                if decoded_frame_idx % frame_stride == 0:
                    frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                decoded_frame_idx += 1

                reached_frame_cap = max_frames is not None and n_frames + len(frames) >= max_frames
                if len(frames) < batch_size and not reached_frame_cap:
                    continue

                if reached_frame_cap:
                    frames = frames[:max_frames - n_frames]
                pixel_values = preprocess_frames(
                    frames, preprocessor, ann.pkg, ann.device, dtype
                )
                forward_imgann(ann, pixel_values)

                for layer in layers:
                    features = ann.features.get(layer)
                    if features is None:
                        raise RuntimeError(f"Hook did not capture features for {layer}")
                    features = features.detach().float().cpu().numpy()
                    append_features(
                        feature_files[layer], temporary_key, features, compression=compression
                    )
                    ann.features[layer] = None
                # end for layer

                n_frames += len(frames)
                frames = []
                del pixel_values
                clear_model_cache(ann.device)

                if reached_frame_cap:
                    break
                # end if reached_frame_cap
            # end while True

            if frames:
                pixel_values = preprocess_frames(
                    frames, preprocessor, ann.pkg, ann.device, dtype
                )
                forward_imgann(ann, pixel_values)

                for layer in layers:
                    features = ann.features.get(layer)
                    if features is None:
                        raise RuntimeError(f"Hook did not capture features for {layer}")
                    features = features.detach().float().cpu().numpy()
                    append_features(
                        feature_files[layer], temporary_key, features, compression=compression
                    )
                    ann.features[layer] = None
                # end for layer
                n_frames += len(frames)
                del pixel_values
                clear_model_cache(ann.device)
            # end if frames
        # end with torch.inference_mode()
    finally:
        cap.release()
    # end try

    if n_frames == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")

    for layer in layers:
        dataset = feature_files[layer][temporary_key]
        dataset.attrs["source_path"] = str(video_path)
        dataset.attrs["source_fps"] = source_fps
        dataset.attrs["n_frames"] = n_frames
        feature_files[layer].move(temporary_key, video_path.name)
        feature_files[layer].flush()
    # end for layer
    return n_frames
# EOF


"""
extract_video_dataset_features
Extracts sequential frame features for every selected video and every requested layer.

INPUT:
    - ann: imgANN -> initialized model wrapper
    - preprocessor: AutoImageProcessor | torchvision.transforms.Compose -> frame preprocessor
    - video_paths: list[Path] -> videos processed in sorted order
    - output_dir: str | Path -> directory receiving layer-specific HDF5 files
    - dataset_name: str -> dataset label used in filenames
    - model_source: str -> model repository or identifier
    - batch_size: int -> number of frames per forward call
    - frame_stride: int -> keep one frame every frame_stride decoded frames
    - max_frames: int | None -> optional retained-frame cap per video
    - dtype: torch.dtype -> inference dtype
    - compression: str | None -> optional HDF5 compression filter
    - overwrite: bool -> whether to recompute existing video datasets

OUTPUT:
    - output_paths: dict[str, Path] -> layer-specific files keyed by layer name
"""
def extract_video_dataset_features(
    ann,
    preprocessor,
    video_paths,
    output_dir,
    dataset_name,
    model_source,
    batch_size,
    frame_stride=1,
    max_frames=None,
    dtype=torch.float32,
    compression=None,
    overwrite=False,
):
    layers = ann.get_relevant_layers()
    output_paths = {
        layer: feature_file_path(
            output_dir, ann.model_name, layer, dataset_name, ann.pooling
        )
        for layer in layers
    }
    feature_files = open_feature_files(
        output_dir, ann, layers, dataset_name, model_source, frame_stride
    )

    try:
        for video_idx, video_path in enumerate(video_paths, start=1):
            if overwrite:
                for layer in layers:
                    if video_path.name in feature_files[layer]:
                        del feature_files[layer][video_path.name]
                    # end if video exists
                # end for layer

            missing_layers = [
                layer for layer in layers if video_path.name not in feature_files[layer]
            ]
            if not missing_layers:
                print(
                    f"[{video_idx}/{len(video_paths)}] "
                    f"Skipping complete video: {video_path.name}"
                )
                continue

            print(
                f"[{video_idx}/{len(video_paths)}] Extracting {video_path.name} "
                f"for {len(missing_layers)} layers"
            )
            n_frames = extract_video_features(
                video_path,
                ann,
                preprocessor=preprocessor,
                feature_files=feature_files,
                layers=missing_layers,
                batch_size=batch_size,
                frame_stride=frame_stride,
                max_frames=max_frames,
                dtype=dtype,
                compression=compression,
            )
            print(f"Saved {n_frames} frames for {video_path.name}")
        # end for video_path
    finally:
        ann.clear_hooks()
        for feature_file in feature_files.values():
            feature_file.close()
        # end for feature_file
    # end try
    return output_paths
# EOF


"""
stack_sliding_video_windows
Builds and batches fixed windows whose right edges advance by one retained frame.

INPUT:
    - video_tensor: torch.Tensor -> complete preprocessed video with batch size one
    - right_frame_indices: list[int] | range -> target retained-frame indices
    - window_size: int -> fixed number of frames per window
    - architecture: str -> "transformer" or "cnn"

OUTPUT:
    - window_batch: torch.Tensor -> windows stacked along the batch axis
"""
def stack_sliding_video_windows(
        video_tensor, right_frame_indices, window_size, architecture,
        ):
    if video_tensor.shape[0] != 1:
        raise ValueError(
            "stack_sliding_video_windows expects one preprocessed video."
        )
    # end if batch size is not one

    windows = []
    for right_frame_index in right_frame_indices:
        window, _, _ = build_left_padded_video_window(
            video_tensor,
            right_frame_index=right_frame_index,
            window_size=window_size,
            architecture=architecture,
        )
        windows.append(window)
    # end for right_frame_index
    if not windows:
        raise ValueError("right_frame_indices cannot be empty.")
    # end if windows is empty
    return torch.cat(windows, dim=0)
# EOF


"""
extract_sliding_video_features
Extracts one vidANN feature per retained frame using fixed left-padded sliding windows.

INPUT:
    - video_path: str | Path -> video processed sequentially
    - ann: vidANN -> initialized video model wrapper
    - preprocessor: object -> model-specific video preprocessor
    - feature_files: dict[str, h5py.File] -> open layer-specific HDF5 files
    - layers: list[str] -> hooked layers still missing this video
    - window_size: int -> fixed frames per model window
    - batch_size: int -> number of overlapping windows per model call
    - frame_stride: int -> retain one decoded frame every frame_stride frames
    - max_frames: int | None -> optional cap on retained frames
    - preprocess_chunk_size: int -> frames transformed per CPU chunk
    - dtype: torch.dtype -> model input dtype
    - model_input_name: str -> video tensor keyword accepted by the model
    - model_forward_kwargs: dict | None -> additional model forward arguments
    - compression: str | None -> optional HDF5 compression filter

OUTPUT:
    - n_frames: int -> number of frame-aligned window features saved
"""
def extract_sliding_video_features(
        video_path,
        ann,
        preprocessor,
        feature_files,
        layers,
        window_size,
        batch_size=1,
        frame_stride=1,
        max_frames=None,
        preprocess_chunk_size=64,
        dtype=torch.float32,
        model_input_name="pixel_values_videos",
        model_forward_kwargs=None,
        compression=None,
        ):
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    # end if invalid batch_size
    if window_size < 1:
        raise ValueError("window_size must be positive.")
    # end if invalid window_size
    if (
        ann.architecture == "transformer"
        and ann.last_frame
        and window_size % ann.tubelet_size
    ):
        raise ValueError(
            "window_size must be divisible by tubelet_size when "
            "last_frame=True for a transformer."
        )
    # end if incomplete transformer tubelet

    video_path = Path(video_path)
    temporary_key = f"__incomplete__{video_path.name}"
    ann.create_forward_hook(layer_names=layers)
    for layer in layers:
        if temporary_key in feature_files[layer]:
            del feature_files[layer][temporary_key]
        # end if temporary dataset exists
    # end for layer

    frames_rgb, source_fps, decoded_indices = decode_selected_video_frames(
        video_path,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    video_tensor = preprocess_video_frames(
        frames_rgb,
        preprocessor=preprocessor,
        pkg=ann.pkg,
        architecture=ann.architecture,
        chunk_size=preprocess_chunk_size,
        dtype=dtype,
    )
    del frames_rgb

    n_frames = len(decoded_indices)
    model_parameter = next(ann.model.parameters())
    model_device = model_parameter.device
    model_dtype = model_parameter.dtype
    model_forward_kwargs = (
        {} if model_forward_kwargs is None else dict(model_forward_kwargs)
    )

    with torch.inference_mode():
        for batch_start in range(0, n_frames, batch_size):
            batch_stop = min(batch_start + batch_size, n_frames)
            right_frame_indices = range(batch_start, batch_stop)
            window_batch = stack_sliding_video_windows(
                video_tensor,
                right_frame_indices=right_frame_indices,
                window_size=window_size,
                architecture=ann.architecture,
            ).to(device=model_device, dtype=model_dtype)

            model_inputs = {
                model_input_name: window_batch,
                **model_forward_kwargs,
            }
            hooked_features = ann.extract_features(
                model_inputs, method="hook",
            )

            for layer in layers:
                features = hooked_features.get(layer)
                if features is None:
                    raise RuntimeError(
                        f"Hook did not capture features for {layer}"
                    )
                # end if missing features
                if features.shape[0] != batch_stop - batch_start:
                    raise RuntimeError(
                        f"Expected {batch_stop - batch_start} window features "
                        f"for {layer}, got {features.shape[0]}."
                    )
                # end if incorrect feature batch

                append_features(
                    feature_files[layer],
                    temporary_key,
                    features.detach().float().cpu().numpy(),
                    compression=compression,
                )
                ann.features[layer] = None
            # end for layer

            del hooked_features, window_batch
            clear_model_cache(ann.device)
        # end for batch_start
    # end with inference_mode

    for layer in layers:
        dataset = feature_files[layer][temporary_key]
        dataset.attrs["source_path"] = str(video_path)
        dataset.attrs["source_fps"] = source_fps
        dataset.attrs["n_frames"] = n_frames
        dataset.attrs["first_source_frame_index"] = int(decoded_indices[0])
        dataset.attrs["last_source_frame_index"] = int(decoded_indices[-1])
        feature_files[layer].move(temporary_key, video_path.name)
        feature_files[layer].flush()
    # end for layer
    return n_frames
# EOF


"""
extract_sliding_video_dataset_features
Extracts sliding-window vidANN features for every selected video and hooked layer.

INPUT:
    - ann: vidANN -> initialized video model wrapper
    - preprocessor: object -> model-specific video preprocessor
    - video_paths: list[Path] -> videos processed in sorted order
    - output_dir: str | Path -> directory receiving layer-specific HDF5 files
    - dataset_name: str -> dataset label used in filenames
    - model_source: str -> model repository or identifier
    - window_size: int -> fixed frames per sliding window
    - batch_size: int -> overlapping windows processed per model call
    - frame_stride: int -> decoded-frame retention stride
    - max_frames: int | None -> optional retained-frame cap per video
    - preprocess_chunk_size: int -> frames transformed per CPU chunk
    - dtype: torch.dtype -> model input dtype
    - model_input_name: str -> video tensor keyword accepted by the model
    - model_forward_kwargs: dict | None -> additional model forward arguments
    - compression: str | None -> optional HDF5 compression filter
    - overwrite: bool -> whether to recompute existing video datasets

OUTPUT:
    - output_paths: dict[str, Path] -> layer-specific files keyed by layer name
"""
def extract_sliding_video_dataset_features(
        ann,
        preprocessor,
        video_paths,
        output_dir,
        dataset_name,
        model_source,
        window_size,
        batch_size=1,
        frame_stride=1,
        max_frames=None,
        preprocess_chunk_size=64,
        dtype=torch.float32,
        model_input_name="pixel_values_videos",
        model_forward_kwargs=None,
        compression=None,
        overwrite=False,
        ):
    layers = ann.get_relevant_layers()
    if not layers:
        raise ValueError("vidANN requires at least one relevant layer.")
    # end if no layers

    output_paths = {
        layer: feature_file_path(
            output_dir, ann.model_name, layer, dataset_name, ann.pooling,
        )
        for layer in layers
    }
    extra_metadata = {
        "extraction_mode": "sliding_window",
        "architecture": ann.architecture,
        "last_frame": ann.last_frame,
        "window_size_frames": window_size,
        "tubelet_size": ann.tubelet_size,
    }
    feature_files = open_feature_files(
        output_dir,
        ann,
        layers,
        dataset_name,
        model_source,
        frame_stride,
        extra_metadata=extra_metadata,
    )

    try:
        for video_index, video_path in enumerate(video_paths, start=1):
            if overwrite:
                for layer in layers:
                    if video_path.name in feature_files[layer]:
                        del feature_files[layer][video_path.name]
                    # end if video exists
                # end for layer
            # end if overwrite

            missing_layers = [
                layer
                for layer in layers
                if video_path.name not in feature_files[layer]
            ]
            if not missing_layers:
                print(
                    f"[{video_index}/{len(video_paths)}] "
                    f"Skipping complete video: {video_path.name}"
                )
                continue
            # end if video is complete

            print(
                f"[{video_index}/{len(video_paths)}] "
                f"Extracting {video_path.name} for "
                f"{len(missing_layers)} layers"
            )
            n_frames = extract_sliding_video_features(
                video_path,
                ann,
                preprocessor=preprocessor,
                feature_files=feature_files,
                layers=missing_layers,
                window_size=window_size,
                batch_size=batch_size,
                frame_stride=frame_stride,
                max_frames=max_frames,
                preprocess_chunk_size=preprocess_chunk_size,
                dtype=dtype,
                model_input_name=model_input_name,
                model_forward_kwargs=model_forward_kwargs,
                compression=compression,
            )
            print(
                f"Saved {n_frames} sliding-window features "
                f"for {video_path.name}"
            )
        # end for video_path
    finally:
        ann.clear_hooks()
        for feature_file in feature_files.values():
            feature_file.close()
        # end for feature_file
    # end try
    return output_paths
# EOF
