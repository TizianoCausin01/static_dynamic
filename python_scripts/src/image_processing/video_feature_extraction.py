from pathlib import Path
import re

import cv2
import h5py
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoImageProcessor


DEFAULT_REPO_URLS = {
    "dino_v3_l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "ijepa_vith14_1k": "facebook/ijepa_vith14_1k",
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

OUTPUT:
    - feature_files: dict[str, h5py.File] -> open files keyed by layer name
"""
def open_feature_files(output_dir, ann, layers, dataset_name, model_source, frame_stride):
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
