import os, sys, yaml
import numpy as np
from pathlib import Path
import h5py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet
import re
ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])
from useful_stuff.general_utils.utils import TimeSeries


"""
match_timed_static_movie_rasters
Align standard-image, timed-image, and movie rasters by stimulus identity.

INPUT:
    - image_rasters: np.ndarray -> channels x time x image-session stimuli.
    - image_names: list[str] -> image-session stimulus names.
    - movie_rasters: np.ndarray -> channels x time x movie-session stimuli.
    - movie_names: list[str] -> movie-session stimulus names.
    - image_prefix: str -> standard-image filename prefix.
    - image_2000ms_prefix: str -> 2000 ms image filename prefix.
    - image_2250ms_prefix: str -> 2250 ms image filename prefix.
    - movie_prefix: str -> movie filename prefix.

OUTPUT:
    - aligned_image_rasters: np.ndarray -> aligned standard-image rasters.
    - aligned_movie_rasters: np.ndarray -> aligned movie rasters.
    - aligned_image_2000ms_rasters: np.ndarray -> aligned 2000 ms rasters.
    - aligned_image_2250ms_rasters: np.ndarray -> aligned 2250 ms rasters.
    - shared_stimuli: list[str] -> identities shared by all four conditions.
    - aligned_names: dict[str, list[str]] -> retained names for every condition.
"""
def match_timed_static_movie_rasters(
        image_rasters, image_names, movie_rasters, movie_names,
        image_prefix="img_", image_2000ms_prefix="img_2000ms_",
        image_2250ms_prefix="img_2250ms_", movie_prefix="vid_",
        ):
    image_indices_by_condition = {
        "image": {},
        "image_2000ms": {},
        "image_2250ms": {},
    }

    # Assign each image-session stimulus to one mutually exclusive condition.
    for index, name in enumerate(image_names):
        stimulus_stem = Path(name).stem
        if stimulus_stem.startswith(image_2000ms_prefix):
            condition = "image_2000ms"
            identity = stimulus_stem[len(image_2000ms_prefix):]
        elif stimulus_stem.startswith(image_2250ms_prefix):
            condition = "image_2250ms"
            identity = stimulus_stem[len(image_2250ms_prefix):]
        elif stimulus_stem.startswith(image_prefix):
            condition = "image"
            identity = stimulus_stem[len(image_prefix):]
        else:
            continue
        # end if stimulus_stem.startswith

        condition_indices = image_indices_by_condition[condition]
        if identity in condition_indices:
            raise ValueError(
                f"Duplicate {condition} stimulus identity: {identity!r}."
            )
        # end if identity in condition_indices
        condition_indices[identity] = index
    # end for index, name

    movie_index_by_identity = {}
    for index, name in enumerate(movie_names):
        stimulus_stem = Path(name).stem
        if not stimulus_stem.startswith(movie_prefix):
            continue
        # end if not stimulus_stem.startswith

        identity = stimulus_stem[len(movie_prefix):]
        if identity in movie_index_by_identity:
            raise ValueError(f"Duplicate movie stimulus identity: {identity!r}.")
        # end if identity in movie_index_by_identity
        movie_index_by_identity[identity] = index
    # end for index, name

    standard_indices = image_indices_by_condition["image"]
    image_2000ms_indices = image_indices_by_condition["image_2000ms"]
    image_2250ms_indices = image_indices_by_condition["image_2250ms"]
    shared_stimuli = [
        identity for identity in standard_indices
        if identity in image_2000ms_indices
        and identity in image_2250ms_indices
        and identity in movie_index_by_identity
    ]
    if len(shared_stimuli) < 2:
        raise ValueError("Need at least two stimuli shared by all four conditions.")
    # end if len(shared_stimuli) < 2

    orders = {
        "image": [standard_indices[key] for key in shared_stimuli],
        "movie": [movie_index_by_identity[key] for key in shared_stimuli],
        "image_2000ms": [
            image_2000ms_indices[key] for key in shared_stimuli
        ],
        "image_2250ms": [
            image_2250ms_indices[key] for key in shared_stimuli
        ],
    }
    aligned_names = {
        "image": [image_names[index] for index in orders["image"]],
        "movie": [movie_names[index] for index in orders["movie"]],
        "image_2000ms": [
            image_names[index] for index in orders["image_2000ms"]
        ],
        "image_2250ms": [
            image_names[index] for index in orders["image_2250ms"]
        ],
    }
    return (
        image_rasters[:, :, orders["image"]],
        movie_rasters[:, :, orders["movie"]],
        image_rasters[:, :, orders["image_2000ms"]],
        image_rasters[:, :, orders["image_2250ms"]],
        shared_stimuli,
        aligned_names,
    )
# EOF


"""
min_max_normalization
Normalizes every neural channel to [0, 1] using all non-channel dimensions.
Optional reference data allows multiple arrays to share the same channel bounds.

INPUT:
    - neural_data: np.ndarray -> channels x time x stimuli neural responses
    - reference_data: np.ndarray | None -> channels x samples data used to estimate
        the per-channel minimum and maximum; defaults to neural_data

OUTPUT:
    - normalized_data: np.ndarray -> neural_data scaled independently per channel
"""
def min_max_normalization(neural_data, reference_data=None):
    neural_data = np.asarray(neural_data)
    if neural_data.ndim < 2:
        raise ValueError("neural_data must have channels on axis 0 and at least one data axis")
    # end if neural_data.ndim < 2

    if reference_data is None:
        reference_data = neural_data
    else:
        reference_data = np.asarray(reference_data)
    # end if reference_data is None

    if reference_data.ndim < 2 or reference_data.shape[0] != neural_data.shape[0]:
        raise ValueError(
            "reference_data must have the same number of channels as neural_data"
        )
    # end if reference_data.ndim < 2

    reference_axes = tuple(range(1, reference_data.ndim))
    bounds_shape = (neural_data.shape[0],) + (1,) * (neural_data.ndim - 1)
    min_firing_rate = reference_data.min(axis=reference_axes).reshape(bounds_shape)
    max_firing_rate = reference_data.max(axis=reference_axes).reshape(bounds_shape)
    firing_rate_range = max_firing_rate - min_firing_rate

    # Constant channels carry no representational information; map them to zero.
    safe_range = np.where(firing_rate_range == 0, 1, firing_rate_range)
    normalized_data = (neural_data - min_firing_rate) / safe_range
    normalized_data = np.where(firing_rate_range == 0, 0, normalized_data)
    return normalized_data
# EOF


"""
decode_matlab_strings
Decodes MATLAB strings stored in a v7.3 .mat file (HDF5 format) into Python strings.
1) Iterates over HDF5 object references pointing to MATLAB char arrays
2) Reads the corresponding uint16 character codes
3) Converts character codes to Python characters and joins them into strings

INPUT:
- h5file: h5py.File -> open HDF5 file corresponding to a MATLAB v7.3 .mat file
- ref_array: np.ndarray -> array of HDF5 object references to MATLAB char arrays

OUTPUT:
- strings: list of str -> decoded MATLAB strings
"""
def decode_matlab_strings(h5file, ref_array):
    strings = []
    for ref in ref_array.squeeze():
        chars = h5file[ref][:]
        s = ''.join(chr(c) for c in chars.flatten()) # MATLAB chars are usually stored as Nx1 uint16
        strings.append(s)
    return strings
# EOF


"""
load_natraster
Loads MATLAB v7.3 natural-stimulus rasters and their corresponding condition names.

INPUT:
    - mat_path: str | Path -> path to the natraster MATLAB file

OUTPUT:
    - rasters: np.ndarray -> channels x time x stimuli raster array
    - stimulus_names: list[str] -> condition name for every stimulus axis entry
"""
def load_natraster(mat_path):
    with h5py.File(mat_path, "r") as f:
        # MATLAB v7.3 dimensions are exposed in reverse order by h5py.
        rasters = f["natraster"][:].transpose(2, 1, 0).astype(np.float32)
        stimulus_names = decode_matlab_strings(f, f["uniqueImage"][:])
    # end with h5py.File

    if rasters.shape[2] != len(stimulus_names):
        raise ValueError(
            "The natraster stimulus axis does not match the number of uniqueImage names: "
            f"{rasters.shape[2]} != {len(stimulus_names)}"
        )
    # end if rasters.shape[2]
    return rasters, stimulus_names
# EOF


"""
select_stimulus_rasters
Selects one stimulus modality while preserving MATLAB's uniqueImage ordering.

INPUT:
    - rasters: np.ndarray -> channels x time x stimuli raster array
    - stimulus_names: list[str] -> condition name for every stimulus axis entry
    - stimulus_prefix: str -> filename prefix identifying the modality, e.g. img_ or vid_

OUTPUT:
    - selected_rasters: np.ndarray -> rasters restricted to the requested modality
    - selected_names: list[str] -> selected names in the same order as selected_rasters
"""
def select_stimulus_rasters(rasters, stimulus_names, stimulus_prefix):
    stimulus_indices = [
        index for index, name in enumerate(stimulus_names)
        if Path(name).stem.startswith(stimulus_prefix)
    ]
    if len(stimulus_indices) < 2:
        raise ValueError(
            f"Need at least two {stimulus_prefix!r} stimuli, found {len(stimulus_indices)}."
        )
    # end if len(stimulus_indices)

    selected_names = [stimulus_names[index] for index in stimulus_indices]
    if len(selected_names) != len(set(selected_names)):
        raise ValueError(f"Duplicate {stimulus_prefix!r} names found in uniqueImage.")
    # end if len(selected_names)

    selected_rasters = rasters[:, :, stimulus_indices]
    return selected_rasters, selected_names
# EOF



"""
imagenet_val_dataloader
Builds a DataLoader for ImageNet validation images with standard ImageNet preprocessing by default.
INPUT:
    - paths: dict -> config paths dictionary containing the ImageNet root path
    - image_size: int -> resize and center-crop size for model input images
    - batch_size: int -> number of images loaded per batch
    - num_workers: int -> number of DataLoader subprocesses for image loading and transforms
    - shuffle: bool -> whether to shuffle validation images before batching
    - preprocess: callable | None -> optional image transform/processor applied to each PIL image

OUTPUT:
    - loader: DataLoader -> ImageNet validation DataLoader returning image and label batches
"""
def imagenet_val_dataloader(paths, image_size, batch_size, num_workers=0, shuffle=True, preprocess=None):
    # Use caller-provided preprocessing when a model needs its own transform/processor.
    if preprocess is None:
        preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Load only the validation split and check that the local ImageNet copy is complete.
    dataset = ImageNet(root=paths["imagenet_path"], split="val", transform=preprocess)
    assert len(dataset) == 50_000, f"Expected 50,000 validation images, found {len(dataset):,}"

    # Use local worker subprocesses for data loading; pin memory only helps with CUDA transfer.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True and torch.cuda.is_available(),
    )
    return dataset, loader
# EOF


"""
load_img_natraster
Loads and preprocesses natural image raster data for a given monkey/session.

1) Loads the MATLAB v7.3 natraster file (HDF5 format)
2) Casts data to float32 and reorders axes to (neurons, time, trials)
3) Optionally slices the signal to a specific brain area
4) Wraps the data in a TimeSeries object
5) Resamples the signal to the target sampling frequency

INPUT:
- paths: dict[str, str] -> dictionary containing base data paths
- cfg: Cfg -> configuration object with required attributes:
    * monkey_name: str
    * date: str
    * new_fs: float
    * brain_area (optional): str

OUTPUT:
- rasters: TimeSeries -> preprocessed neural raster time series
"""
def load_img_natraster(paths: dict[str: str], monkey_name, date, new_fs=None, brain_area=None):
    rasters_path = f"{paths['data_path']}/data/{monkey_name}_natraster{date}.mat"
    with h5py.File(rasters_path, "r") as f:
        rasters = f["natraster"][:]      
    rasters = rasters.astype(np.float32)
    rasters = rasters.transpose(2, 1, 0)
    rasters = TimeSeries(rasters, 1000)
    if brain_area is not None:
            brain_areas_obj = BrainAreas(monkey_name)
            rasters = brain_areas_obj.slice_brain_area(rasters, brain_area)
    # end if brain_area is not None:
    if new_fs is not None:
        rasters.resample(new_fs)
    # if new_fs is not None:
    return rasters
# EOF


"""
map_image_order_from_ann_to_monkey
Creates an index mapping to align ANN image order with monkey presentation order.

What this function does:
1) Loads the list of images presented to the monkey from a MATLAB file
2) Decodes MATLAB string references into Python strings
3) Removes duplicate image names while preserving order
4) Extracts the ANN image presentation order from the dataset
5) Computes the index mapping from monkey order to ANN order

INPUT:
- paths: dict -> dictionary with base paths
- monkey_name: str -> monkey identifier
- date: str -> experiment date
- dataset: torchvision.datasets.ImageFolder -> ANN image dataset

OUTPUT:
- mapping_idx: list[int] -> indices to reorder ANN features to monkey order
"""
def map_image_order_from_ann_to_monkey(paths, monkey_name, date, dataset):
    allimgs_path = f"{paths['data_path']}/data/{monkey_name}_allimages{date}.mat"
    with h5py.File(allimgs_path, "r") as f:
        try:
            refs = f["allimages"][:]      # shape (N, 1) of object refs
        except KeyError:
            refs = f["uniqueImage"][:]
        # end try:
        monkey_presentation_order = decode_matlab_strings(f, refs)
        monkey_presentation_order = sorted(set(monkey_presentation_order))
    ann_presentation_order = [os.path.basename(path) for path, _ in dataset.samples] # creates the order with which images are presented to the ANN
    if os.path.basename(Path(dataset.root))=="talia_20each_tizi": # little detour because I have changed the filenames for talia_20each_tizi
        monkey_presentation_order = rename_talia_dataset(monkey_presentation_order)
    # end if dataset=="talia_20each_tizi":
    mapping_idx = [ann_presentation_order.index(x) for x in monkey_presentation_order] # Creates a mapping from the monkey to the ann presentation order
    newly_ordered_ann = [ann_presentation_order[i] for i in mapping_idx]
    assert newly_ordered_ann == monkey_presentation_order
    return mapping_idx # by applying this to the ann features we'll get the same order as the monkeys'
# EOF


"""
rename_talia_dataset
just renaming the names the same way I did in the folder also in the uniqueImages file, 
otherwise I wouldn't be able to do the correct mapping. 
We add an underscore between the image name and the number and we take off the spaces.
"""
def rename_talia_dataset(monkey_presentation_order):
    monkey_presentation_order_renamed = []
    for f in monkey_presentation_order:
        # Step 1: insert underscore before first number following a letter
        newname = re.sub(r'([a-zA-Z])([0-9])', r'\1_\2', f)
        # Step 2: remove spaces
        newname = newname.replace(' ', '')
        # Rename if changed
        monkey_presentation_order_renamed.append(newname)
    # end for f in monkey_presentation_order:
    return monkey_presentation_order_renamed
# EOF


"""
BrainAreas
Utility class for slicing neural data into predefined brain areas.
1) Loads brain-area channel indices from a YAML configuration file
2) Validates input rasters against the expected number of channels
3) Extracts and concatenates channel ranges corresponding to a given brain area

INPUT:
- monkey_name: str -> identifier used to select the correct brain-area mapping

OUTPUT (slice_brain_area):
- brain_area_response: np.ndarray -> subset of rasters corresponding to the selected brain area
"""
class BrainAreas:
    def __init__(self, monkey_name: str):
        self.monkey_name = monkey_name
        with open("../../brain_areas.yaml", "r") as f:
            config = yaml.safe_load(f)
        try:
            self.areas_idx = config[self.monkey_name]
            self.brain_areas = [k for k in self.areas_idx.keys() if k!='n_chan']
        except KeyError:
            raise KeyError(f"Monkey '{self.monkey_name}' not found.", f"Supported monkeys {list(config.keys())}") from None
        # end try:
    # EOF
    # --- GETTERS ---
    def get_brain_areas_idx(self):
        return self.areas_idx
    #EOF
    def get_brain_areas(self):
        return self.brain_areas
    #EOF
    # --- OTHER FUNCTIONS ---
    def slice_brain_area(self, rasters: "TimeSeries", brain_area_name: str):
        if rasters.get_array().shape[0] < self.areas_idx["n_chan"][0]:
            raise ValueError(f"Rasters of shape {rasters.get_array().shape} doesn't match the original number of channels ({self.areas_idx["n_chan"]}).")
        # end if rasters.shape[0] < self.areas_idx["n_chan"][0]:
        try:
            target_brain_area = self.areas_idx[brain_area_name]
        except KeyError:
            raise KeyError(f"Brain area '{brain_area_name}' not found for monkey '{self.monkey_name}'.", f"Supported brain areas: {list(self.areas_idx.keys())}") from None
            
        except TypeError:
            if isinstance(brain_area_name, list) and len(brain_area_name) == 2:
                for idx in brain_area_name:
                    if idx > self.areas_idx["n_chan"][0]:
                        raise ValueError(f"Indices passed {brain_area_name} don't match the original number of channels ({self.areas_idx["n_chan"]}).")
                    # end if idx > self.areas_idx["n_chan"][0]:
                # end for idx in brain_area_name:
                target_brain_area = [brain_area_name] # it's setting the limits in terms of channels idx where we don't have precise info about a brain area, wrapping them in a list of lists
            else:
                raise TypeError(f"brain_area_name should be either a str or a list of len 2.")
            # end if isinstance(brain_area_name, list) and len(brain_area_name) == 2:
        # end try:
        brain_area_response = []
        for lims in target_brain_area:
            start, end = lims
            brain_area_response.append(rasters.get_array()[start:end, ...])
        # end for lims in target_brain_area:
        brain_area_response = np.concatenate(brain_area_response)
        brain_area_response = TimeSeries(brain_area_response, rasters.fs)
        return brain_area_response
    # EOF
# EOC
