# Static–Dynamic Project Handoff

Last updated: 2026-07-14

This document is the context package for continuing development of this repository from a new ChatGPT/Codex account. It summarizes the scientific motivation, code organization, data and model pipelines, environment assumptions, current local state, and the most useful next steps.

## 1. Executive summary

`static_dynamic` is a scientific Python project for studying how static images and dynamic visual stimuli are represented by artificial neural networks and macaque visual cortex recordings.

The code currently supports four connected lines of work:

1. **Video stimulus construction**
   - Download candidate clips from online videos.
   - Interactively choose exact 2.5-second segments.
   - Interactively crop each segment to a moving or fixed 500 × 500 pixel square.
   - Manually curate a final set.
   - Standardize frame rate, frame count, codec, dimensions, and final-frame hold.
   - Produce matched static image-videos from the held final frame.

2. **ANN feature extraction and dimensionality reduction**
   - Load models through the external `useful_stuff.imgANN` interface.
   - Extract activations from relevant model layers.
   - Fit Incremental PCA or a Sparse Random Projection followed by PCA on ImageNet validation activations.
   - Project ImageNet or custom Livingstone Lab image stimuli into the saved low-dimensional feature spaces.

3. **Neural-data loading and alignment**
   - Load MATLAB/HDF5 natural-image raster recordings.
   - Resample neural time series and select monkey-specific brain areas.
   - Align ANN image order with the order used in the monkey experiment.

4. **Representational analyses**
   - Compare ANN and neural representations with dynamic RSA (`dRSA`).
   - Compare representational neighborhoods with dynamic Information Imbalance (`dynInformationImbalance`).
   - Study DINOv3 feature autocorrelation through videos and correlations with final-frame representations.
   - Visualize images that are extreme along ANN principal components or particularly exciting for neural populations.

The broad scientific question is inferred from the repository name and implemented analyses: **how do visual representations evolve from dynamic input toward a static/final-frame representation, and how do these model dynamics compare with macaque visual cortical dynamics?**

## 2. Repository and Git state

- Local repository: `/Users/tizianocausin/Desktop/static_dynamic`
- GitHub remote: `git@github.com:TizianoCausin01/static_dynamic.git`
- Current branch: `main`
- Current committed revision: `26d5f8b` (`Added code generation commandments`)
- `origin/main` points to the same revision.
- The repository does not currently contain a `README.md`, although `pyproject.toml` declares it as the project readme. This handoff can serve as the basis for a future README.

### Uncommitted work on 2026-07-14

Modified tracked files:

- `python_scripts/scripts/select_video_segments.py`
  - Adds lower-resolution decoding for smoother GUI review.
  - Adds fallback frame-reading behavior for difficult videos.
- `python_scripts/scripts/crop_video_segments.py`
  - Adds low-resolution/preloaded display frames while preserving full-resolution output.
  - Adds reset and discard controls.
  - Adds a crop-stage discarded-video directory.
- `python_scripts/scripts/dinov3_video_autocorrelation.ipynb`
  - Active development of DINOv3 video autocorrelation and RDM analyses.

Untracked files:

- `brain_areas.yaml`
- `python_scripts/scripts/create_padded_60fps_image_videos.py`
- `python_scripts/scripts/most_exciting_images_from_natraster.ipynb`
- `python_scripts/scripts/rename_padded_60fps_videos.py`
- `python_scripts/scripts/standardize_final_video_selection.py`
- `python_scripts/scripts/stimuli_generation_dev.ipynb`
- `python_scripts/scripts/test_padded_60fps_image_videos.py`
- `python_scripts/scripts/test_standardized_final_video_selection.py`

These files are important and must not be lost during account or machine migration. The video standardization, static image-video creation, reversible renaming, validation scripts, and brain-area definitions are currently only local.

Before starting new work, run:

```bash
cd /Users/tizianocausin/Desktop/static_dynamic
git status --short
git diff --stat
```

Do not reset, clean, or overwrite the working tree until the uncommitted work has been reviewed and committed or backed up.

## 3. Repository structure

```text
static_dynamic/
├── config.yaml
├── brain_areas.yaml                       # currently untracked
├── pyproject.toml
├── uv.lock
├── PROJECT_HANDOFF.md
├── bash_scripts/
│   ├── imagenet_ipca_dipsen.slrm
│   ├── par_imagenet_ipca_dipsen.slrm
│   ├── par_imagenet_ipca_o2.slrm
│   ├── par_imagenet_srp_pca_dipsen.slrm
│   ├── par_imagenet_srp_pca_o2.slrm
│   ├── par_imagenet_srp_pca_extraction_projection_dipsen.slrm
│   └── par_imagenet_srp_pca_extraction_projection_o2.slrm
└── python_scripts/
    ├── CODE_GENERATION_COMMANDMENTS.md
    ├── src/
    │   ├── image_processing/
    │   │   └── feature_extraction.py
    │   └── project_specific_utils/
    │       └── dataloader.py
    └── scripts/
        ├── run_*.py                       # executable model/data pipelines
        ├── select_video_segments.py       # interactive video endpoint selection
        ├── crop_video_segments.py         # interactive crop-path selection
        ├── standardize_*.py               # final stimulus encoding
        ├── create_*.py                    # matched static image-videos
        ├── rename_*.py                    # reversible final filename cleanup
        ├── test_*.py                      # artifact validators
        └── *.ipynb                        # analyses and development notebooks
```

Reusable functions should live under `python_scripts/src/`; scripts and notebooks should mainly configure and call those functions.

## 4. Environment setup

### Python environment

The project uses `uv` and currently pins:

- Python `3.14.0`
- PyTorch `2.10.0`
- torchvision `0.25.0`
- NumPy, SciPy, scikit-learn, OpenCV, h5py, matplotlib, mpi4py, timm, Transformers, Jupyter, and related scientific packages.
- CUDA 12.8 PyTorch wheels on Linux/Windows; standard wheels on Apple Silicon.

Observed local tools on 2026-07-14:

- `.venv/bin/python`: Python 3.14.0
- `uv`: 0.10.4
- `ffmpeg`/`ffprobe`: 8.0

Set up the environment with:

```bash
cd /Users/tizianocausin/Desktop/static_dynamic
uv sync --frozen
```

Activate it if desired:

```bash
source .venv/bin/activate
```

### Required external repository: `useful_stuff`

This project is not self-contained. It imports a sibling repository through a path in `config.yaml`:

```text
/Users/tizianocausin/Desktop/useful_stuff/python_scripts/src
```

The current local `useful_stuff` state is:

- Branch: `main`
- Revision: `738eaca` (`Added the possibility to have cosine_mean_cnt and double centered`)

Important imported utilities include:

- `imgANN`, model-layer discovery, hooks, and feature pooling.
- `compute_img_ipca` and `compute_img_srp`.
- `TimeSeries`, `get_device`, `print_wise`, and dtype conversion.
- `dRSA` and `dynInformationImbalance`.
- MPI helpers `parallel_setup` and `master_workers_queue`.

A new machine needs both repositories, with the `useful_stuff_path` in `config.yaml` updated accordingly. Because `useful_stuff` is not declared as a normal package dependency, `uv sync` alone is insufficient.

### Non-Python tools

The video workflow requires:

- `ffmpeg`
- `ffprobe`
- `yt-dlp`
- A graphical desktop for the OpenCV interactive selectors.

The cluster pipelines additionally require:

- MPI/OpenMPI
- Slurm
- CUDA for GPU jobs when available
- Model weights already present in the Hugging Face cache when offline mode is enabled

### Environment selection

All path-dependent scientific scripts use `MY_ENV` to choose a section of `config.yaml`.

Available values:

- `tiziano_mac_mini`
- `tiziano_local`
- `o2_cluster`
- `hpc_unitn`
- `dipsen_hpc`

Typical local setup:

```bash
export MY_ENV=tiziano_mac_mini
export MPLCONFIGDIR=/tmp/matplotlib
```

Important: several scripts default to `MY_ENV=dev`, but no `dev` entry exists in `config.yaml`. Always set `MY_ENV` explicitly before running them.

## 5. Configuration and data locations

`config.yaml` maps each environment to some or all of these paths:

- `src_path`: this repository's reusable Python modules.
- `data_path`: project data, generated stimuli, fitted components, and projected features.
- `useful_stuff_path`: external utilities repository.
- `imagenet_path`: local ImageNet root.
- `livingstone_lab`: Livingstone Lab data/stimulus root.

Local `tiziano_mac_mini` paths currently resolve to:

```text
src_path        /Users/tizianocausin/Desktop/static_dynamic/python_scripts/src
data_path       /Users/tizianocausin/sd_local
useful_stuff    /Users/tizianocausin/Desktop/useful_stuff/python_scripts/src
imagenet_path   /Users/tizianocausin/datasets/imagenet
livingstone_lab /Users/tizianocausin/livingstone_lab_local
```

The local `data_path` is approximately 69 GB and is intentionally outside Git:

```text
/Users/tizianocausin/sd_local/
├── data/           # ~33 GB; neural and MATLAB data
├── models/         # ~13 GB; PCA/SRP objects and projected ANN features
├── possible_vids/  # ~20 GB; candidate and processed videos
└── stimuli/        # ~3.4 GB; final/static/dynamic stimuli
```

The `hpc_unitn` config currently lacks `imagenet_path` and `livingstone_lab`. Scripts requiring those paths will fail there until the entries are added.

No API keys, passwords, or access tokens were found in the tracked non-notebook text files. Nevertheless, do not upload private neural data or model caches to ChatGPT; share code and small metadata summaries only.

## 6. Video stimulus pipeline

The active final set contains 100 dynamic videos and 100 matched static image-videos.

### Step 1: download candidate clips

Script:

```text
python_scripts/scripts/run_download_clip.py
```

It uses `yt-dlp --download-sections` to download one or more short ranges without intentionally downloading the full source video. It retries with several YouTube client/format strategies and records source URLs in:

```text
<data_path>/possible_vids/links.txt
```

Example:

```bash
cd python_scripts/scripts
MY_ENV=tiziano_mac_mini ../../.venv/bin/python run_download_clip.py \
  --video_url 'https://www.youtube.com/watch?v=...' \
  --video_filename example_source \
  --starting_point 03:18.5 04:20 \
  --duration 2.5
```

Downloaded clips go to:

```text
<data_path>/possible_vids/
```

### Step 2: select exact 2.5-second segments

Script:

```text
python_scripts/scripts/select_video_segments.py
```

This OpenCV GUI selects an ending frame and saves the preceding fixed-duration segment. It skips sources that already have an output unless `--force` is used.

Default directories:

```text
input       <data_path>/possible_vids
segments    <input>/segments
discarded   <input>/discarded_vids
```

Main controls:

- Left/right or mouse wheel: one frame.
- Up/down: one second.
- Page Up/Page Down: five seconds.
- `p`: play source video.
- `s`, Enter, or Space: review selected segment.
- `n`: skip.
- `x`: move source to discarded videos.
- `q`: quit.

The modified local version decodes low-resolution frames for GUI responsiveness while writing the selected segment from the original full-resolution source.

### Step 3: choose crop paths and write 500 × 500 clips

Script:

```text
python_scripts/scripts/crop_video_segments.py
```

This OpenCV GUI creates a square crop whose center can move over time. Crop-center keyframes are linearly interpolated. The selected square is resized to the requested output size, currently 500 × 500 pixels.

Default directories:

```text
input       <data_path>/possible_vids/segments
output      <input>/cropped_segments
discarded   <input>/discarded_cropping_vids
```

Main controls:

- Click: place crop center at the current frame.
- `i`: add an intermediate keyframe.
- `r`: reset all centers.
- `d`: delete the current intermediate keyframe.
- `f`/`g`: first/last frame.
- Arrows or mouse wheel: move through frames.
- `+`/`-`: adjust source square by 25 pixels.
- `[`/`]`: adjust source square by 5 pixels.
- `c`: review a centered crop.
- `s`: review the selected crop path.
- `n`: skip.
- `x`: move the source to crop-stage discarded videos.
- `p`: play source.
- `q`: quit.

The current local version can preload downsampled display frames, but it always applies the final crop to the original-resolution frames.

### Step 4: manual final selection

The project currently organizes curated cropped clips under:

```text
<data_path>/possible_vids/final_video_selection/
├── yes/
├── maybe/
└── no/
```

There is no repository script for this classification; it appears to be a manual filesystem curation step.

Current state:

- `yes/`: 100 source videos.

### Step 5: standardize dynamic videos and add a final-frame hold

Script:

```text
python_scripts/scripts/standardize_final_video_selection.py
```

For the current 60 fps workflow:

1. Standardized video:
   - 500 × 500 pixels inherited from crop stage.
   - 60 fps constant frame rate.
   - 2.5 seconds plus the frame exactly at 2.5 seconds.
   - 151 frames (`60 × 2.5 + 1`).
   - H.264 High-compatible output, `yuv420p`, `avc1` tag.
2. Padded dynamic video:
   - 60 fps.
   - Exactly 3.0 seconds and 180 frames.
   - Final frame held through the final 0.5 seconds.
   - ProRes HQ, `yuv422p10le`, `.mov`.

Run:

```bash
.venv/bin/python python_scripts/scripts/standardize_final_video_selection.py \
  --input_dir /Users/tizianocausin/sd_local/possible_vids/final_video_selection/yes \
  --fps 60
```

Outputs:

```text
yes/standardized_60fps/
yes/standardized_60fps/padded_60fps/
```

Current state:

- 100 standardized 60 fps videos.
- 100 padded 60 fps videos.

### Step 6: create matched static image-videos

Script:

```text
python_scripts/scripts/create_padded_60fps_image_videos.py
```

For each padded dynamic video, this extracts the frame at 2.5 seconds and repeats it for:

- 0.5 seconds
- 60 fps
- 30 frames
- ProRes HQ
- `yuv422p10le`

Outputs go to:

```text
yes/standardized_60fps/padded_60fps/images/
```

Current state:

- 100 static image-videos.

### Step 7: shorten names reversibly

Script:

```text
python_scripts/scripts/rename_padded_60fps_videos.py
```

This renames padded dynamic videos and their matching static image-videos together. It groups long names by a cleaned base and creates names such as `base1.mov`, `base2.mov`, while writing:

```text
rename_mapping.csv
```

Use `--mode apply` to shorten names and `--mode restore` to reverse the operation. Never manually edit one folder without updating its matched counterpart.

### Step 8: validate final artifacts

Dynamic validator:

```bash
.venv/bin/python python_scripts/scripts/test_standardized_final_video_selection.py \
  --input_dir /Users/tizianocausin/sd_local/possible_vids/final_video_selection/yes \
  --fps 60
```

It checks file correspondence, dimensions, frame rate, exact frame counts, durations, codecs, pixel formats, and whether the padded final-frame interval is constant.

Static validator:

```bash
.venv/bin/python python_scripts/scripts/test_padded_60fps_image_videos.py
```

It checks that exactly 100 outputs are 500 × 500, 60 fps, 30 frames, 0.5 seconds, ProRes HQ, `yuv422p10le`, and contain identical decoded frames.

Both validation commands completed successfully on 2026-07-14.

## 7. ANN feature and dimensionality-reduction pipeline

The central reusable module is:

```text
python_scripts/src/image_processing/feature_extraction.py
```

It supports two dimensionality-reduction paths.

### Incremental PCA path

`ipca_imagenet_wrapper`:

1. Builds one expected output path per model layer.
2. Skips layers whose component files already exist.
3. Registers `imgANN` forward hooks only for missing layers.
4. Streams ImageNet validation activations through `IncrementalPCA`.
5. Saves one joblib `.pkl` per layer.

Canonical save pattern:

```text
<data_path>/models/imagenet_components/
<model>_<layer>_imagenet_val_<n_components>components_<pooling>pool.pkl
```

Serial runner:

```text
python_scripts/scripts/run_imagenet_ipca.py
```

MPI runner:

```text
python_scripts/scripts/run_par_imagenet_ipca.py
```

### SRP followed by PCA path

This is the main high-dimensional feature path for large ANN layers.

`srp_pca_dataset_wrapper`:

1. Determines the flattened dimensionality of each target layer.
2. Fits a `SparseRandomProjection` from the original activation space to `n_srp_components`.
3. Streams dataset activations through the ANN and SRP.
4. Concatenates SRP features for each layer.
5. Fits a standard PCA from the SRP space to `n_pca_components`.
6. Saves the SRP and PCA objects separately.

Typical current settings are:

- SRP: 10,000 dimensions.
- PCA: 1,000 dimensions.
- Component-fitting dataset: ImageNet validation.
- Pooling: `all` for full flattened layer features in the PCA pipelines.

Canonical component patterns:

```text
<model>_<layer>_<PCs_dataset>_<n_srp>_srp_components_<pooling>pool.pkl
<model>_<layer>_<PCs_dataset>_<n_srp>_to_<n_pca>_srp_to_pca_components_<pooling>pool.pkl
```

MPI fitting runner:

```text
python_scripts/scripts/run_par_imagenet_srp_pca.py
```

### Project features into saved SRP/PCA spaces

`srp_pca_project_dataset_wrapper`:

1. Loads saved SRP and PCA objects fitted on `PCs_dataset`.
2. Preserves dataset order by using a non-shuffled loader for projections.
3. Extracts ANN activations batch by batch.
4. Applies `SRP.transform`, then `PCA.transform`.
5. Concatenates and saves final features as an `.npz` file with key `features`.

Canonical projection pattern:

```text
<data_path>/models/
<model>_<layer>_<dataset>_<n_srp>_to_<n_pca>_srp_to_pca_<PCs_dataset>_components_<pooling>pool.npz
```

ImageNet projection runner:

```text
python_scripts/scripts/run_par_imagenet_srp_pca_extraction_projection.py
```

Livingstone Lab stimulus projection runner:

```text
python_scripts/scripts/run_par_stimuli_srp_pca_extraction_projection.py
```

The custom-stimulus runner expects an `ImageFolder` under:

```text
<livingstone_lab>/Stimuli/<folder_name>/
```

It uses standard ImageNet resize, center crop, tensor conversion, and normalization.

### Supported model patterns

Examples actively used in scripts and notebooks:

- AlexNet through torchvision:
  - `model_name=alexnet`
  - `pkg=torchvision`
  - `img_size=224`
- DINOv3-L through Hugging Face:
  - `model_name=dino_v3_l`
  - `pkg=hf`
  - repository `facebook/dinov3-vitl16-pretrain-lvd1689m`
  - commonly `img_size=384` for image-feature pipelines and 224 in the video-autocorrelation notebook

The exact relevant layers are supplied by `useful_stuff.image_processing.computational_models.get_relevant_output_layers` or `imgANN.get_relevant_layers`.

### MPI design

The parallel runners build a task per relevant layer and call `master_workers_queue`:

- MPI rank 0 is the master and does not load the ANN/DataLoader.
- Worker ranks load the ANN and DataLoader.
- On O2/DIPSEN, worker ranks are assigned GPUs round-robin when GPUs are available.
- Locally, `get_device()` may select Apple MPS.

Because every worker creates its own model and DataLoader, memory and I/O scale with the number of workers. Choose `--ntasks`, batch size, and layer grouping conservatively.

### Current local model artifacts

As of 2026-07-14:

- 106 `.pkl` component files under `sd_local/models/imagenet_components/`.
- 38 projected `.npz` feature files directly under `sd_local/models/`.
- Artifacts exist for AlexNet and DINOv3-L.
- Projected datasets include ImageNet validation and `talia_20each_tizi`.

Some older files use legacy/inconsistent filename forms such as missing underscores before `allpool`. The current helpers define the canonical names. Before deleting or recomputing anything, check whether an apparently missing output exists under a legacy name.

## 8. Neural-data loading and image-order alignment

The reusable module is:

```text
python_scripts/src/project_specific_utils/dataloader.py
```

### Natural-image rasters

`load_img_natraster` loads:

```text
<data_path>/data/<monkey_name>_natraster<date>.mat
```

The files are MATLAB v7.3/HDF5. The function:

1. Reads `natraster`.
2. Casts to `float32`.
3. Reorders axes to neurons × time × trials.
4. Wraps the array in `TimeSeries` at 1,000 Hz.
5. Optionally selects a brain area.
6. Optionally resamples to a target frequency, commonly 100 Hz.

### Brain-area definitions

`brain_areas.yaml` currently defines channel groups for:

- `paul`
- `three0`
- `baby1`
- `og`
- `octavius`
- `friday`
- `baby5`
- `red`

Areas include combinations of V1, V2, V3, PIT, CIT, and AIT depending on the monkey/session. The file is currently untracked and should be committed or separately backed up.

The local `BrainAreas` class currently opens `../../brain_areas.yaml`, so it assumes execution from `python_scripts/scripts`. This path should eventually be made relative to `dataloader.py` or repository root.

### Aligning monkey and ANN stimulus order

`map_image_order_from_ann_to_monkey` loads:

```text
<data_path>/data/<monkey_name>_allimages<date>.mat
```

It:

1. Reads `allimages`, falling back to `uniqueImage`.
2. Decodes MATLAB character arrays.
3. Removes duplicates and sorts names.
4. Reads file basenames from a torchvision `ImageFolder`.
5. Applies the special `talia_20each_tizi` filename cleanup when needed.
6. Returns indices that reorder ANN features into monkey stimulus order.

Order preservation is scientifically critical. Do not shuffle the custom-stimulus DataLoader used for saved feature projections, and do not change filename normalization without revalidating the mapping assertion.

## 9. Representational analyses

### Static dRSA and dynamic Information Imbalance

Notebook:

```text
python_scripts/scripts/SRP-PCA_static_dRSA_and_dynII_cluster_dev.ipynb
```

Current example configuration uses:

- Monkey `three0`, date `250313`.
- Stimulus dataset `talia_20each_tizi`.
- Brain area V1.
- Neural sampling at 100 Hz.
- AlexNet SRP/PCA features.
- Correlation RDM metrics.
- Information Imbalance neighborhood size `k=50`.

The notebook loads neural rasters, aligns image order, loads projected ANN features, and saves/plots:

- Static dRSA through neural time.
- ANN-to-neural dynamic Information Imbalance.
- Neural-to-ANN dynamic Information Imbalance.

The code supports running dRSA and Information Imbalance together or separately.

### DINOv3 video autocorrelation

Notebook:

```text
python_scripts/scripts/dinov3_video_autocorrelation.ipynb
```

This active notebook:

1. Loads frames from one or more videos.
2. Extracts mean-pooled DINOv3-L layer features.
3. Computes full frame-by-frame feature autocorrelation matrices.
4. Computes lag-averaged autocorrelation decay.
5. Correlates each frame representation with the final frame.
6. Builds time-resolved RDMs across sampled videos.
7. Compares each time-point RDM with final-frame RDMs using dRSA.
8. Supports cross-layer comparisons to a selected final-frame target layer.
9. Computes autocorrelation of the RDM time series itself.

Configurable inputs include `DINOV3_VIDEO_DIR`, `DINOV3_VIDEO_PATH`, frame stride, maximum frames, number of videos, number of sampled time points, RDM metric, RSA metric, and target layer.

Default saved-output directory:

```text
<data_path>/models/dinov3_autocorr_outputs/
```

### Long image presentation

Notebooks:

- `load_long_img_presentation.ipynb`
- `long_img_presentation_RSA.ipynb`

These inspect `george_062818_data.mat` and analyze `imageRSVP`. The RSA notebook downsamples from 1,000 Hz to 100 Hz, randomly splits stimuli into two halves, builds time-resolved RDMs, and computes dynamic RSA between the splits.

### Most exciting images

Notebook:

```text
python_scripts/scripts/most_exciting_images_from_natraster.ipynb
```

This untracked notebook ranks natural images by mean neural activity in a configurable post-onset time window. The current example uses `three0`, V1, 60–200 ms, and displays the top and bottom images while preserving monkey-to-ImageFolder mapping.

## 10. Notebook inventory and maturity

### Active analysis notebooks

- `dinov3_video_autocorrelation.ipynb`
  - Active, modified, and central to the static/dynamic question.
- `SRP-PCA_static_dRSA_and_dynII_cluster_dev.ipynb`
  - Active prototype for model-neural comparisons.
- `most_exciting_images_from_natraster.ipynb`
  - Useful active analysis, but untracked.
- `long_img_presentation_RSA.ipynb`
  - Focused dynamic RSA analysis for long image presentations.

### Feature-pipeline development notebooks

- `ipca_imagenet_prototype.ipynb`
  - Early AlexNet iPCA prototype.
- `ipca_imagenet_imgANN_dev.ipynb`
  - iPCA development through `imgANN`.
- `srp_pca_imagenet_imgANN_dev.ipynb`
  - SRP/PCA fitting development.
- `srp_pca_extraction_projection_dev.ipynb`
  - SRP/PCA fitting and projection development.

Much of their reusable logic has moved into `feature_extraction.py` and executable runners. Prefer updating source modules and runners rather than maintaining duplicate notebook implementations.

### Visualization notebooks

- `PCs_visualization.ipynb`
  - Loads iPCA objects and visualizes ImageNet images at PC extremes.
- `SRP_PCs_visualization.ipynb`
  - Loads already-projected `.npz` features and visualizes PC extremes without rerunning the ANN.
- `visualize_imagenet_validation.ipynb`
  - Sanity-checks ImageNet validation loading and labels.
- `visualize_moments_in_time.ipynb`
  - Builds an HTML browser for classes and example videos from Moments in Time.

### Data inspection and stimulus-development notebooks

- `load_george_movie.ipynb`
  - Basic inspection of `george_121918_data.mat`.
- `load_long_img_presentation.ipynb`
  - Basic inspection of long-presentation data.
- `stimuli_generation_dev.ipynb`
  - Untracked experiments with camera paths, translation, and zoom over images/videos.

The notebooks contain large embedded outputs; the local Git object database is roughly 287 MB despite the source tree being small. Clear unnecessary notebook outputs before future commits when those outputs are not part of the scientific record.

## 11. Cluster execution

Slurm launchers exist for O2 and DIPSEN.

### O2

- Repository path: `/home/tic569/static_dynamic`
- Data root: `/n/data2/hms/neurobio/livingstone/tiziano/sd_o2`
- Account: `livingstone`
- GPU partition: `gpu`
- Loads GCC 14.2, CUDA 12.8, Python 3.13.1, and OpenMPI.

### DIPSEN

- Repository path: `/home/tiziano.causin/static_dynamic`
- Data root: `/mnt/storage/tier2/morwur/Projects/TIZIANO/sd_nas`
- GPU partition: `all-compute-gpu`
- Activates the repository `.venv` after sourcing `.bashrc`.

Both environments set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Therefore Hugging Face weights must already exist in the configured `HF_HOME` cache.

Typical submission pattern:

```bash
sbatch --export=ALL,\
model_name=dino_v3_l,\
pkg=hf,\
pooling=all,\
n_srp_components=10000,\
n_pca_components=1000,\
batch_size=1024,\
img_size=384 \
bash_scripts/par_imagenet_srp_pca_o2.slrm
```

Review task count, memory, wall time, and local module versions before resubmitting old scripts.

## 12. Coding conventions

The project coding contract is in:

```text
python_scripts/CODE_GENERATION_COMMANDMENTS.md
```

The most important rules are:

- Prefer readable, direct scientific Python over clever abstractions.
- Match existing naming, imports, file organization, and save-name conventions.
- Inspect both project `src/` and `useful_stuff` before adding helpers.
- Put reusable logic under `python_scripts/src/`, not in execution scripts.
- Use `config.yaml`, `MY_ENV`, and `Path(__file__)` instead of new hard-coded paths.
- Use dataclass config objects and argparse for script parameters.
- Add the established structured docstring to non-trivial functions.
- Use explicit end-of-block comments where they improve readability.
- Avoid new dependencies unless necessary.
- Preserve the current repository structure.

There are still older files that do not fully follow the newest commandments. New edits should move the code toward the contract without unnecessary broad refactors.

## 13. Known caveats and technical debt

1. **Uncommitted scientific work**
   - Important active files are modified or untracked. Protect and commit them first.

2. **No `README.md`**
   - `pyproject.toml` points to a missing README.

3. **External `useful_stuff` dependency**
   - The project will not import correctly without a compatible checkout and config path.

4. **Inconsistent `MY_ENV` defaults**
   - Several scripts use `dev`, which is absent from `config.yaml`.

5. **Working-directory-sensitive config reads**
   - Parallel runners open `../../config.yaml`.
   - The local `BrainAreas` class opens `../../brain_areas.yaml`.
   - Run these from `python_scripts/scripts` until paths are made robust.

6. **Hard-coded local defaults in video scripts**
   - The paths match the current machine but should be overridden through CLI arguments on another machine.

7. **Large notebooks and duplicated prototype logic**
   - Some notebook code duplicates source modules and may drift.

8. **Legacy artifact names**
   - Existing component files include naming variants from earlier development.

9. **No general automated test suite**
   - The two video artifact validators are the only dedicated test scripts.

10. **Brain-area indexing needs care**
    - YAML ranges look like inclusive channel endpoints, while Python slicing excludes the endpoint. Confirm the intended convention before changing neural analyses.

11. **Image ordering is fragile by design**
    - Filename changes, sorting changes, or DataLoader shuffling can invalidate ANN–neural alignment.

12. **Configuration coverage differs by environment**
    - Not every config environment defines ImageNet and Livingstone Lab roots.

13. **Cluster scripts assume cached models**
    - Offline Hugging Face mode fails if model revisions are not already cached.

## 14. Recommended next steps

### Immediate migration safety

1. Review `git diff` and all untracked files.
2. Commit the active video pipeline, tests, `brain_areas.yaml`, and intended notebooks in logical commits.
3. Push `main` or a dedicated `codex/...` branch to GitHub.
4. Back up `sd_local`, especially the 100 final dynamic/static stimuli and `rename_mapping.csv`.
5. Record or back up the exact `useful_stuff` revision.
6. Confirm the new account can access the GitHub repository; ChatGPT account migration does not automatically migrate local files or GitHub credentials.

### Code quality

1. Add a concise `README.md` based on this document.
2. Make every config and YAML path repository-relative rather than working-directory-relative.
3. Replace invalid `MY_ENV=dev` defaults with a real environment or a clear required-variable error.
4. Export `BrainAreas` consistently if it is intended as public project API.
5. Consolidate repeated MPI device-selection code into a reusable project helper only if it remains stable across runners.
6. Add focused unit tests for timestamp conversion, filename mapping, image-order mapping, and feature save paths.

### Scientific development

1. Finalize and document the DINOv3 autocorrelation outputs.
2. Decide the canonical dynamic-vs-static stimulus comparison and expected trial timing.
3. Verify that final dynamic and static files have the intended matched endpoint frame after renaming.
4. Run ANN features on the final dynamic/static stimulus set, not only ImageNet and natural-image folders.
5. Define how video features will be sampled in time and aligned to neural bins.
6. Move stable dRSA/Information Imbalance functions out of notebooks into `src/`.
7. Save analysis configs and source feature paths alongside every result for reproducibility.

## 15. Suggested prompt for the new ChatGPT Edu account

Attach or paste this file at the start of a new project conversation, then use a prompt like:

> This is the handoff for my `static_dynamic` neuroscience project. Read it completely before editing code. The repository follows `python_scripts/CODE_GENERATION_COMMANDMENTS.md`. First inspect the current Git status and relevant source files; do not overwrite uncommitted work. Continue from the “Recommended next steps” section and keep data paths configurable through `config.yaml` and `MY_ENV`.

For a specific task, add the exact objective, expected output, target environment, and whether the change should remain a notebook prototype or become reusable source code.

## 16. Quick restart checklist

```bash
cd /Users/tizianocausin/Desktop/static_dynamic

# Protect current work.
git status --short
git diff --stat

# Select paths and a writable matplotlib cache.
export MY_ENV=tiziano_mac_mini
export MPLCONFIGDIR=/tmp/matplotlib

# Restore/check the locked environment.
uv sync --frozen

# Confirm the custom dependency is importable through the configured path.
.venv/bin/python -c "import sys, yaml; from pathlib import Path; config = yaml.safe_load(Path('config.yaml').read_text()); sys.path.append(config['tiziano_mac_mini']['paths']['useful_stuff_path']); from useful_stuff.general_utils import TimeSeries; print('useful_stuff OK')"

# Validate the current final stimuli.
.venv/bin/python python_scripts/scripts/test_standardized_final_video_selection.py \
  --input_dir /Users/tizianocausin/sd_local/possible_vids/final_video_selection/yes \
  --fps 60
.venv/bin/python python_scripts/scripts/test_padded_60fps_image_videos.py
```

If these checks pass and the working tree has been safely backed up, development can continue from the current local state.
