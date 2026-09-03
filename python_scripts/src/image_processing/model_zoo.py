"""
Registry of the computational models compared against the static/dynamic
recordings. Every entry fixes the model source, the input geometry, and the
ordered list of hooked layers, so extraction and analysis read one definition.

Layer choice follows a single rule per model family: one readout per
architectural block, always taken at the same point inside the block.
Transformers are read at the output projection of the block MLP
(mlp.fc2 / mlp.down_proj / output.dense), the plain CNN (AlexNet) at every
ReLU, and ConvNeXt at every block output. No residual/pooling mixtures.
"""

from dataclasses import dataclass, field


"""
stagewise_layers
Builds one layer name per block of a staged backbone in true depth order.

INPUT:
    - template: str -> format string using {stage} and {block}
    - depths: tuple[int, ...] -> number of blocks in each stage

OUTPUT:
    - layer_names: list[str] -> ordered layer names from shallow to deep
"""
def stagewise_layers(template: str, depths: tuple[int, ...]) -> list[str]:
    return [
        template.format(stage=stage_index, block=block_index)
        for stage_index, stage_depth in enumerate(depths)
        for block_index in range(stage_depth)
    ]
# EOF


"""
blockwise_layers
Builds one layer name per block of a flat transformer in true depth order.

INPUT:
    - template: str -> format string using {block}
    - num_blocks: int -> number of transformer blocks

OUTPUT:
    - layer_names: list[str] -> ordered layer names from shallow to deep
"""
def blockwise_layers(template: str, num_blocks: int) -> list[str]:
    return [template.format(block=block_index) for block_index in range(num_blocks)]
# EOF


@dataclass
class ModelSpec:
    model_name: str
    repo_url: str
    layers: list[str]
    # "image" runs one forward call per frame, "video" runs a sliding window.
    modality: str
    family: str
    label: str
    pkg: str = "hf"
    img_size: int = 224
    pooling: str | None = "mean"
    batch_size: int = 16
    attn_implementation: str | None = "eager"
    # Sliding-window settings, ignored by the per-frame image models.
    architecture: str = "transformer"
    window_size_frames: int = 16
    model_input_name: str = "pixel_values"
    skip_predictor: bool = False
    preprocess_chunk_size: int = 64
    # Attribute of the loaded model to hook instead of the model itself, for
    # checkpoints that ship several towers (SigLIP's vision plus text).
    submodule: str | None = None
    # Channel statistics for the torchvision video path; None keeps ImageNet.
    normalization_mean: tuple[float, ...] | None = None
    normalization_std: tuple[float, ...] | None = None
    extra: dict = field(default_factory=dict)
# EOF


MODEL_ZOO = {
    # --- image CNNs -------------------------------------------------------
    "alexnet": ModelSpec(
        model_name="alexnet",
        repo_url="alexnet",
        pkg="torchvision",
        # Every ReLU of the network, convolutional stack then classifier.
        layers=[
            "features.1", "features.4", "features.7", "features.9",
            "features.11", "classifier.2", "classifier.5",
        ],
        modality="image",
        family="image CNN",
        label="AlexNet",
        batch_size=64,
        attn_implementation=None,
    ),
    "convnext_base": ModelSpec(
        model_name="convnext_base",
        repo_url="facebook/convnext-base-224",
        # drop_path is an identity at inference and carries the block's own
        # pre-residual output in NCHW, which is ConvNeXt's exact analogue of a
        # transformer block's MLP output projection. The post-residual block
        # output was tried first and is dominated by a few outlier channels.
        layers=stagewise_layers(
            "encoder.stages.{stage}.layers.{block}.drop_path", (3, 3, 27, 3),
        ),
        modality="image",
        family="image CNN",
        label="ConvNeXt-B",
        batch_size=32,
    ),

    # --- image transformers ----------------------------------------------
    "swin_base": ModelSpec(
        model_name="swin_base",
        repo_url="microsoft/swin-base-patch4-window7-224",
        layers=stagewise_layers(
            "encoder.layers.{stage}.blocks.{block}.output.dense", (2, 2, 18, 2),
        ),
        modality="image",
        family="image ViT",
        label="Swin-B",
        batch_size=32,
    ),
    "hiera_base": ModelSpec(
        model_name="hiera_base",
        repo_url="facebook/hiera-base-224-hf",
        layers=stagewise_layers(
            "encoder.stages.{stage}.layers.{block}.mlp.fc2", (2, 3, 16, 3),
        ),
        modality="image",
        family="image ViT",
        label="Hiera-B",
        batch_size=32,
    ),
    "siglip2_so400m": ModelSpec(
        model_name="siglip2_so400m",
        repo_url="google/siglip2-so400m-patch14-224",
        # The checkpoint holds a vision and a text tower; only the vision
        # transformer is hooked, so its own layer paths lose the tower prefix.
        submodule="vision_model",
        layers=blockwise_layers("encoder.layers.{block}.mlp.fc2", 27),
        modality="image",
        family="image ViT (VLM)",
        label="SigLIP2-so400m",
        batch_size=16,
    ),
    "dino_v3_h": ModelSpec(
        model_name="dino_v3_h",
        repo_url="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        layers=blockwise_layers("layer.{block}.mlp.down_proj", 32),
        modality="image",
        family="image ViT (SSL)",
        label="DINOv3-H+",
        batch_size=16,
    ),
    "ijepa_vith14_1k": ModelSpec(
        model_name="ijepa_vith14_1k",
        repo_url="facebook/ijepa_vith14_1k",
        layers=blockwise_layers("encoder.layer.{block}.output.dense", 32),
        modality="image",
        family="image ViT (SSL)",
        label="I-JEPA-H",
        batch_size=16,
    ),

    # --- video models -----------------------------------------------------
    "vjepa2_vitl_fpc64_256": ModelSpec(
        model_name="vjepa2_vitl_fpc64_256",
        repo_url="facebook/vjepa2-vitl-fpc64-256",
        layers=blockwise_layers("encoder.layer.{block}.mlp.fc2", 24),
        modality="video",
        family="video world model",
        label="V-JEPA2-L",
        img_size=256,
        window_size_frames=16,
        model_input_name="pixel_values_videos",
        skip_predictor=True,
        attn_implementation="sdpa",
        batch_size=2,
    ),
    "videomae_base": ModelSpec(
        model_name="videomae_base",
        repo_url="MCG-NJU/videomae-base-finetuned-kinetics",
        layers=blockwise_layers("encoder.layer.{block}.output.dense", 12),
        modality="video",
        family="video transformer",
        label="VideoMAE-B",
        window_size_frames=16,
        batch_size=8,
    ),
    "videomae_base_ssv2": ModelSpec(
        model_name="videomae_base_ssv2",
        repo_url="MCG-NJU/videomae-base-finetuned-ssv2",
        layers=blockwise_layers("encoder.layer.{block}.output.dense", 12),
        modality="video",
        family="video transformer",
        label="VideoMAE-B (SSv2)",
        window_size_frames=16,
        batch_size=8,
    ),
    "r3d_18": ModelSpec(
        model_name="r3d_18",
        repo_url="r3d_18",
        pkg="torchvision",
        # Every residual block of the 3D ResNet, stem excluded.
        layers=[
            f"layer{stage_index}.{block_index}"
            for stage_index in range(1, 5)
            for block_index in range(2)
        ],
        modality="video",
        family="video CNN",
        label="R3D-18",
        architecture="cnn",
        img_size=112,
        window_size_frames=16,
        model_input_name="x",
        attn_implementation=None,
        batch_size=8,
        normalization_mean=(0.43216, 0.394666, 0.37645),
        normalization_std=(0.22803, 0.22145, 0.216989),
    ),
    # --- low-level baselines ----------------------------------------------
    # Extracted by run_pixel_value_extraction.py / run_optical_flow_extraction.py
    # rather than by the model sweep; listed here so the RSA analysis can use
    # them as reference levels with no layer hierarchy.
    "pixel_values_rgb_step30": ModelSpec(
        model_name="pixel_values_rgb_step30",
        repo_url="pixel_values_rgb_step30",
        pkg="none",
        layers=["pixels"],
        modality="baseline",
        family="low-level baseline",
        label="RGB pixels",
        pooling=None,
        attn_implementation=None,
    ),
    "optical_flow_farneback": ModelSpec(
        model_name="optical_flow_farneback",
        repo_url="optical_flow_farneback",
        pkg="none",
        layers=["flow"],
        modality="baseline",
        family="low-level baseline",
        label="Optical flow (Farneback)",
        pooling=None,
        attn_implementation=None,
    ),
}


"""
get_model_spec
Returns one registry entry and fails loudly on an unknown model name.

INPUT:
    - model_name: str -> registry key

OUTPUT:
    - spec: ModelSpec -> complete model definition
"""
def get_model_spec(model_name: str) -> ModelSpec:
    if model_name not in MODEL_ZOO:
        raise KeyError(
            f"{model_name!r} is not in the model zoo. "
            f"Available: {sorted(MODEL_ZOO)}"
        )
    # end if unknown model
    return MODEL_ZOO[model_name]
# EOF
