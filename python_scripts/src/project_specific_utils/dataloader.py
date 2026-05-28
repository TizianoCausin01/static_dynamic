
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet


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
