import argparse
import os
from pathlib import Path

import numpy as np
import tifffile
import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet, UNet
from monai.transforms import Activations, AsDiscrete, Compose
from scipy import ndimage as ndi
from skimage import measure
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "config.yml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run binary hair-cell segmentation, optionally followed by watershed instance labeling."
    )
    parser.add_argument("image", help="Path to a single 2D TIFF image.")
    parser.add_argument("--output", help="Output TIFF path.")
    parser.add_argument(
        "--output-type",
        choices=("binary", "labels", "both"),
        default=None,
        help="Write a binary mask, watershed instance labels, or both. Defaults to config inference.output_type.",
    )
    parser.add_argument(
        "--labels-output",
        help="Instance-label output path when --output-type both is used.",
    )
    parser.add_argument(
        "--config",
        default=os.fspath(DEFAULT_CONFIG_PATH),
        help="Path to config.yml.",
    )
    parser.add_argument(
        "--model",
        help="Path to model checkpoint. Defaults to model.best_model_path from config.yml.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default=None,
        help="Inference device. Defaults to config inference.device.",
    )
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--min-distance", type=int, default=None)
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(path, base_dir):
    path = Path(path)
    if path.is_absolute():
        return path

    candidate = base_dir / path
    if candidate.exists():
        return candidate

    basename_candidate = base_dir / path.name
    if basename_candidate.exists():
        return basename_candidate

    return candidate


def select_device(device_name):
    if device_name in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(device_name)


def build_model(model_config):
    model_kwargs = {
        "spatial_dims": model_config["spatial_dims"],
        "in_channels": model_config["in_channels"],
        "out_channels": model_config["out_channels"],
        "channels": tuple(model_config["channels"]),
        "strides": tuple(model_config["strides"]),
    }

    architecture = model_config["architecture"]
    if architecture == "unet":
        return UNet(**model_kwargs, num_res_units=model_config["num_res_units"])
    if architecture == "attention_unet":
        return AttentionUnet(**model_kwargs)
    raise ValueError(f"Unknown model architecture: {architecture}")


def load_model(config, config_dir, model_path, device):
    model_config = config["model"]
    model = build_model(model_config)

    configured_path = model_path or model_config.get("best_model_path")
    if not configured_path:
        configured_path = "best_model.pth"
    checkpoint_path = resolve_path(configured_path, config_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint_path


def load_image(image_path):
    image = tifffile.imread(image_path).squeeze().astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected one 2D image, got shape {image.shape}")
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    return image


def predict_binary_mask(model, image, config, device):
    validation_config = config["validation"]
    image_tensor = torch.from_numpy(image)[None, None].to(device)
    prediction_transform = Compose([
        Activations(sigmoid=validation_config["prediction_sigmoid"]),
        AsDiscrete(threshold=validation_config["prediction_threshold"]),
    ])

    with torch.no_grad():
        logits = sliding_window_inference(
            image_tensor,
            roi_size=tuple(validation_config["roi_size"]),
            sw_batch_size=validation_config["sw_batch_size"],
            predictor=model,
        )
        prediction = prediction_transform(logits)

    return prediction[0, 0].cpu().numpy().astype(np.uint8)


def binary_to_instance_labels(binary_mask, min_size=20, min_distance=5):
    mask = binary_mask > 0
    distance = ndi.distance_transform_edt(mask)

    coords = peak_local_max(
        distance,
        labels=mask,
        min_distance=min_distance,
        exclude_border=False,
    )
    if len(coords) == 0:
        return np.zeros_like(binary_mask, dtype=np.uint32)

    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    labels = watershed(-distance, markers, mask=mask)
    if min_size > 0:
        labels = remove_small_objects(labels, min_size=min_size)

    labels = measure.label(labels)
    props = measure.regionprops(labels)
    sorted_props = sorted(props, key=lambda prop: prop.centroid[1])

    relabeled = np.zeros_like(labels, dtype=np.uint32)
    for new_label, prop in enumerate(sorted_props, start=1):
        relabeled[labels == prop.label] = new_label
    return relabeled


def default_output_path(image_path, suffix):
    image_path = Path(image_path)
    if image_path.suffix.lower() in (".tif", ".tiff"):
        return image_path.with_name(f"{image_path.stem}_{suffix}{image_path.suffix}")
    return image_path.with_name(f"{image_path.name}_{suffix}.tif")


def write_outputs(binary_mask, args, config):
    inference_config = config.get("inference", {})
    postprocessing_config = config.get("postprocessing", {})

    output_type = args.output_type or inference_config.get("output_type", "binary")
    min_size = args.min_size
    if min_size is None:
        min_size = postprocessing_config.get("min_size", 20)
    min_distance = args.min_distance
    if min_distance is None:
        min_distance = postprocessing_config.get("min_distance", 5)

    if output_type == "binary":
        output = Path(args.output) if args.output else default_output_path(args.image, "prediction")
        tifffile.imwrite(output, binary_mask.astype(np.uint8))
        return [output]

    labels = binary_to_instance_labels(
        binary_mask,
        min_size=min_size,
        min_distance=min_distance,
    )

    if output_type == "labels":
        output = Path(args.output) if args.output else default_output_path(args.image, "labels")
        tifffile.imwrite(output, labels.astype(np.uint32))
        return [output]

    binary_output = Path(args.output) if args.output else default_output_path(args.image, "prediction")
    labels_output = (
        Path(args.labels_output)
        if args.labels_output
        else default_output_path(args.image, "labels")
    )
    tifffile.imwrite(binary_output, binary_mask.astype(np.uint8))
    tifffile.imwrite(labels_output, labels.astype(np.uint32))
    return [binary_output, labels_output]


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config_dir = config_path.parent

    inference_config = config.get("inference", {})
    device = select_device(args.device or inference_config.get("device", "auto"))
    model, checkpoint_path = load_model(config, config_dir, args.model, device)

    image = load_image(args.image)
    binary_mask = predict_binary_mask(model, image, config, device)
    outputs = write_outputs(binary_mask, args, config)

    print(f"Device: {device}")
    print(f"Loaded model: {checkpoint_path}")
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
