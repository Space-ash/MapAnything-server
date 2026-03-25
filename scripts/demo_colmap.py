# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Demo script to export MapAnything outputs in COLMAP format.

This script runs MapAnything inference on images and exports the results
to COLMAP format with scene-adaptive voxel downsampling.

Usage:
    python scripts/demo_colmap.py
    python scripts/demo_colmap.py --config configs/demo_colmap_config.yaml

Output structure:
    output_dir/
        images/           # Processed images (model input resolution)
            img1.jpg
            img2.jpg
            ...
        sparse/
            cameras.bin
            images.bin
            points3D.bin
            points.ply
"""

import argparse
import glob
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import yaml

from mapanything.models import MapAnything
from mapanything.utils.colmap_export import export_predictions_to_colmap
from mapanything.utils.image import load_images
from mapanything.utils.misc import seed_everything
from mapanything.utils.viz import predictions_to_glb

# Configure CUDA settings (only if available)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "demo_colmap_config.yaml")
DEFAULT_CONFIG = {
    "paths": {
        "images_dir": None,
        "output_dir": None,
    },
    "runtime": {
        "seed": 42,
        "device": "auto",
    },
    "model": {
        "use_apache": False,
        "local_model_path": "./weights/map-anything",
        "hf_model_name": None,
    },
    "inference": {
        "memory_efficient_inference": True,
        "minibatch_size": 1,
        "use_amp": True,
        "amp_dtype": "bf16",
        "apply_mask": True,
        "mask_edges": True,
    },
    "export": {
        "voxel_fraction": 0.01,
        "voxel_size": None,
        "save_glb": False,
        "skip_point2d": False,
    },
}


def _deep_update(base, updates):
    merged = dict(base)
    for key, value in updates.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_project_path(path_value):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_value))


def _resolve_device(device_name):
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def load_config(config_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    if not isinstance(user_config, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")

    return _deep_update(DEFAULT_CONFIG, user_config)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MapAnything COLMAP Export Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="YAML config file path",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Override config: directory containing input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override config: directory to save COLMAP output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config: random seed for reproducibility",
    )
    parser.add_argument(
        "--voxel_fraction",
        type=float,
        default=None,
        help="Override config: adaptive voxel fraction",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Override config: explicit voxel size in meters",
    )
    parser.add_argument(
        "--apache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config: use Apache 2.0 licensed model",
    )
    parser.add_argument(
        "--save_glb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config: also save dense reconstruction as GLB file",
    )
    parser.add_argument(
        "--skip_point2d",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config: skip Point2D backprojection for faster export",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Override config: local pretrained model directory",
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        default=None,
        help="Override config: Hugging Face model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override config: device to use (auto/cpu/cuda)",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=None,
        help="Override config: inference minibatch size",
    )
    parser.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config: enable AMP during inference",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default=None,
        help="Override config: AMP dtype (bf16/fp16)",
    )
    parser.add_argument(
        "--apply_mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config: apply model mask during inference",
    )
    parser.add_argument(
        "--mask_edges",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config: mask edges during inference",
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config: enable memory-efficient inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = (
        args.config
        if os.path.isabs(args.config)
        else os.path.abspath(args.config)
    )
    config = load_config(config_path)

    if args.images_dir is not None:
        config["paths"]["images_dir"] = args.images_dir
    if args.output_dir is not None:
        config["paths"]["output_dir"] = args.output_dir
    if args.seed is not None:
        config["runtime"]["seed"] = args.seed
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.apache is not None:
        config["model"]["use_apache"] = args.apache
    if args.local_model_path is not None:
        config["model"]["local_model_path"] = args.local_model_path
    if args.hf_model_name is not None:
        config["model"]["hf_model_name"] = args.hf_model_name
    if args.minibatch_size is not None:
        config["inference"]["minibatch_size"] = args.minibatch_size
    if args.use_amp is not None:
        config["inference"]["use_amp"] = args.use_amp
    if args.amp_dtype is not None:
        config["inference"]["amp_dtype"] = args.amp_dtype
    if args.apply_mask is not None:
        config["inference"]["apply_mask"] = args.apply_mask
    if args.mask_edges is not None:
        config["inference"]["mask_edges"] = args.mask_edges
    if args.memory_efficient_inference is not None:
        config["inference"]["memory_efficient_inference"] = (
            args.memory_efficient_inference
        )
    if args.voxel_fraction is not None:
        config["export"]["voxel_fraction"] = args.voxel_fraction
    if args.voxel_size is not None:
        config["export"]["voxel_size"] = args.voxel_size
    if args.save_glb is not None:
        config["export"]["save_glb"] = args.save_glb
    if args.skip_point2d is not None:
        config["export"]["skip_point2d"] = args.skip_point2d

    images_dir = _resolve_project_path(config["paths"]["images_dir"])
    output_dir = _resolve_project_path(config["paths"]["output_dir"])
    model_path = _resolve_project_path(config["model"]["local_model_path"])
    seed = config["runtime"]["seed"]
    device = _resolve_device(config["runtime"]["device"])
    inference_cfg = config["inference"]
    export_cfg = config["export"]

    if images_dir is None:
        raise ValueError("Please set paths.images_dir in the YAML config or via --images_dir.")
    if output_dir is None:
        raise ValueError("Please set paths.output_dir in the YAML config or via --output_dir.")

    # Print configuration
    print("=" * 60)
    print("MapAnything COLMAP Export")
    print("=" * 60)
    print(f"Config file: {config_path}")
    print(f"Input images: {images_dir}")
    print(f"Output directory: {output_dir}")
    if export_cfg["voxel_size"] is not None:
        print(f"Voxel size: {export_cfg['voxel_size']}m (explicit)")
    else:
        print(f"Voxel fraction: {export_cfg['voxel_fraction']} (adaptive)")
    print(f"Random seed: {seed}")

    # Set seed for reproducibility
    seed_everything(seed)
    print(f"Using device: {device}")

    # Initialize model
    if model_path is not None and os.path.isdir(model_path):
        print(f"Loading local MapAnything model from: {model_path}")
        model = MapAnything.from_pretrained(model_path).to(device)
    else:
        if config["model"]["hf_model_name"]:
            model_name = config["model"]["hf_model_name"]
        elif config["model"]["use_apache"]:
            model_name = "facebook/map-anything-apache"
        else:
            model_name = "facebook/map-anything"

        print(f"Loading MapAnything model from Hugging Face: {model_name}")
        model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()
    print("Model loaded successfully!")

    # Get image paths
    if not os.path.isdir(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_path_list = []
    for ext in image_extensions:
        image_path_list.extend(glob.glob(os.path.join(images_dir, ext)))
    image_path_list = sorted(image_path_list)

    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {images_dir}")

    print(f"Found {len(image_path_list)} images")

    # Get image names for COLMAP output
    image_names = [os.path.basename(path) for path in image_path_list]

    # Load and preprocess images
    print("Loading images...")
    views = load_images(image_path_list)
    print(f"Loaded {len(views)} views")

    # Run inference with memory-efficient defaults
    print("Running MapAnything inference...")
    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=inference_cfg["memory_efficient_inference"],
            minibatch_size=inference_cfg["minibatch_size"],
            use_amp=inference_cfg["use_amp"],
            amp_dtype=inference_cfg["amp_dtype"],
            apply_mask=inference_cfg["apply_mask"],
            mask_edges=inference_cfg["mask_edges"],
        )
    print("Inference complete!")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Export to COLMAP format (includes saving processed images)
    print("Exporting to COLMAP format...")
    _ = export_predictions_to_colmap(
        outputs=outputs,
        processed_views=views,
        image_names=image_names,
        output_dir=output_dir,
        voxel_fraction=export_cfg["voxel_fraction"],
        voxel_size=export_cfg["voxel_size"],
        data_norm_type=model.encoder.data_norm_type,
        save_ply=True,
        save_images=True,
        skip_point2d=export_cfg["skip_point2d"],
    )

    print(f"COLMAP reconstruction saved to: {output_dir}")

    # Export GLB if requested
    if export_cfg["save_glb"]:
        glb_output_path = os.path.join(output_dir, "dense_mesh.glb")
        print(f"Saving GLB file to: {glb_output_path}")

        # Collect data for GLB export
        world_points_list = []
        images_list = []
        masks_list = []

        for pred in outputs:
            pts3d = pred["pts3d"][0].cpu().numpy()
            mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
            valid_depth = pts3d[..., 2] > 0
            combined_mask = mask & valid_depth

            world_points_list.append(pts3d)
            images_list.append(pred["img_no_norm"][0].cpu().numpy())
            masks_list.append(combined_mask)

        # Stack all views
        world_points = np.stack(world_points_list, axis=0)
        images = np.stack(images_list, axis=0)
        final_masks = np.stack(masks_list, axis=0)

        # Create predictions dict for GLB export
        predictions = {
            "world_points": world_points,
            "images": images,
            "final_masks": final_masks,
        }

        # Convert to GLB scene
        scene_3d = predictions_to_glb(predictions, as_mesh=True)
        scene_3d.export(glb_output_path)
        print(f"Successfully saved GLB file: {glb_output_path}")

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
