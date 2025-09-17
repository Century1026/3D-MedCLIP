"""Command line entry point for self-supervised MRI pretraining.

This script wires together the :class:`SelfSupervisedPretrainer` with a
command line interface.  Example usage::

    python examples/pretrain_mri.py \
        --data-source /path/to/brains/ \
        --output-dir ./outputs/mae \
        --mode mae

The ``--data-source`` argument can either be a directory that contains NIfTI
files (``.nii``/``.nii.gz``) or a text file with one absolute file path per
line.  The script saves both the full model checkpoint and the encoder weights
required for downstream fine-tuning.

To fine-tune on a downstream task, instantiate the same encoder architecture
and load the saved ``encoder.pt`` weights via ``load_state_dict`` before adding
your task-specific prediction head.
"""

from __future__ import annotations

import argparse
from medclip.mri_self_supervised import SelfSupervisedConfig, SelfSupervisedPretrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-supervised pretraining for 3D brain MRI volumes."
    )
    parser.add_argument(
        "--data-source",
        required=True,
        help="Directory of NIfTI files or a text file listing absolute paths.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where checkpoints and encoder weights will be stored.",
    )
    parser.add_argument(
        "--mode",
        choices=["mae", "contrastive"],
        default="mae",
        help="Pretraining objective to use.",
    )
    parser.add_argument(
        "--volume-size",
        type=int,
        default=(96, 96, 96),
        nargs=3,
        metavar=("D", "H", "W"),
        help="Spatial size that each volume will be resized to (depth, height, width).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=(16, 16, 16),
        nargs=3,
        metavar=("PD", "PH", "PW"),
        help="Patch size used by the transformer encoder (must divide volume-size).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay coefficient.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=(0.5, 0.7),
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Range of voxel masking ratios sampled each iteration (MAE only).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Softmax temperature used by the contrastive objective.",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=128,
        help="Projection head output dimension for contrastive learning.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Logging frequency in training steps.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for the dataloader.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SelfSupervisedConfig(
        data_source=args.data_source,
        output_dir=args.output_dir,
        mode=args.mode,
        volume_size=tuple(args.volume_size),
        patch_size=tuple(args.patch_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        mask_ratio_range=tuple(args.mask_ratio),
        temperature=args.temperature,
        projection_dim=args.projection_dim,
        log_every=args.log_every,
        num_workers=args.num_workers,
    )
    trainer = SelfSupervisedPretrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

