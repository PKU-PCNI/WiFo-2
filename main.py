"""Command-line entry point for WiFo-2 inference."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from DataLoader import load_test_data
from inference import InferenceRunner
from model import WiFo2_model


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def create_argparser():
    parser = argparse.ArgumentParser(description="Run WiFo-2 zero-shot inference.")
    parser.add_argument("--device_id", default="0")
    parser.add_argument("--task", default="recovery", choices=("recovery",))
    parser.add_argument("--size", default="tinypro")
    parser.add_argument("--MoE", type=str2bool, default=False)
    parser.add_argument("--mask_strategy", default="random", choices=("random", "temporal", "fre", "CE"))
    parser.add_argument("--mask_strategy_random", default="batch", choices=("none", "batch"))
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--dataset", default="D17")
    parser.add_argument("--data_root", default="./dataset")
    parser.add_argument("--file_load_path", required=True)
    parser.add_argument("--few_ratio", type=float, default=0.0)
    parser.add_argument("--t_patch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--pos_emb", default="SinCos_3D")
    parser.add_argument("--no_qkv_bias", type=int, default=0)
    parser.add_argument("--his_len", type=int, default=6)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=str2bool, default=False)
    return parser


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_id):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    print("CUDA is unavailable; using CPU.")
    return torch.device("cpu")


def resolve_checkpoint(path_value):
    path = Path(path_value)
    if path.suffix not in (".pkl", ".pt", ".pth"):
        path = Path(f"{path}.pkl")
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Download the released model and "
            "pass its path with --file_load_path."
        )
    return path


def safe_load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    args = create_argparser().parse_args()
    if args.few_ratio != 0.0:
        raise ValueError("This release is inference-only; --few_ratio must be 0.0.")

    setup_seed(args.seed)
    test_loaders = load_test_data(args)
    device = resolve_device(args.device_id)
    model = WiFo2_model(args=args).to(device)

    checkpoint = resolve_checkpoint(args.file_load_path)
    state_dict = safe_load_state_dict(checkpoint, device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"Missing checkpoint keys: {incompatible.missing_keys}")
        print(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    InferenceRunner(model=model, device=device, args=args).run(test_loaders)


if __name__ == "__main__":
    main()
