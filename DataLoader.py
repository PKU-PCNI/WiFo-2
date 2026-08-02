"""Test-set-only data loading for WiFo-2 inference."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class ChannelDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        return self.samples[index]


def _extract_test_tensor(payload, path):
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, (tuple, list)) and len(payload) >= 3:
        return payload[2]
    if isinstance(payload, dict):
        for key in ("test", "X_test", "test_data"):
            if key in payload:
                return payload[key]
    raise ValueError(
        f"Unsupported dataset format in {path}. Expected a test tensor, a "
        "(train, validation, test) tuple, or a dict containing 'test'/'X_test'."
    )


def _safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_test_file(data_root, dataset_name):
    dataset_directory = Path(data_root) / dataset_name
    candidates = (
        dataset_directory / "X_test.pt",
        dataset_directory / "data.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = " or ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Dataset not found. Expected {expected}. Download or generate X_test.pt "
        "and place it in the dataset directory."
    )


def load_test_data(args):
    """Load only the zero-shot test splits requested by ``args.dataset``."""
    loaders = {}
    sequence_length = None

    for dataset_name in args.dataset.split("*"):
        dataset_name = dataset_name.strip()
        if not dataset_name:
            continue
        path = _find_test_file(args.data_root, dataset_name)

        samples = _extract_test_tensor(_safe_torch_load(path), path)
        if not torch.is_tensor(samples) or samples.ndim < 3:
            raise ValueError(f"Invalid test tensor in {path}: expected at least 3 dimensions.")

        current_length = samples.shape[2]
        if sequence_length is None:
            sequence_length = current_length
        elif current_length != sequence_length:
            raise ValueError("All datasets in one run must have the same sequence length.")

        loaders[dataset_name] = DataLoader(
            ChannelDataset(samples),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

    if not loaders:
        raise ValueError("No dataset name was supplied.")
    args.seq_len = sequence_length
    return loaders
