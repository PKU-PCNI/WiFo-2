"""Evaluation-only runner for WiFo-2 channel reconstruction tasks."""

from contextlib import nullcontext

import torch


MASK_CONFIGS = {
    "random": (0.85,),
    "temporal": (0.25,),
    "fre": (0.25,),
    "CE": ([4, 1, 12],),
}


class InferenceRunner:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args

    def _tasks_for(self, dataset_name):
        if "CE" in dataset_name:
            return ("CE",)
        if self.args.mask_strategy_random == "none":
            return (self.args.mask_strategy,)
        if "Argos" in dataset_name or "Dich" in dataset_name:
            return ("random", "fre")
        return ("random", "temporal", "fre")

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    @torch.inference_mode()
    def _evaluate_one(self, loader, dataset_name, mask_strategy, mask_ratio):
        squared_error = 0.0
        sample_count = 0

        for batch in loader:
            batch = batch.to(self.device, non_blocking=self.args.pin_memory)
            with self._autocast_context():
                _, _, prediction, target, mask = self.model(
                    batch,
                    mask_ratio=mask_ratio,
                    mask_strategy=mask_strategy,
                    seed=self.args.seed,
                    data=dataset_name,
                )

            batch_size, _, _, last_dim = prediction.shape
            prediction = prediction[mask == 1].reshape(
                batch_size, -1, 2, last_dim
            ).permute(0, 1, 3, 2).reshape(batch_size, -1, 2)
            target = target[mask == 1].reshape(
                batch_size, -1, 2, last_dim
            ).permute(0, 1, 3, 2).reshape(batch_size, -1, 2)

            numerator = torch.mean(torch.sum((target - prediction) ** 2, dim=2), dim=1)
            denominator = torch.mean(torch.sum(target ** 2, dim=2), dim=1)
            squared_error += torch.sum(numerator / denominator.clamp_min(1e-12)).item()
            sample_count += batch_size

        if sample_count == 0:
            raise ValueError(f"The test split for {dataset_name} is empty.")
        return squared_error / sample_count

    def run(self, test_loaders):
        self.model.eval()
        results = {}
        for dataset_name, loader in test_loaders.items():
            results[dataset_name] = {}
            for strategy in self._tasks_for(dataset_name):
                if strategy not in MASK_CONFIGS:
                    raise ValueError(f"Unsupported inference strategy: {strategy}")
                results[dataset_name][strategy] = {}
                for ratio in MASK_CONFIGS[strategy]:
                    nmse = self._evaluate_one(loader, dataset_name, strategy, ratio)
                    ratio_label = tuple(ratio) if isinstance(ratio, list) else ratio
                    results[dataset_name][strategy][ratio_label] = nmse
                    nmse_db = 10.0 * torch.log10(torch.tensor(nmse)).item()
                    print(
                        f"{dataset_name} | {strategy} | mask={ratio_label} | "
                        f"NMSE={nmse:.8f} ({nmse_db:.4f} dB)"
                    )
        return results
