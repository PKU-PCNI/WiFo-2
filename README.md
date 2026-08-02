# WiFo-2

Inference code for WiFo-2. This release evaluates released WiFo-2 checkpoints
on zero-shot wireless channel tasks; it does not include model training code.

[📄 Paper](https://arxiv.org/abs/2511.22222) ·
[🤗 Model](https://huggingface.co/pku-pcni-lab/WiFo-2) ·
[🤗 Dataset](https://huggingface.co/datasets/pku-pcni-lab/WiFo-2)

> [!IMPORTANT]
> WiFo-2 is being open-sourced in stages. We are starting with inference code,
> a lightweight model checkpoint, and zero-shot evaluation data, and will
> progressively release additional models, code, and supporting resources. Our
> final goal is the complete open-source release of WiFo-2.

## 🗓️ Release roadmap

| Release date | Status | Open-source content |
| --- | --- | --- |
| August 2, 2026 | ✅ Released | Inference code, Tiny-Dense model checkpoint, and test splits of the released zero-shot datasets |
| September 2026 | 🕒 Expected | Small-Dense, Little-Dense, and Base-Dense model checkpoints |
| To be announced | 📋 Planned | Remaining WiFo-2 code, checkpoints, and supporting resources toward the complete open-source release |

## ✨ Supported tasks

- Frequency-domain channel prediction (`fre`)
- Time-domain channel prediction (`temporal`)
- Channel estimation (`CE`)

The Tiny-Dense checkpoint and selected zero-shot test subsets are hosted on
Hugging Face. No model weights or datasets are tracked by Git.

## 📦 Released assets

- [Tiny-Dense checkpoint](https://huggingface.co/pku-pcni-lab/WiFo-2/blob/main/Tiny-dense/model_best.pkl)
- [Zero-shot test datasets](https://huggingface.co/datasets/pku-pcni-lab/WiFo-2/tree/main)

Download the checkpoint and datasets with the Hugging Face CLI:

```bash
hf download pku-pcni-lab/WiFo-2 Tiny-dense/model_best.pkl --local-dir ./experiments
hf download pku-pcni-lab/WiFo-2 --repo-type dataset --local-dir ./dataset
```

## 🛠️ Setup

Python 3.10 or later is recommended. Install the dependencies in your own
environment:

```bash
pip install -r requirements.txt
```

Put downloaded files in the following layout:

```text
WiFo-2/
├── dataset/
│   ├── D17/X_test.pt
│   └── D17CE/X_test.pt
└── experiments/
    └── Tiny-dense/model_best.pkl
```

## 🚀 Inference

Run the released Tiny-Dense model on all supported masks for `D17`:

```bash
python main.py --device_id 0 --task recovery --size tinypro --MoE False --mask_strategy_random batch --dataset D17 --data_root ./dataset --file_load_path ./experiments/Tiny-dense/model_best.pkl --few_ratio 0.0 --t_patch_size 4 --patch_size 4 --batch_size 16 --pos_emb SinCos_3D
```

For a non-CE dataset, `--mask_strategy_random batch` evaluates the published
random recovery, time-domain prediction, and frequency-domain prediction masks.
To run one task only, use (for example):

```bash
python main.py [other arguments] --mask_strategy_random none --mask_strategy fre --mask_ratio 0.25
```

Channel-estimation datasets are identified by `CE` in the directory name and
use the released CE mask configuration:

```bash
python main.py [other arguments] --dataset D17CE
```

The default model selection is Tiny-Dense (`--size tinypro --MoE False`). Match
`--size`, `--MoE`, `--t_patch_size`, `--patch_size`, and `--pos_emb` to the
checkpoint being evaluated.

## 🗂️ Dataset name mapping

The directory names in the released files differ from the names used in the
paper. Use the following mapping when reproducing the reported results.

| Released directory | Name in the paper |
| --- | --- |
| `D17` | `QC17` |
| `D18` | `QC18` |
| `D19` | `QC19` |
| `Argos_43` | `AC6` |
| `Argos_56` | `AC7` |
| `RM_070703` | `RMC8` |
| `RM_070704` | `RMC9` |
| `O1drone` | `DEC3` |
| `SionnaRT15` | `RTC3` |
| `D17CE` | `QF17` |
| `D18CE` | `QF18` |
| `D19CE` | `QF19` |
| `RM_070703_CE` | `RMF8` |
| `RM_070704_CE` | `RMF9` |
| `O1droneCE1` | `DEF3` |

## 📝 Citation

If you find this repository useful, please cite:

```bibtex
@article{liu2025wifo2,
  title   = {{WiFo-2}: A Generalist Foundation Model Unifies Heterogeneous Wireless System Design},
  author  = {Liu, Boxun and Liu, Xuanyu and Gao, Shijian and Cai, Xuesong and Cheng, Xiang and Yang, Liuqing},
  journal = {arXiv preprint arXiv:2511.22222},
  year    = {2025},
  doi     = {10.48550/arXiv.2511.22222},
  url     = {https://arxiv.org/abs/2511.22222}
}
```

Paper: [WiFo-2: a generalist foundation model unifies heterogeneous wireless
system design](https://arxiv.org/abs/2511.22222)
