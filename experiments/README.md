# Model checkpoints

The released Dense checkpoints are hosted in the
[WiFo-2 model repository](https://huggingface.co/pku-pcni-lab/WiFo-2/tree/main).

| Checkpoint path under this directory | Inference arguments |
| --- | --- |
| `Tiny-dense/model_best.pkl` | `--size tinypro --MoE False` |
| `Little-dense/model_best.pkl` | `--size littlepro --MoE False` |
| `Small-dense/model_best.pkl` | `--size smallpro --MoE False` |
| `Base-dense/model_best.pkl` | `--size basepro1 --MoE False` |

Download the desired checkpoint under this directory and pass its path to
`--file_load_path`. The `.pkl` suffix is optional in the command.
See the [main README](../README.md) for download commands and inference examples.
