# Zero-shot test data

The released test subsets are hosted in the
[WiFo-2 dataset repository](https://huggingface.co/datasets/pku-pcni-lab/WiFo-2/tree/main).

Place each dataset at:

```text
dataset/<DATASET_NAME>/X_test.pt
```

`X_test.pt` contains the test tensor directly. Training and validation splits
are not part of this release.

For channel estimation, use a released dataset name containing `CE` (for
example, `D17CE` or `O1droneCE1`).
