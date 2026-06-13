# Config Reference

This document explains the YAML configuration fields used by training and inference.

## Training config structure

Typical files:

- `dinov2_aicc.yaml`
- `dinov3_aicc.yaml`
- `osnet_aicc.yaml`

### model

```yaml
model:
  backbone: dinov3
  use_lora: true
  lora_rank: 16
  osnet_weights: null
```

- `backbone`: one of `osnet`, `dinov2`, `dinov3`
- `use_lora`: for `dinov2`/`dinov3`, `true` enables LoRA and `false` enables full fine-tuning
- `lora_rank`: LoRA rank for DINO backbones
- `osnet_weights`: optional path for OSNet pretrained weights

### data

```yaml
data:
  train: /path/to/train_records.csv
  query: /path/to/query_records.csv
  gallery: /path/to/gallery_records.csv
  num_workers: 4
```

### sampler

```yaml
sampler:
  P: 16
  K: 4
```

Batch size is `P * K`.

### optim

```yaml
optim:
  epochs: 1000
  warmup: 10
  lr: 0.0005
  mixed_precision: true

  label_smoothing: 0.1
  margin: 0.3

  triplet_weight: 1.0
  supcon_weight: 0.0
  supcon_temperature: 0.07

  max_grad_norm: 1.0
```

Training loss is:

`total_loss = CE + triplet_weight * Triplet + supcon_weight * SupCon`

### eval

```yaml
eval:
  interval: 50
  batch_size: 64
```

### output

```yaml
output:
  checkpoint_dir: checkpoints_max
  log_path: training_log.csv
```

## Inference config structure

Typical file:

- `dinov2_aicc_infer.yaml`

```yaml
model:
  backbone: dinov2
  use_lora: false
  lora_rank: 16
  osnet_weights: null

checkpoint:
  path: checkpoints/best_model.pth
  use_pretrained: false

data:
  query: data_csv/query_records.csv
  gallery: data_csv/gallery_records.csv
  num_workers: 4

eval:
  batch_size: 64

rerank:
  enabled: false
  k1: 20
  k2: 6
  lambda: 0.3

output:
  distance_matrix: distance_matrix.pt
  matching_results: matching_results.csv
```

Notes:

- If `checkpoint.use_pretrained: true`, the model runs with pretrained weights only.
- For checkpoint inference, class count is inferred from checkpoint classifier weights (no fixed class-count dependency).
- Set `rerank.enabled: true` to apply k-reciprocal reranking.

## Minimal usage examples

Train:

```bash
python -m reid.train --config configs/dinov3_aicc.yaml
```

Inference:

```bash
python -m reid.inference --config configs/dinov2_aicc_infer.yaml
```
