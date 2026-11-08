# RNA 3D Structure Prediction — Stanford RNA 3D Folding Part 2

Predicts 3D **C1′ atom coordinates** for RNA molecules from nucleotide sequence alone, as an entry to the Kaggle competition [Stanford RNA 3D Folding Part 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2). The competition metric is **TM-score, best of 5 predictions per target**, so the pipeline is built to emit five diverse structures rather than one.

> **About the name:** this repository is called `Env-science-project` for historical reasons. The contents are RNA structure prediction.

## Prediction cascade

Sequence-to-structure is attempted by the cheapest reliable method first, falling back when it has nothing to offer:

1. **Template-based modeling (TBM)** — if a similar RNA exists in the training set, copy and adapt its geometry. This carries the submission.
2. **Learned transformer** — a from-scratch neural model for sequences with no usable template.
3. **Idealized A-form helix** — last resort, so every target gets five coordinate sets.

## Template-based modeling

Implemented in `notebooks/kaggle_submission.ipynb` (section 8):

- **k-mer shortlisting** narrows thousands of candidate templates to a shortlist of 128 before any expensive alignment runs.
- **Pairwise alignment** with BioPython's `PairwiseAligner` scores the shortlist; hits below 50% identity are discarded, and the top 30 survive.
- **Coordinate adaptation** maps template C1′ coordinates onto the query through the alignment, segment by segment, with a 128-residue chunk overlap for long sequences.
- **Geometry refinement** enforces RNA bond-length and angle constraints on the adapted coordinates.
- **Diversity transforms** generate five distinct candidates from a single good template, then quality-score and deduplicate them — the metric rewards spread, so five near-identical structures waste four slots.

## The learned model

`src/model.py` — a transformer encoder with AlphaFold-inspired pairwise reasoning, roughly 8 layers at `d_model=256`:

| Stage | What it does |
| --- | --- |
| Embedding | Nucleotide tokens (`A C G U` + pad/unk) plus sinusoidal positional encoding |
| `PairwiseModule` | Outer product of the single representation → pair representation → projected down to a per-pair scalar **attention bias** |
| `StructureAwareAttention` | Multi-head self-attention that adds that pairwise bias to its logits, so sequence-distant but structurally-adjacent residues can attend to each other |
| `StructureModule` | Five independent coordinate-head MLPs emitting `(L, 5, 3)`, plus a sigmoid confidence head (pLDDT-like) |

**Training** (`src/train.py`): mixed-precision AMP, 1000-step warmup into cosine decay, gradient clipping at 1.0, TM-score evaluation on a validation split, checkpointing on the best score.

**Loss** (`src/losses.py`): `CombinedLoss` = FAPE (weight 1.0) + distance-matrix loss (0.5) + confidence MSE (0.1), each averaged across the five predictions. The confidence target is derived from each head's own normalized quality, so the model learns which of its five guesses to trust. TM-score uses Kabsch superposition and is reported best-of-N, matching the competition.

**Data** (`src/dataset.py`, `scripts/download_data.py`): C1′ coordinates are parsed from both PDB and mmCIF; `download_data.py` queries the RCSB PDB REST search API for RNA-containing structures under 4.0 Å resolution, alongside the competition's own files. Variable-length sequences are padded and masked to `max_seq_len=512`.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | `model.py`, `dataset.py`, `losses.py`, `train.py`, `inference.py` |
| `notebooks/kaggle_submission.ipynb` | Self-contained submission notebook: auto-detects Kaggle vs. local, includes the full TBM pipeline |
| `configs/default.yaml` | Model, training and inference hyperparameters |
| `scripts/` | `download_data.py`, `run_training.sh`, `run_inference.sh` |

## Running it

```bash
pip install -r requirements.txt

python scripts/download_data.py --source pdb --output data/train
bash scripts/run_training.sh configs/default.yaml
bash scripts/run_inference.sh checkpoints/best_model.pt data/test_sequences.csv
```

Inference writes `submission.csv` as `ID, resname, resid, x_1, y_1, z_1, …, x_5, y_5, z_5`.

Defaults: `batch_size=4`, `lr=3e-4`, `weight_decay=1e-4`, 100 epochs, `seed=42`.

## Status

This repository is the **pipeline**, not a result: no trained checkpoint and no leaderboard score are committed, and the training run is GPU-bound. The TBM path is the part that stands on its own, since it needs no training at all.
