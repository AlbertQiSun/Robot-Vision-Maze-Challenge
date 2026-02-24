# Visual Navigation Player — Cross-Device Context

> **Purpose**: This file provides context for AI assistants (like Cursor/Claude)
> when working on this project from a different machine. Read this first.

---

## Project Overview

**Course**: NYU ROB-GY 6203 Robot Perception — Embodied AI Challenge
**Task**: Autonomous visual navigation in a 31×31 maze using onboard FPV camera
**Repo**: https://github.com/ai4ce/vis_nav_player (our fork)
**Game Engine**: https://github.com/ai4ce/vis_nav_game_public

### Grading (15 pts total)
| Category | Points | Our Target |
|---|---|---|
| Report | 5 pts | PDF format |
| Participation | 2 pts | ✅ Automatic |
| Achieving Goal | 3 pts | ✅ Check in at goal |
| Time to completion | 3 pts | Under 1 min → 3 pts |
| Methodology | 2 pts | **Full Automation = 2 pts** |
| Judge Evaluations | 3 pts | Quality approach |

---

## Repository Structure

```
vis_nav_player/
│
├── player.py                       ← ENTRY POINT (game submission)
├── baseline.py                     ← original VLAD baseline (reference)
│
├── vis_nav/                        ← main Python package
│   ├── __init__.py
│   ├── config.py                   ← ALL constants & hyperparameters
│   ├── utils.py                    ← device detection, caching, I/O
│   │
│   ├── models/                     ← neural network modules
│   │   ├── __init__.py
│   │   ├── backbone.py             ← DINOv2 + GeM + ProjectionMLP
│   │   └── action_predictor.py     ← fallback action MLP
│   │
│   ├── navigation/                 ← navigation logic
│   │   ├── __init__.py
│   │   ├── graph.py                ← topological graph (temporal + visual)
│   │   ├── localizer.py            ← FAISS index + temporal consistency
│   │   └── planner.py              ← goal setup, check-in, stuck recovery
│   │
│   └── data/                       ← datasets & transforms
│       ├── __init__.py
│       ├── transforms.py           ← image transforms + batch extraction
│       └── maze_dataset.py         ← MazeExplorationDataset + PKSampler
│
├── scripts/                        ← standalone CLI training scripts
│   ├── train_projection.py         ← train DINOv2 projection head (CUDA)
│   ├── train_action_predictor.py   ← train action MLP (CUDA)
│   └── generate_mazes.py           ← generate random maze layouts
│
├── models/                         ← trained weights (Git LFS)
│   └── .gitkeep
├── cache/                          ← auto-generated (gitignored)
├── data/                           ← maze exploration data (gitignored)
├── training_data/                  ← multi-maze data (gitignored)
│
├── source/
│   └── baseline.py                 ← kept for README backward compat
│
├── setup.sh                        ← one-command env setup
├── environment.yaml                ← conda env spec
├── requirements.txt
├── .gitattributes                  ← Git LFS tracking rules
├── .gitignore
├── Idea.md                         ← full system design document
├── CONTEXT.md                      ← THIS FILE
└── README.md
```

---

## Architecture

```
FPV Image → DINOv2 ViT-B/14 (frozen) → GeM Pooling → Projection MLP → 256-dim
                                                                          │
                                                              FAISS Nearest Neighbor
                                                                          │
                                                          Topological Graph + A*
                                                                          │
                                                              Autonomous Navigation
```

### Config lives in one place

All tunable constants are in `vis_nav/config.py` as frozen dataclasses:

- `PathCfg` — file paths
- `FeatureCfg` — backbone / projection hyper-params
- `GraphCfg` — graph construction
- `NavCfg` — online navigation
- `TrainCfg` — training hyper-params

Other modules import from config — no magic numbers scattered around.

---

## Compute Devices

| Device | Role | PyTorch device |
|---|---|---|
| Apple M4 Max (128 GB) | Dev, testing, competition inference | `mps` |
| NVIDIA RTX 5090 Laptop | Training option A | `cuda` |
| NVIDIA H20 96 GB | Training option B | `cuda` |

---

## Setup on a New Machine

```bash
git clone https://github.com/YOUR_USERNAME/vis_nav_player.git
cd vis_nav_player
chmod +x setup.sh && ./setup.sh
conda activate game
```

---

## Training Workflow (CUDA Machine)

### Step 1: Train Projection Head
```bash
# Single maze (quick):
python scripts/train_projection.py --data-dir data/ --single-maze --device cuda

# Multi-maze (full):
python scripts/train_projection.py --data-dir training_data/ --device cuda
```

### Step 2: Train Action Predictor
```bash
python scripts/train_action_predictor.py \
    --data-dir data/ --single-maze --device cuda
```

### Step 3: Push Trained Models
```bash
git add models/ && git commit -m "trained weights" && git push
```

---

## Running the Player

```bash
conda activate game
python player.py              # autonomous player
python baseline.py            # original VLAD baseline
```

---

## Current Status

- [x] Package structure (`vis_nav/`)
- [x] Centralised config
- [x] DINOv2 + GeM + Projection backbone
- [x] Topological navigation graph
- [x] FAISS localiser + temporal consistency
- [x] Goal planner (multi-view, stuck recovery, check-in)
- [x] Action predictor fallback
- [x] Training scripts (projection + action)
- [x] setup.sh + Git LFS
- [ ] **TRAINING NOT YET DONE** — run on CUDA machine
- [ ] **END-TO-END TEST** with trained models
- [ ] Report (PDF)

---

## Important Notes

1. **Model weights** tracked via Git LFS (`models/*.pth`, `models/*.pt`)
2. **Cache** (`cache/`) is gitignored — auto-generated at runtime
3. **Competition maze is DIFFERENT** — models must generalise
4. DINOv2 backbone downloaded from `torch.hub` on first run (~350 MB)
5. Player auto-detects trained models; falls back to keyboard if missing
