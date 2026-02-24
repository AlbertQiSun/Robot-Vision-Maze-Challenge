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
| Time to completion | 3 pts | Under 1 min (3 pts) |
| Methodology | 2 pts | **Full Automation = 2 pts** |
| Judge Evaluations | 3 pts | Quality approach |
| Extra Credits | Bonus | Final competition |

---

## Architecture

```
FPV Image → DINOv2 ViT-B/14 (frozen) → GeM Pooling → Projection MLP → 256-dim descriptor
                                                                              │
                                                                              ▼
                                                                    FAISS Nearest Neighbor
                                                                              │
                                                                              ▼
                                                              Topological Graph + A* Planning
                                                                              │
                                                                              ▼
                                                                    Autonomous Navigation
```

### Key Files
| File | Purpose | Run On |
|---|---|---|
| `source/player.py` | **Main autonomous player (submission)** | M4 Max / Any |
| `source/feature_extractor.py` | DINOv2 + GeM + Projection module | Imported |
| `source/nav_graph.py` | Topological graph construction | Imported |
| `source/train_projection.py` | Train DINOv2 projection head | **CUDA GPU** |
| `source/train_action_predictor.py` | Train action predictor MLP | **CUDA GPU** |
| `source/generate_mazes.py` | Generate random maze layouts | Any |
| `source/baseline.py` | Original VLAD baseline (reference only) | Any |
| `models/projection_head.pth` | Trained projection weights (Git LFS) | — |
| `models/action_predictor.pth` | Trained action predictor weights (Git LFS) | — |

---

## Compute Devices

### Device 1: Apple M4 Max (Development & Inference)
- **OS**: macOS
- **RAM**: 128 GB
- **GPU**: MPS (Metal Performance Shaders)
- **Role**: Development, testing, competition-day inference
- **PyTorch device**: `mps`

### Device 2: NVIDIA RTX 5090 Laptop (Training Option A)
- **Role**: Training projection head + action predictor
- **PyTorch device**: `cuda`

### Device 3: NVIDIA H20 96 GB (Training Option B)
- **Role**: Heavy training (multi-maze, large batches)
- **PyTorch device**: `cuda`

---

## Setup on a New Machine

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/vis_nav_player.git
cd vis_nav_player

# 2. Run setup (auto-detects CUDA/MPS/CPU)
chmod +x setup.sh
./setup.sh

# 3. Activate environment
conda activate game
```

---

## Training Workflow (CUDA Machine)

### Step 1: Prepare Training Data
```bash
# Option A: Use single practice maze (data/ already exists)
# Option B: Generate multiple mazes
python source/generate_mazes.py --n-mazes 30 --output-dir training_data/

# For actual images, run the game engine on each maze
# (see generate_mazes.py output for details)
```

### Step 2: Train Projection Head
```bash
# Single maze (quick test):
python source/train_projection.py \
  --data-dir data/ \
  --single-maze \
  --output models/projection_head.pth \
  --device cuda \
  --epochs-phase-a 30 \
  --epochs-phase-b 20

# Multi-maze (full training):
python source/train_projection.py \
  --data-dir training_data/ \
  --output models/projection_head.pth \
  --device cuda \
  --epochs-phase-a 30 \
  --epochs-phase-b 20 \
  --texture-dir data/textures/ \
  --synthetic-ratio 0.2
```

### Step 3: Train Action Predictor
```bash
python source/train_action_predictor.py \
  --data-dir data/ \
  --single-maze \
  --projection-model models/projection_head.pth \
  --output models/action_predictor.pth \
  --device cuda
```

### Step 4: Push Trained Models
```bash
git add models/
git commit -m "Add trained model weights"
git push
```

---

## Running the Player

```bash
# Activate environment
conda activate game

# Run autonomous player
python source/player.py

# Run original baseline (for comparison)
python source/baseline.py
```

---

## Current Status

- [x] Feature extractor module (DINOv2 + GeM + Projection)
- [x] Navigation graph module
- [x] Training scripts (projection + action predictor)
- [x] Multi-maze generation script
- [x] Fully autonomous player
- [x] Setup script (Linux/macOS)
- [x] Git LFS for model weights
- [ ] **TRAINING NOT YET DONE** — need to run on CUDA machine
- [ ] **TESTING** — need to verify end-to-end with trained models
- [ ] Report (PDF)

---

## Important Notes

1. **Model weights** are tracked via Git LFS (`models/*.pth`, `models/*.pt`)
2. **Cache files** (`cache/`) are gitignored — auto-generated at runtime
3. **Training data** (`training_data/`) is gitignored — too large for git
4. **Practice maze data** (`data/`) is gitignored — downloaded by game engine
5. The **competition maze is DIFFERENT** from practice — models must generalize
6. The player auto-detects if trained models exist; falls back to keyboard if not
7. DINOv2 backbone is downloaded from torch.hub on first run (~350 MB)

---

## Key Design Decisions

- **DINOv2 ViT-B/14** (not ViT-S) for better feature quality
- **256-dim** output descriptors (compact for FAISS, sufficient capacity)
- **GeM Pooling** over CLS token (better for retrieval)
- **Multi-Similarity Loss** (better than triplet loss for metric learning)
- **Cross-maze training** (critical for generalization to unseen mazes)
- **A\* with feature heuristic** (faster than pure Dijkstra)
- **Multi-view goal matching** (uses all 4 target views, not just front)
- **Stuck detection + recovery** (6-step progressive recovery sequence)
- **Multi-frame check-in confirmation** (avoids false positive check-ins)

---

## Troubleshooting

### "vis_nav_game not found"
```bash
pip install --extra-index-url https://test.pypi.org/simple/ vis-nav-game
```

### "CUDA out of memory" during training
- Reduce `--batch-size` (e.g., 32 instead of 64)
- Use `--epochs-phase-b 0` to skip backbone fine-tuning

### "Model not found" when running player
- Train on CUDA machine first, then `git pull` on inference machine
- Or run without trained model (uses raw DINOv2 features)

### Cache issues
- Delete `cache/` directory to force re-computation
- Cache is keyed by data hash, so changing data auto-invalidates
