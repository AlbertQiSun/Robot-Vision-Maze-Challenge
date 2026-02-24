# Autonomous Visual Navigation — Full System Design

## Goal
Build a **fully autonomous** player that navigates a 31×31 maze to a target location
in **under 30 seconds**, using only onboard camera images from the exploration phase.

### Compute Resources
- **Local Dev & Inference**: Apple M4 Max (MPS) — 128 GB RAM
- **Training Option A**: NVIDIA RTX 5090 Laptop
- **Training Option B**: NVIDIA H20 96 GB

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OFFLINE: TRAINING (H20 / 5090)                       │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Stage 1: Multi-Maze Data Collection                               │  │
│  │   → Generate 20-50 random mazes via public game engine            │  │
│  │   → Run exploration on each → 300K-900K images total             │  │
│  │   → Augment with texture collages + heavy transforms              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Stage 2: DINOv2-B/14 backbone (frozen → partial unfreeze)        │  │
│  │   → Patch tokens (768-dim × 256 patches)                        │  │
│  │   → GeM pooling → 768-dim global descriptor                     │  │
│  │   → Projection MLP (768 → 512 → 256)                            │  │
│  │   → L2 normalize → 256-dim place descriptor                     │  │
│  │                                                                   │  │
│  │ Train with: Multi-Similarity Loss + Hard Negative Mining         │  │
│  │             on ALL mazes combined (cross-maze negatives)          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Stage 3: Re-ranking with SuperPoint + LightGlue                  │  │
│  │   → Local feature matching for geometric verification            │  │
│  │   → Filters out false positive place matches                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Stage 4: Action Prediction Head                                   │  │
│  │   → Given (current_descriptor, goal_descriptor) → predict action │  │
│  │   → Small MLP trained on ALL maze trajectories                   │  │
│  │   → Backup when graph-based planning fails                       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Exported models + cached features
┌─────────────────────────────────────────────────────────────────────────┐
│                    OFFLINE: PRE-NAVIGATION (M4 Max)                     │
│                                                                         │
│  Exploration Images ──► DINOv2-B + Projection ──► Feature DB (N×256)   │
│         │                                              │                │
│         │                                     FAISS IVF Index           │
│         │                                              │                │
│         └──► Topological Graph                         │                │
│              (temporal + visual + loop-closure edges)   │                │
│                                                        │                │
│  4 Target Images ──► Feature Extraction ──► Multi-View Goal Fusion     │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ONLINE: NAVIGATION LOOP (M4 Max)                     │
│                                                                         │
│  FPV ──► Feature ──► FAISS top-K ──► Temporal Consistency Filter       │
│                                              │                          │
│                                    Current Node Estimate                │
│                                              │                          │
│                              Dijkstra + A* to Goal Node                │
│                                              │                          │
│                     ┌────────────────────────┼──────────────┐           │
│                     │                        │              │           │
│                  Temporal Edge          Visual Edge     At Goal?        │
│                  → Execute Action      → Relocalize    → CHECKIN       │
│                     │                        │                          │
│                     └────────┬───────────────┘                          │
│                              │                                          │
│                     Stuck Detection + Recovery                          │
│                     Frontier Exploration Fallback                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Training Data — Collection & Extension

### 1A. Primary Data: Exploration Trajectories

The game engine provides exploration data for each maze:

```
data/
├── data_info.json     # 18,295 entries: {step, image, action}
└── images/            # 18,295 JPEG images (320×240 RGB)
```

**Breakdown of the practice maze data:**
| Category | Count | Notes |
|---|---|---|
| Total frames | 18,295 | Full exploration trajectory |
| FORWARD | 4,282 | Robot moving through corridors |
| LEFT | 3,991 | Turning left at junctions |
| RIGHT | 4,030 | Turning right at junctions |
| BACKWARD | 14 | Rare reversals |
| IDLE | 5,913 | Robot stationary (drop these) |
| Multi-action | 65 | Simultaneous keys (drop these) |
| **Usable motion frames** | **12,317** | Single pure-action frames |

**Coverage**: ~38 frames per maze cell → every corridor is visited multiple times
from different angles. The maze is well-covered by exploration.

### 1B. Data Extension Strategy 1: Multi-Maze Generation (CRITICAL)

The competition maze is **different** from the practice maze. To build a model
that generalizes, train on **many different mazes**.

**How to generate:**
```bash
# Clone the public game engine
git clone https://github.com/ai4ce/vis_nav_game_public.git

# Generate random mazes with different seeds
# The engine creates maze + textures + exploration path automatically
# Run exploration in headless mode, save all frames + data_info.json

# Collect 20-50 mazes:
training_data/
├── maze_001/
│   ├── data_info.json
│   └── images/          # ~18K images
├── maze_002/
│   ├── data_info.json
│   └── images/
├── ...
└── maze_050/
    ├── data_info.json
    └── images/
```

**Data volume:**
| # Mazes | Total Images | Disk Space | Training Time (H20) |
|---|---|---|---|
| 1 (practice only) | 18K | ~450 MB | ~30 min |
| 10 | ~180K | ~4.5 GB | ~2 hrs |
| 20 | ~360K | ~9 GB | ~4 hrs |
| 50 | ~900K | ~22 GB | ~8-10 hrs |

**Recommendation: 20-30 mazes** — sweet spot between diversity and training time.

**Why this is the single most important data decision:**
- Each maze uses different random combinations of the 200 texture patterns
- The model learns: "same texture ≠ same place" (cross-maze negatives)
- The model learns: "corridor junctions look similar across mazes" (structural priors)
- DINOv2 backbone generalizes the visual understanding; projection head learns
  the place-recognition task across diverse environments

### 1C. Data Extension Strategy 2: Texture Collage Synthesis

You have all **200 wall texture patterns** in `data/textures/` (pattern_1.png
through pattern_200.png, various sizes ~400-900px). Use them to synthesize
additional training images:

**Synthetic corridor generation:**
```python
def generate_synthetic_corridor(textures, img_size=(320, 240)):
    """Create a fake corridor view from random textures."""
    img = np.zeros((*img_size[::-1], 3), dtype=np.uint8)

    # Pick 2-4 random textures
    left_tex = random.choice(textures)
    right_tex = random.choice(textures)
    floor_color = np.random.randint(50, 200, 3)

    # Compose with perspective transform to simulate depth
    # Left wall: left 30% of image, with perspective
    # Right wall: right 30% of image, with perspective
    # Floor: bottom 40%
    # Ceiling: top 20%

    h, w = img_size[1], img_size[0]

    # Left wall
    left_crop = cv2.resize(left_tex, (w // 3, h))
    pts_src = np.float32([[0,0], [w//3,0], [w//3,h], [0,h]])
    pts_dst = np.float32([[0,h//5], [w//4,0], [w//4,h], [0,h*4//5]])
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    left_wall = cv2.warpPerspective(left_crop, M, (w, h))

    # Right wall (similar, mirrored)
    right_crop = cv2.resize(right_tex, (w // 3, h))
    pts_dst_r = np.float32([[w,h//5], [w*3//4,0], [w*3//4,h], [w,h*4//5]])
    M_r = cv2.getPerspectiveTransform(pts_src, pts_dst_r)
    right_wall = cv2.warpPerspective(right_crop, M_r, (w, h))

    # Compose
    mask_l = left_wall.sum(axis=2) > 0
    mask_r = right_wall.sum(axis=2) > 0
    img[h*3//5:] = floor_color  # floor
    img[:h//5] = floor_color * 0.8  # ceiling
    img[mask_l] = left_wall[mask_l]
    img[mask_r] = right_wall[mask_r]

    return img
```

**Training usage:**
- Generate 50K-100K synthetic images on-the-fly during training
- Synthetic images are **always negatives** — they don't correspond to any
  real place in any maze
- This teaches the anti-aliasing objective: "knowing which textures you see
  is NOT enough to know WHERE you are"
- Mix ratio: 20% synthetic, 80% real exploration images per batch

### 1D. Data Extension Strategy 3: Heavy Augmentation

Applied to ALL real exploration images during training. These augmentations
simulate the differences between exploration-time and navigation-time views:

```python
train_transform = T.Compose([
    # --- Spatial ---
    T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
    T.RandomPerspective(distortion_scale=0.15, p=0.3),

    # --- Photometric ---
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    T.RandomGrayscale(p=0.05),
    T.GaussianBlur(kernel_size=7, sigma=(0.1, 3.0)),

    # --- Corruption ---
    T.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # partial occlusion
    RandomJPEGCompression(quality=(50, 95), p=0.3),  # compression artifacts

    # --- DO NOT USE ---
    # T.RandomHorizontalFlip  → breaks left/right spatial info!
    # T.RandomVerticalFlip    → physically impossible view

    # --- Normalize ---
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

**Why each augmentation matters:**
| Augmentation | Simulates | Impact |
|---|---|---|
| RandomResizedCrop | Robot at slightly different position | High — most common difference |
| RandomPerspective | Different viewing angle | Medium — at turns/junctions |
| ColorJitter | Lighting variation | Medium — game renderer differences |
| GaussianBlur | Motion blur while moving | Low-medium |
| RandomErasing | Partial view obstruction | Low — robustness padding |
| JPEG Compression | Image quality variation | Low — robustness padding |

**Effective dataset size with augmentation:**
Each epoch sees every image with different random transforms → effectively
**infinite** training data from 18K base images.

### 1E. Data Extension Strategy 4: Multi-Run Exploration

Run the game multiple times on the same practice maze to collect more views:

```python
class DataCollectorPlayer(Player):
    """Run this player multiple times to collect additional exploration data."""

    def __init__(self, run_id):
        super().__init__()
        self.run_id = run_id
        self.save_dir = f'data/extra_runs/run_{run_id}'
        os.makedirs(self.save_dir, exist_ok=True)
        self.frame_count = 0
        self.log = []

    def reset(self):
        pygame.init()
        self.keymap = {pygame.K_ESCAPE: Action.QUIT}

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return
        if self._state and self._state[1] == Phase.EXPLORATION:
            cv2.imwrite(f'{self.save_dir}/{self.frame_count}.jpg', fpv)
            self.log.append({
                'step': self.frame_count,
                'image': f'{self.frame_count}.jpg',
                'state': list(self._state) if self._state else None
            })
            self.frame_count += 1

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return Action.QUIT
        return Action.IDLE  # let the engine drive exploration

    def pre_navigation(self):
        # Save log and quit after exploration
        with open(f'{self.save_dir}/data_info.json', 'w') as f:
            json.dump(self.log, f)
        print(f'Saved {self.frame_count} frames to {self.save_dir}')
```

**Value:** Same maze, different exploration paths → more viewpoints of the same
corridors. Useful for validation and for training data diversity within one maze.

**Run 3-5 times** → ~60-90K total images for the practice maze.

---

### Training Data Summary

| Source | Images | Effort | Generalization Value |
|---|---|---|---|
| **Practice maze** (provided) | 18K | Zero | Low (one maze only) |
| **Multi-maze generation** | 360K-900K | 3-4 hrs setup | **Very High** (different layouts + textures) |
| **Texture collage synthesis** | 50K-100K | 1 hr implement | Medium (anti-aliasing) |
| **Heavy augmentation** | ∞ (on-the-fly) | Zero | High (robustness) |
| **Multi-run exploration** | 60-90K | 1 hr (run game 5x) | Low-Medium (same maze, more views) |
| **Total effective training pool** | **500K-1M+** | | |

---

## Part 2: Feature Backbone — DINOv2-B/14 + Learned Projection

### Why DINOv2-B (not ViT-S)?
- **ViT-B/14** produces **768-dim** patch tokens (vs 384 for ViT-S)
- With H20 96GB and 5090, we can comfortably use the larger model
- Better feature quality = fewer localization errors = faster navigation
- At inference on M4 Max MPS, ViT-B/14 still runs at ~15-20 ms per image

### Architecture

```
┌──────────────────────────────────────────────────┐
│  DINOv2 ViT-B/14 (frozen, from torch.hub)        │
│  Input: 224×224 RGB                               │
│  Output: [CLS] + 256 patch tokens, each 768-dim  │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Generalized Mean (GeM) Pooling                   │
│  pool(X) = (1/N Σ x_i^p)^(1/p),  p=3 (learnable)│
│  Output: 768-dim global descriptor                │
│  (better than CLS token for retrieval tasks)      │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Projection MLP (trained)                         │
│  Linear(768, 512) → BN → GELU → Dropout(0.1)    │
│  Linear(512, 256) → L2 normalize                  │
│  Output: 256-dim place descriptor                 │
└──────────────────────────────────────────────────┘
```

### Why GeM Pooling Over CLS Token?
- CLS token is a single learned "summary" — can miss spatial details
- GeM aggregates ALL patch tokens with learnable emphasis
- Higher `p` = more emphasis on high-activation patches (salient regions)
- Proven superior for image retrieval (used in all top VPR methods)

---

## Part 3: Training — Multi-Similarity Loss with Cross-Maze Hard Mining

### Training Objective

**Multi-Similarity Loss** (Wang et al., CVPR 2019) — better than triplet loss:
- Considers ALL positive and negative pairs in a batch simultaneously
- Automatically weights hard pairs more
- Converges faster and to better solutions

```python
class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=50.0, base=0.5):
        super().__init__()
        self.alpha = alpha   # weight for positive pairs
        self.beta = beta     # weight for negative pairs
        self.base = base     # margin / threshold

    def forward(self, embeddings, labels):
        # embeddings: (B, 256), L2-normalized
        # labels: (B,) — place group ID (encodes maze_id + trajectory_position)
        sim_mat = embeddings @ embeddings.T

        loss = 0
        for i in range(len(embeddings)):
            pos_mask = (labels == labels[i]) & (torch.arange(len(labels)) != i)
            neg_mask = labels != labels[i]

            pos_sim = sim_mat[i][pos_mask]
            neg_sim = sim_mat[i][neg_mask]

            # Hard mining: only keep informative pairs
            pos_loss = (1/self.alpha) * torch.log(
                1 + torch.sum(torch.exp(-self.alpha * (pos_sim - self.base)))
            )
            neg_loss = (1/self.beta) * torch.log(
                1 + torch.sum(torch.exp(self.beta * (neg_sim - self.base)))
            )
            loss += pos_loss + neg_loss

        return loss / len(embeddings)
```

### Training Data Construction — Cross-Maze Place Groups

**Within one maze** (same as before):
```python
POSITIVE_RANGE = 5      # frames within ±5 steps = same place
NEGATIVE_MIN_GAP = 50   # frames > 50 steps apart = different place
```

**Across mazes** (the key innovation):
```python
# Every frame from maze_A is a NEGATIVE to every frame from maze_B
# This is automatically true because place labels encode maze_id

# Label format: place_label = maze_id * 100000 + trajectory_position // POSITIVE_RANGE
# E.g., maze_3, step 150 → label = 300030
#        maze_7, step 150 → label = 700030 (different label, even if they look similar!)

# Batch sampling strategy:
# Sample P=16 places from DIFFERENT mazes:
#   - 4 places from maze_A
#   - 4 places from maze_B
#   - 4 places from maze_C
#   - 4 places from maze_D
# Each place has K=4 images
# Total batch = 64 images

# This creates HARD cross-maze negatives:
# Two corridor images from different mazes with the same wall texture
# MUST be pushed apart in embedding space
```

**Why cross-maze negatives are so powerful:**
- The 200 texture patterns are reused across mazes
- Maze A might have pattern_42 at position (5, 10)
- Maze B might have pattern_42 at position (20, 3)
- Without cross-maze training, the model might confuse these
- With cross-maze training, the model learns: "texture alone is not identity —
  spatial context (what's around you) determines WHERE you are"

### Hard Negative Mining — Enhanced

```python
class CrossMazeHardMiner:
    """Mine the hardest negatives across all mazes."""

    def __init__(self, features_by_maze, labels_by_maze):
        # features_by_maze: dict[maze_id] → (N, 256) feature matrix
        # labels_by_maze: dict[maze_id] → (N,) place labels
        self.all_features = np.vstack(list(features_by_maze.values()))
        self.all_labels = np.concatenate(list(labels_by_maze.values()))
        self.maze_ids = np.concatenate([
            np.full(len(f), mid) for mid, f in features_by_maze.items()
        ])

    def mine_batch(self, batch_size=64, P=16, K=4):
        """Sample a batch with hard negatives."""
        # Step 1: Sample P random places from random mazes
        unique_labels = np.unique(self.all_labels)
        selected_labels = np.random.choice(unique_labels, P, replace=False)

        batch_indices = []
        batch_labels = []
        for label in selected_labels:
            candidates = np.where(self.all_labels == label)[0]
            chosen = np.random.choice(candidates, min(K, len(candidates)), replace=False)
            batch_indices.extend(chosen)
            batch_labels.extend([label] * len(chosen))

        # Step 2: For each anchor, find hardest cross-maze negative
        batch_feats = self.all_features[batch_indices]
        sim_matrix = batch_feats @ self.all_features.T

        hard_negatives = []
        for i, idx in enumerate(batch_indices):
            label = batch_labels[i]
            maze = self.maze_ids[idx]
            # Negatives: different label AND different maze (hardest)
            neg_mask = (self.all_labels != label)
            neg_sims = sim_matrix[i].copy()
            neg_sims[~neg_mask] = -2  # mask positives
            hardest = np.argmax(neg_sims)
            hard_negatives.append(hardest)

        # Add hard negatives to batch
        batch_indices.extend(hard_negatives[:P])  # add P hard negatives
        batch_labels.extend([self.all_labels[i] for i in hard_negatives[:P]])

        return batch_indices, batch_labels
```

### Training Details

| Setting | Value |
|---|---|
| Backbone | DINOv2 ViT-B/14 |
| Trainable (Phase A) | GeM pooling (p) + Projection MLP |
| Trainable (Phase B) | + last 2 ViT blocks |
| Loss | Multi-Similarity Loss |
| Optimizer | AdamW, lr=3e-4 (projection), 1e-5 (backbone) |
| Scheduler | CosineAnnealingWarmRestarts, T_0=10 |
| Batch | P=16 places × K=4 = 64 images (from mixed mazes) |
| Phase A epochs | 30 (frozen backbone) |
| Phase B epochs | 20 (unfreeze last 2 blocks) |
| Mining | Hard negative mining every 5 epochs |
| Augmentation | Heavy (see Part 1D) |
| Synthetic mix | 20% texture collages per batch |
| Train data | 20-30 mazes (~360K-540K images) |
| Train device | H20 96GB |
| Train time | ~4-8 hrs total |

### Two-Phase Training Strategy

**Phase A: Frozen backbone, train projection only (30 epochs)**
- Fast convergence, learns the task-specific embedding space
- GeM pooling `p` starts at 3.0, learns optimal value
- All 20-30 mazes used, cross-maze batches
- Heavy augmentation + 20% synthetic collages

**Phase B: Unfreeze last 2 ViT blocks + projection (20 epochs)**
- Fine-tune backbone for maze-specific texture discrimination
- Lower learning rate: 1e-5 for backbone, 1e-4 for projection
- H20 96GB handles ViT-B/14 backprop comfortably
- This phase teaches DINOv2 to attend to the specific visual cues that
  distinguish one maze corridor from another

---

## Part 4: Re-Ranking with SuperPoint + LightGlue

### Why Re-Ranking?
Global descriptors find "roughly similar" places. But in a maze with
200 texture patterns, there WILL be false positives — two corridors with
identical wall textures but in different maze locations.

**Local feature matching** can verify: do the geometric details actually match?

### Pipeline
```
Step 1: Global retrieval
  → FAISS top-20 candidates from DINOv2 descriptor

Step 2: Re-rank with local features (only for top-20, so it's fast)
  → SuperPoint keypoints + descriptors on query image
  → SuperPoint on each candidate image (pre-computed offline)
  → LightGlue matching → count inliers after RANSAC
  → Re-rank by inlier count
  → Top-1 after re-ranking = localized node
```

### SuperPoint + LightGlue
- **SuperPoint**: learned keypoint detector, much better than SIFT in
  low-texture / repetitive environments
- **LightGlue**: lightweight learned matcher (successor to SuperGlue)
- Both available in `kornia` or standalone repos
- Pre-compute SuperPoint descriptors for all exploration images offline

### Offline Pre-computation
```python
# For each exploration image:
#   1. Extract SuperPoint keypoints + descriptors → cache to disk
#   2. Store as dict: {image_idx: {'keypoints': ..., 'descriptors': ...}}
# Total storage: ~500 MB for 18K images (manageable)
```

### Online Re-ranking (only when needed)
```python
def _rerank(self, query_img, candidates, k=5):
    """Re-rank top-K candidates using local feature matching."""
    query_sp = self.superpoint.extract(query_img)

    scores = []
    for idx in candidates:
        db_sp = self.sp_cache[idx]
        matches = self.lightglue.match(query_sp, db_sp)
        n_inliers = len(matches['inliers'])
        scores.append(n_inliers)

    reranked = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
    return reranked
```

### When to Use Re-Ranking
- **NOT every frame** — too slow
- Use re-ranking only when:
  1. Initial localization confidence is low (top-1 similarity < threshold)
  2. Top-1 and top-2 candidates are far apart in the graph (ambiguous)
  3. After stuck recovery (need confident re-localization)

---

## Part 5: Navigation Graph — Enhanced

### Node Selection
- **Subsample rate = 2** (use every 2nd pure-motion frame)
- ~6,200 nodes from ~12,400 pure-motion frames
- Keep all junction frames (where action changes, e.g., FORWARD→LEFT)

### Edge Types

**1. Forward Temporal Edges**
```
Node i → Node i+1
  weight = 1.0
  action = recorded action at frame i
  edge_type = 'temporal_forward'
```

**2. Backward Temporal Edges**
```
Node i+1 → Node i
  weight = 1.5  (slightly penalized — going backward is less reliable)
  action = REVERSE(recorded action at frame i)
  edge_type = 'temporal_backward'
```

**3. Visual Shortcut Edges (Loop Closures)**
```
Node i ↔ Node j, where:
  - |i - j| > MIN_GAP (50)
  - cosine_similarity(feat_i, feat_j) > SIMILARITY_THRESHOLD (0.85)
  - Top-K globally (K = 500) + per-node top-3

  weight = 0.5  (these are essentially "teleportation" —
                  if localized correctly, you're already there)
  edge_type = 'visual'
```

**Why weight 0.5 for visual edges?**
Visual edges represent *loop closures* — the robot is at a place it has
visited before. The edge doesn't require the robot to move; it just means
"you are now at this other node in the graph." This is free (0 actions),
but we add 0.5 to prevent the planner from chaining many visual jumps
(which could mean localization errors are compounding).

### Per-Node Top-K Shortcut Edges
In addition to global top-K, ensure every node has at least 2-3 shortcut
connections. This prevents "island" regions with no shortcuts:

```python
# For each node, find top-3 most similar nodes (beyond MIN_GAP)
for i in range(n):
    sims = features[i] @ features.T
    sims[max(0,i-MIN_GAP):min(n,i+MIN_GAP+1)] = -1  # mask nearby
    top3 = np.argsort(sims)[-3:]
    for j in top3:
        if sims[j] > 0.80:  # minimum quality threshold
            G.add_edge(i, j, weight=0.5, edge_type='visual')
```

### A* Heuristic for Faster Path Planning

Instead of vanilla Dijkstra, use A* with a heuristic based on
feature similarity to the goal:

```python
def _heuristic(self, node):
    """Estimated cost from node to goal based on feature distance."""
    sim = float(self.features[node] @ self.goal_feature)
    # Higher similarity = closer to goal = lower heuristic
    return max(0, 1 - sim) * HEURISTIC_SCALE

path = nx.astar_path(self.G, current, goal,
                     heuristic=self._heuristic, weight='weight')
```

---

## Part 6: Autonomous Navigation Controller

### Full State Machine

```
                    ┌─────────────────────────────┐
                    │          INIT                │
                    │  Load models, features, graph │
                    │  Find goal node               │
                    └────────────┬────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │            LOCALIZE                   │
              │  Extract feature from FPV              │
              │  FAISS top-K → temporal filter         │
              │  (Optional) Re-rank with LightGlue    │
              │  → current_node                        │
              └──────────────┬───────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────────┐
              │          PLAN PATH                    │
              │  A* from current_node to goal_node    │
              │  Analyze first edge                    │
              └──────────────┬───────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌──────────┐  ┌──────────┐  ┌──────────────┐
       │ TEMPORAL  │  │  VISUAL  │  │  AT GOAL     │
       │ Execute   │  │  (jump)  │  │  → CHECKIN   │
       │ recorded  │  │  Skip to │  │  (multi-     │
       │ action    │  │  next    │  │   confirm)   │
       │ for N     │  │  temporal│  │              │
       │ steps     │  │  edge    │  │              │
       └────┬─────┘  └────┬─────┘  └──────────────┘
            │              │
            └──────┬───────┘
                   │
                   ▼
       ┌───────────────────────┐
       │   STUCK CHECK         │
       │   Same FPV for 10+    │──── Yes ──► RECOVERY
       │   frames?             │               │
       └───────────┬───────────┘               │
                   │ No                        │
                   ▼                           │
          re-localize every                    │
          N steps ──────────────────────►──────┘
                                        back to LOCALIZE
```

### Action Execution Details

```python
ACTION_MAP = {
    'FORWARD':  Action.FORWARD,
    'BACKWARD': Action.BACKWARD,
    'LEFT':     Action.LEFT,
    'RIGHT':    Action.RIGHT,
}

class AutoPlayer(Player):
    def __init__(self):
        super().__init__()
        self.step_count = 0
        self.current_path = []
        self.path_index = 0
        self.current_node = None
        self.relocalize_interval = 5     # re-localize every 5 actions
        self.stuck_buffer = []           # recent feature history
        self.stuck_counter = 0
        self.recovery_phase = 0

    def act(self):
        # Handle pygame events (keep window responsive)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return Action.QUIT

        if self.fpv is None:
            return Action.IDLE

        self.step_count += 1

        # 1. Check-in?
        if self._should_checkin():
            return Action.CHECKIN

        # 2. Stuck?
        recovery = self._check_stuck()
        if recovery is not None:
            return recovery

        # 3. Re-localize?
        if (self.step_count % self.relocalize_interval == 0
                or self.current_path is None
                or self.path_index >= len(self.current_path) - 1):
            self._localize_and_plan()

        # 4. Follow path
        return self._next_action()
```

### Multi-Confirmation Check-in

Don't check in based on a single frame — false positives waste time:

```python
def _should_checkin(self):
    feat = self._extract_feature(self.fpv)

    # Check similarity to ALL 4 target views
    sims = [float(np.dot(feat, gf)) for gf in self.goal_features]
    max_sim = max(sims)
    avg_sim = np.mean(sims[:2])  # front + right are most reliable

    # Method 1: High visual similarity to target
    if max_sim > 0.90:
        self.checkin_confidence += 1
    else:
        self.checkin_confidence = max(0, self.checkin_confidence - 1)

    # Method 2: Graph says we're very close
    if self.current_node is not None:
        try:
            path_len = nx.shortest_path_length(
                self.G, self.current_node, self.goal_node, weight='weight')
            if path_len <= 2:
                self.checkin_confidence += 2
        except nx.NetworkXNoPath:
            pass

    # Require 3+ consecutive confident frames before checking in
    return self.checkin_confidence >= 3
```

### Stuck Detection & Recovery

```python
def _check_stuck(self):
    if len(self.stuck_buffer) < 8:
        return None

    # Compare last 8 features — are we going in circles?
    recent = np.array(self.stuck_buffer[-8:])
    pairwise_sim = recent @ recent.T
    avg_sim = (pairwise_sim.sum() - np.trace(pairwise_sim)) / (8 * 7)

    if avg_sim > 0.97:
        self.stuck_counter += 1
    else:
        self.stuck_counter = 0
        return None

    if self.stuck_counter < 3:
        return None

    # We're stuck! Progressive recovery:
    self.recovery_phase = (self.recovery_phase + 1) % 6
    recovery_actions = [
        Action.LEFT,                    # 1. Turn left
        Action.FORWARD,                 # 2. Try forward
        Action.RIGHT,                   # 3. Turn right
        Action.RIGHT,                   # 4. Turn right more
        Action.FORWARD,                 # 5. Try forward
        Action.BACKWARD,               # 6. Back up
    ]
    # Force re-localization after recovery
    self.current_path = None
    self.stuck_counter = 0
    return recovery_actions[self.recovery_phase]
```

---

## Part 7: Multi-View Goal Matching

### Use All 4 Target Views

The game provides Front, Right, Back, Left views of the goal location.
This is a powerful constraint — use it:

```python
def _setup_goal(self):
    targets = self.get_target_images()
    # targets[0] = Front, targets[1] = Right,
    # targets[2] = Back,  targets[3] = Left

    self.goal_features = []
    for img in targets:
        feat = self._extract_feature(img)
        self.goal_features.append(feat)

    # Strategy: query each view against database, collect candidates
    all_candidates = {}  # node_idx → cumulative score
    view_weights = [1.0, 0.7, 0.5, 0.7]  # front > side > back

    for view_idx, (feat, weight) in enumerate(
            zip(self.goal_features, view_weights)):
        D, I = self.index.search(feat.reshape(1, -1), k=20)
        for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
            idx = int(idx)
            # Decay score by rank
            score = float(dist) * weight / (1 + 0.1 * rank)
            all_candidates[idx] = all_candidates.get(idx, 0) + score

    # Pick node with highest fused score
    self.goal_node = max(all_candidates, key=all_candidates.get)
    print(f"Goal node: {self.goal_node} "
          f"(score: {all_candidates[self.goal_node]:.4f})")

    # Also keep top-5 goal candidates for fallback
    sorted_candidates = sorted(all_candidates.items(),
                               key=lambda x: x[1], reverse=True)
    self.goal_candidates = [c[0] for c in sorted_candidates[:5]]
```

### Goal Node Verification with Re-Ranking

For the goal node specifically, use SuperPoint+LightGlue re-ranking
to make sure we have the right place:

```python
# Re-rank top-20 goal candidates with local feature matching
# This runs once during pre_navigation — can afford to be slow
verified_goal = self._rerank(targets[0], goal_candidates[:20])[0]
```

---

## Part 8: Learned Action Predictor (Backup Controller)

### Why?
The graph-based planner works well when localization is correct.
But when localization fails (the robot is in an area not well-covered
by the graph), we need a fallback.

### Architecture
A small network that directly predicts the best action given
current and goal visual features:

```
┌─────────────────────────────────────────────┐
│  Input:                                      │
│    current_feature (256-dim)                 │
│    goal_feature    (256-dim)                 │
│    last_3_actions  (3 × one-hot, 12-dim)    │
│                                              │
│  Concat → 524-dim                            │
│  → Linear(524, 256) → ReLU → Dropout(0.2)  │
│  → Linear(256, 128) → ReLU → Dropout(0.2)  │
│  → Linear(128, 4)   → Softmax              │
│                                              │
│  Output: P(FORWARD), P(LEFT), P(RIGHT),     │
│          P(BACKWARD)                         │
└─────────────────────────────────────────────┘
```

### Training Data — Multi-Maze Action Pairs
From ALL exploration trajectories across ALL mazes:
```python
# For each maze m in training_mazes:
#   For each frame i in maze m's trajectory:
#     For several random future frames j (50 to 200 steps ahead):
#       Input:  (feature_i, feature_j)
#       Label:  action_at_frame_i
#
# Also generate reverse traversals:
#   Input:  (feature_j, feature_i)
#   Label:  REVERSE(action_at_frame_j-1)
#
# Training on multi-maze data teaches the action predictor to generalize:
# "given what I see now and what the goal looks like, which way should I go?"
# This works across mazes because the relationship between visual change
# and action is universal (turning left always shifts the view right, etc.)
```

**Dataset size:** 20 mazes × 12K frames × 10 goal samples = **~2.4M pairs**.
Tiny MLP trains in minutes even on CPU.

### When to Use the Action Predictor
- When the graph-based planner returns '?' (unknown action)
- When stuck recovery hasn't worked after 3 cycles
- When localization confidence is very low
- As a tiebreaker when the planner is ambiguous

---

## Part 9: Competition-Day Pipeline

### The Problem
On competition day, the maze is **completely new**. Your pipeline must:
1. Process new exploration data
2. Build all caches from scratch
3. Navigate autonomously

### The Solution: Fully Automated Pipeline

```python
def pre_navigation(self):
    """Everything runs automatically when the game starts navigation phase."""

    # Step 1: Load exploration data
    self._load_trajectory_data()
    # → self.motion_frames, self.file_list

    # Step 2: Extract DINOv2 features for all exploration images
    # Uses pre-trained projection head (trained on 20-30 practice mazes,
    # generalizes because: frozen backbone + cross-maze training)
    self._extract_all_features()
    # → self.features (N × 256), cached to disk

    # Step 3: Build FAISS index
    self._build_faiss_index()
    # → self.index

    # Step 4: Build navigation graph
    self._build_graph()
    # → self.G (networkx graph)

    # Step 5: Locate goal from target images
    self._setup_goal()
    # → self.goal_node, self.goal_features, self.goal_candidates

    print(f"Ready: {len(self.features)} nodes, "
          f"{self.G.number_of_edges()} edges, "
          f"goal={self.goal_node}")
```

### Timing Budget for Pre-Navigation

| Step | M4 Max MPS | Notes |
|---|---|---|
| Load trajectory data | < 1 sec | JSON parsing |
| Extract features (18K images) | 3-5 min | DINOv2-B/14, batch=64 |
| Build FAISS index | < 1 sec | Flat index, 18K×256 |
| Build graph | 5-10 sec | Similarity matrix + shortcuts |
| Goal matching | < 2 sec | 4 FAISS queries + fusion |
| **Total** | **~4-6 min** | One-time cost, cached |

### Caching Strategy
```
cache/
├── dinov2_features_{maze_hash}.npy    # keyed by maze identity
├── faiss_index_{maze_hash}.bin
├── nav_graph_{maze_hash}.pkl
├── superpoint_cache_{maze_hash}.pkl   # if using re-ranking
└── projection_head.pth               # trained model (portable)
```

The maze hash can be computed from `data_info.json` content hash.
This way, re-running on the same maze skips all computation.

### What Transfers vs What Gets Recomputed

| Component | Trained on practice mazes | Recompute on new maze? |
|---|---|---|
| DINOv2 backbone | Frozen (ImageNet pretrained) | No — universal |
| GeM pooling params | Trained on 20-30 mazes | **No** — generalizes |
| Projection MLP | Trained on 20-30 mazes | **No** — generalizes |
| Action predictor | Trained on 20-30 mazes | **No** — generalizes |
| Feature database | Per-maze | **Yes** — extract from new images |
| FAISS index | Per-maze | **Yes** — rebuild from new features |
| Navigation graph | Per-maze | **Yes** — rebuild from new trajectory |
| SuperPoint cache | Per-maze | **Yes** — extract from new images |
| Goal node | Per-maze | **Yes** — match target to new database |

**Key insight:** All the trained models (projection head, action predictor)
generalize across mazes because they were trained on diverse mazes. Only the
maze-specific data structures need recomputation (~5 min on M4 Max).

---

## Part 10: Training Scripts & Pipeline

All scripts live in `scripts/` and import from the `vis_nav` package.
All hyperparameters are centralised in `vis_nav/config.py`.

### Script 1: `scripts/generate_mazes.py` (Run on any machine)
```
Usage:
  python scripts/generate_mazes.py \
    --n-mazes 30 \
    --size 31 \
    --output-dir training_data/

Outputs:
  training_data/
  ├── maze_001/ (data_info.json + images/)
  ├── maze_002/
  └── ...
```

### Script 2: `scripts/train_projection.py` (Run on H20 / 5090)
```
Usage:
  # Multi-maze (recommended):
  python scripts/train_projection.py \
    --data-dir training_data/ \
    --device cuda

  # Single practice maze (quick test):
  python scripts/train_projection.py \
    --data-dir data/ \
    --single-maze \
    --device cuda

Outputs:
  - models/projection_head.pth       (GeM + Projection MLP weights)
  - models/projection_head_full.pth  (full model incl. fine-tuned backbone)
  - models/training_log.json         (loss, config, etc.)
```

### Script 3: `scripts/train_action_predictor.py` (Run on H20 / 5090)
```
Usage:
  python scripts/train_action_predictor.py \
    --data-dir data/ \
    --single-maze \
    --device cuda

Outputs:
  - models/action_predictor.pth
```

Feature extraction and graph construction are **not** separate scripts —
they run automatically inside `player.py → pre_navigation()` with
disk caching (keyed by data hash). This means the system works
end-to-end without manual intervention on competition day.

---

## Part 11: Key Parameters

| Parameter | Value | Rationale |
|---|---|---|
| **Backbone** | DINOv2 ViT-B/14 | Best quality we can afford at inference |
| **Output dim** | 256 | Compact, fast FAISS, sufficient capacity |
| **Subsample rate** | 2 | Fine-grained graph, every junction captured |
| **N global shortcuts** | 500 | Rich graph topology |
| **Per-node shortcuts** | 3 | Ensures no isolated regions |
| **Min shortcut gap** | 30 | Prevents trivially close shortcuts |
| **Shortcut sim threshold** | 0.80 | Quality filter |
| **Temporal edge weight** | 1.0 | One action = unit cost |
| **Backward edge weight** | 1.5 | Slight penalty for reversing |
| **Visual edge weight** | 0.5 | Free "teleportation" (already there) |
| **Re-localize interval** | 5 | Every 5 actions |
| **Stuck threshold** | 0.97 | Average pairwise sim of last 8 frames |
| **Check-in confidence** | 3 | Need 3 consecutive high-confidence frames |
| **Check-in sim threshold** | 0.90 | Very high — avoid false check-ins |
| **FAISS top-K** | 10 | Candidates for localization |
| **Re-rank top-K** | 20 | Only when confidence is low |
| **Training mazes** | 20-30 | Diversity vs training time sweet spot |
| **Synthetic ratio** | 20% | Texture collages per batch |

---

## Part 12: Expected Performance

### Navigation Speed
| Scenario | Estimated Time | Notes |
|---|---|---|
| Easy (goal near exploration path) | 10-20 sec | Mostly temporal edges |
| Medium (some shortcuts needed) | 20-40 sec | Few visual jumps |
| Hard (far from any explored area) | 40-60 sec | Heavy re-localization |

### Accuracy
| Component | Expected Accuracy |
|---|---|
| Place recognition (top-1) | 90-95% (cross-maze trained) |
| Place recognition (top-5) | 99%+ |
| With re-ranking (top-1) | 97%+ |
| Goal identification | 95%+ (multi-view fusion) |
| Check-in success | 95%+ (multi-confirmation) |

### vs Single-Maze Training
| Metric | Single maze | Multi-maze (20+) |
|---|---|---|
| Practice maze accuracy | 95% | 93% (slightly lower — more general) |
| **New maze accuracy** | **60-70%** | **90-95%** |
| Competition robustness | Low | **High** |

---

## Part 13: File Structure

```
vis_nav_player/
│
├── player.py                       # ENTRY POINT — game submission
├── baseline.py                     # Original VLAD baseline (reference)
│
├── vis_nav/                        # Main Python package
│   ├── __init__.py
│   ├── config.py                   # ALL constants & hyperparameters
│   ├── utils.py                    # Device detection, caching, I/O
│   │
│   ├── models/                     # Neural network modules
│   │   ├── __init__.py
│   │   ├── backbone.py             # DINOv2 + GeM + ProjectionMLP
│   │   └── action_predictor.py     # Fallback action MLP
│   │
│   ├── navigation/                 # Online navigation logic
│   │   ├── __init__.py
│   │   ├── graph.py                # Topological graph (temporal + visual)
│   │   ├── localizer.py            # FAISS index + temporal consistency
│   │   └── planner.py              # Goal setup, check-in, stuck recovery
│   │
│   └── data/                       # Datasets & transforms
│       ├── __init__.py
│       ├── transforms.py           # Image transforms + batch extraction
│       └── maze_dataset.py         # MazeExplorationDataset + PKSampler
│
├── scripts/                        # Standalone CLI training scripts
│   ├── train_projection.py         # Train DINOv2 projection head (CUDA)
│   ├── train_action_predictor.py   # Train action MLP (CUDA)
│   └── generate_mazes.py           # Generate random maze layouts
│
├── models/                         # Trained weights (Git LFS)
│   ├── projection_head.pth
│   └── action_predictor.pth
├── cache/                          # Auto-generated, gitignored
├── data/                           # Current maze exploration data
│   ├── data_info.json
│   ├── images/
│   └── textures/                   # 200 wall texture patterns
├── training_data/                  # Multi-maze training data (gitignored)
│
├── setup.sh                        # One-command environment setup
├── environment.yaml                # Conda environment spec
├── .gitattributes                  # Git LFS tracking rules
├── Idea.md                         # This document
├── CONTEXT.md                      # Cross-device AI assistant context
└── README.md
```

---

## Summary: Why This Wins

| Dimension | Baseline | Our System |
|---|---|---|
| Features | SIFT + VLAD (16K-dim) | DINOv2-B + Projection (256-dim) |
| Training data | None (hand-crafted) | 20-30 mazes + synthetic + augmentation |
| Vocabulary | KMeans (slow, fragile) | End-to-end learned (cross-maze) |
| Graph | 30 shortcuts, subsample 5 | 500+ shortcuts, subsample 2, per-node K |
| Goal matching | Front view only | All 4 views, weighted fusion + re-ranking |
| Navigation | Manual keyboard | Fully autonomous with re-planning |
| Localization | Single-frame, no filtering | FAISS + temporal consistency + re-ranking |
| Stuck handling | None | 6-step progressive recovery |
| Check-in | Manual SPACE key | Multi-frame confidence thresholding |
| Generalization | Single maze only | Cross-maze trained, texture-aware |
| Fallback | None | Learned action predictor |
| Speed | Human reaction time | ~20-40 sec typical |
| Methodology pts | 1 (manual) | 2 (full automation) |
