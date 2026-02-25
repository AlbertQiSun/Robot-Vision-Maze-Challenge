# vis_nav_player — Full Diagnosis & Redesign Plan
> Give this entire file to Cursor before touching any code.

---

## 0. What's Actually Broken (Priority Order)

| # | Severity | Location | Description |
|---|---|---|---|
| 1 | 🔴 CRITICAL | `player.py` + `planner.py` | `check_stuck()` / stuck buffer **never called** — recovery is dead code |
| 2 | 🔴 CRITICAL | `player.py` / `graph.py` | Visual-shortcut edges have no physical action — path follower silently skips them and may return the **wrong** temporal edge |
| 3 | 🔴 CRITICAL | `player.py` | `path_index` is **always reset to 0** and never incremented — the robot replays the start of the path every step instead of advancing |
| 4 | 🟠 HIGH | `planner.py` | `stuck_buffer` is never populated (`.append()` is never called from `player.py`) |
| 5 | 🟠 HIGH | `train_action_predictor.py` | Action predictor trained on **single-view goals**; inference uses a **fused 4-view goal** — distribution mismatch |
| 6 | 🟠 HIGH | `action_predictor.py` | Place descriptors are not geometrically structured; `diff = c_enc − g_enc` does **not** encode a navigable direction |
| 7 | 🟡 MEDIUM | `config.py` | `checkin_sim_threshold = 0.92` + `checkin_confidence_needed = 8` — will almost never fire in an unseen competition maze |
| 8 | 🟡 MEDIUM | `config.py` | `temporal_bwd_weight = 10.0` — over-penalises backtracking; in competition the robot needs it |
| 9 | 🟡 MEDIUM | `player.py` | `extract_single_feature()` calls full DINOv2 every frame — expensive on MPS; no feature caching between frames |
| 10 | 🔵 LOW | `graph.py` | Visual shortcut edges carry no recorded `action` attribute at all, so there is no fallback even when manually inspected |

---

## 1. Bug-by-Bug Deep Dive

### Bug 1 — `check_stuck()` is dead code

**File**: `player.py` `act()` method and `planner.py`

`GoalPlanner` has a full recovery system:
- `stuck_buffer` — deque that should hold recent feature vectors
- `check_stuck()` — computes pairwise similarity across the window; fires if avg > `stuck_sim_threshold`
- `_recovery_queue` / `_RECOVERY_SEQUENCES` — queued action bursts (BACKWARD×10, LEFT×5, etc.)

**None of it is wired in.** `player.py` never calls `planner.check_stuck()` and never appends to `planner.stuck_buffer`. The robot can spin in a circle forever.

**Fix**: In `player.py`'s `act()`, after extracting `_current_feat`, add:
```python
# Feed stuck detector
self.planner.stuck_buffer.append(self._current_feat)
recovery_action = self.planner.check_stuck()
if recovery_action:
    return _ACT_MAP[recovery_action]
```
This must come **before** the localize / plan steps so recovery overrides everything.

---

### Bug 2 — Visual-shortcut edges have no physical action

**Files**: `graph.py` → `get_edge_action()` and `player.py` → `_get_first_path_action()`

When A* plans a path it may choose a visual shortcut as one of the edges (because `visual_edge_weight = 0.3`, much cheaper than temporal). But `get_edge_action()` returns `"VISUAL"` for those edges — not a real action name.

`_get_first_path_action()` iterates the path from index 0 and skips any edge that is not in `_ACT_MAP`:
```python
for i in range(len(p.current_path) - 1):
    a, b = p.current_path[i], p.current_path[i + 1]
    edge = self.nav_graph.get_edge_action(a, b)
    if edge in _ACT_MAP:
        return edge   # ← returns the FIRST temporal edge found, not the correct next step
return None
```

**Example failure**: Path = `[342, 1203(visual), 1204, 1205]`. The function skips `342→1203` (VISUAL) and returns the action on `1203→1204`, which might be FORWARD — but the robot is still at node 342 and has no idea how to get to 1203.

**Fix**: Visual edges represent "teleport to a similar-looking place". The correct behaviour is: if the next edge is a visual shortcut, treat it as "navigate toward that node using the action predictor". Specifically, when the next node is a visual-shortcut hop, pass `features[target_node]` as the immediate mini-goal to the action predictor instead of `goal_feature_fused`.

Alternatively, the simpler fix: **remove visual-shortcut edges from consideration during path-following**. Only traverse temporal edges. Change `_get_first_path_action()` to skip shortcuts by removing them from the path or by excluding them from A*'s edge set during path-following phase (add a `follow_temporal_only` flag to `find_path`).

---

### Bug 3 — `path_index` is always 0

**File**: `player.py`

Every call to `act()` does:
```python
p.current_path = self.nav_graph.find_path(...)
p.path_index = 0
```
And `_get_first_path_action()` always loops from `i=0`. The variable `path_index` is declared and stored but **never read or incremented anywhere**. This makes the path-following equivalent to replanning from scratch every step — which it now does even with the throttling fix, because each new plan starts at `current_node` anyway.

This is actually acceptable if re-planning is throttled (as we fixed). But `path_index` should either be used (increment after each step) or removed to reduce confusion.

**Fix (proper path-following)**:
- When a path is computed, set `path_index = 0`
- In `_get_first_path_action()`, start the loop from `path_index`, not 0
- After returning an action, increment `path_index`
- Only replan when `path_index >= len(current_path) - 1` (path exhausted) or stuck

---

### Bug 4 — `stuck_buffer` never populated (see Bug 1)

Same root cause as Bug 1. The buffer is declared in `GoalPlanner.__init__` but `player.py` never calls `planner.stuck_buffer.append(...)`.

---

### Bug 5 — Action predictor: goal distribution mismatch

**Files**: `train_action_predictor.py` and `player.py`

**Training**: goal = `features[j]` = single L2-normalised vector from exploration frame j
**Inference**: goal = `planner.goal_feature_fused` = weighted average of 4 views, then re-normalised

These are different distributions. A single-view feature is a clean unit-norm vector in the 256-sphere. A fused multi-view feature is the centroid of 4 points, renormalised — its "style" is different from any single training sample.

**Fix Option A**: Change training to also sample fused goals. For each frame i, randomly pick 4 nearby frames (within ±10) and average their features as the training goal, simulating what inference does.

**Fix Option B**: During inference, pick the single best-matching goal view (highest dot-product with current feature) and pass that to the action predictor instead of the fused feature.

Option B is a one-line change in `player.py`:
```python
# Instead of:
goal_feat_for_predictor = p.goal_feature_fused
# Use:
sims = [float(np.dot(self._current_feat, gf)) for gf in p.goal_features]
goal_feat_for_predictor = p.goal_features[np.argmax(sims)]
```

---

### Bug 6 — Place descriptors are not geometrically structured

**File**: `action_predictor.py`

The `ActionPredictor` computes:
```python
diff = c_enc - g_enc   # current - goal in encoded space
prod = c_enc * g_enc
```

L2-normalised place embeddings encode **"what place this is"**, not **"where this place is relative to me"**. The difference of two place embeddings does not reliably encode a navigable direction. Two places that look identical (same texture, e.g. identical corridor walls) can be in opposite directions. The model is asked to predict FORWARD/LEFT/RIGHT/BACKWARD purely from appearance, which is geometrically underdetermined.

**Evidence**: The model trains to reasonable accuracy on a single maze (it memorises the trajectory), but generalises poorly to new mazes. In the competition maze (which is different), the action predictor has no reliable basis to predict direction.

**Better architectures (see Section 3)**.

---

### Bug 7 — Check-in thresholds too strict for competition

**File**: `config.py`

```python
checkin_sim_threshold: float = 0.92
checkin_confidence_needed: int = 8
```

A cosine similarity of 0.92 means the current view is nearly identical to one of the 4 goal views. In the **competition maze** (different from training), the robot approaches the goal from a different angle than the reference images were captured. Getting 0.92 sustained for 8 frames is unlikely.

**Fix**: Lower to `checkin_sim_threshold = 0.82` and `checkin_confidence_needed = 5`. If this causes false check-ins, the actual check-in logic at the game level will simply be incorrect — much better than never checking in.

Also: the second check-in path in `player.py` uses `nav_cfg.checkin_sim_threshold` correctly, but the graph-distance-based boost (`checkin_graph_dist`) in `planner.should_checkin()` is never called from `player.py` (only the similarity path is called). Consider also triggering check-in when graph distance to goal is ≤ 2.

---

### Bug 8 — `temporal_bwd_weight = 10.0` kills backtracking

**File**: `config.py`

The graph penalises backward traversal 10× compared to forward. In competition, the exploration trajectory may have taken a completely different route than needed. To reach the goal, the robot might need to backtrack extensively. A weight of 10.0 means A* will prefer very long forward routes over any backtracking, which can result in the robot wandering for minutes.

**Fix**: Lower `temporal_bwd_weight` to `3.0`–`4.0`. The asymmetry is still preserved (prevent trivial ping-pong) but backtracking becomes viable when the forward path is very long.

---

### Bug 9 — DINOv2 runs every single frame (expensive)

**File**: `player.py` `act()`

`extract_single_feature()` runs a full DINOv2 ViT-B/14 forward pass on every frame. On Apple MPS this is ~80–150ms per call. If the game engine calls `act()` faster than this, you will stall. Also, if the image hasn't changed much (robot is pushing against a wall), you're wasting time re-computing the same feature.

**Fix**: Cache the last feature and compare raw pixel similarity (or frame hash) before re-extracting:
```python
# Only re-extract if frame changed significantly
if self._last_fpv_hash != hash(self.fpv.tobytes()[::1000]):
    self._current_feat = extract_single_feature(...)
    self._last_fpv_hash = hash(self.fpv.tobytes()[::1000])
```

---

## 2. The Root Cause of "Shaking"

Three compounding issues cause the jitter:

1. **Replanning every step** (fixed by our replan throttle) — node index noise flips the first path edge
2. **No action smoothing** (fixed by our majority-vote deque) — raw argmax oscillates on close logits
3. **Bug 3 above** — path_index=0 every step means the "path" is always replayed from the start, amplifying noise from localization

Even with our anti-shake patch, the robot will still oscillate if it genuinely can't localise itself (competition maze mismatch). The localizer returning node 342 one frame and node 678 the next (because both look similar in an unfamiliar maze) gives the path planner completely different starting points, producing completely different first actions.

---

## 3. Better Architecture Options

### Option A: Pure Graph-Follower (No Action Predictor) — RECOMMENDED FOR COMPETITION

**Philosophy**: Trust the graph completely. The action predictor is the source of most uncertainty.

1. Localise to nearest graph node using FAISS
2. Plan path to goal node using A* (temporal edges only — no visual shortcuts during execution)
3. Execute the first temporal edge's action
4. Increment `path_index`
5. Every 5 steps, re-localise and re-plan only if the estimated node has moved ≥ 5 frames

This is deterministic and interpretable. It works as long as localisation is reliable (R@1 > 0.7 in competition). The action predictor adds noise, not signal, because it was trained on a different maze.

**Change required in `player.py`**:
- Remove the action predictor fallback entirely (or make it truly last-resort after N consecutive path failures)
- Add `check_stuck()` wiring (Bug 1 fix)
- Fix `path_index` (Bug 3 fix)

---

### Option B: Graph-Follower + Bearing-Based Steering (Better generalisation)

When localisation is poor or there's no path, estimate a rough bearing to the goal and execute left/right/forward based on it.

The idea: run FAISS top-K and check which direction of temporal edges from the candidate nodes point "toward" the goal (by measuring feature similarity to subsequent nodes along the graph). This gives a soft bearing without any learned model.

Pseudocode:
```python
def direction_to_goal(current_node, goal_feat, graph, features, k=5):
    # For each direction: FORWARD means following the temporal edge forward
    # score each by: similarity(features[next_temporal_node], goal_feat)
    fwd = features[current_node + 1] @ goal_feat if current_node + 1 < len(features) else -1
    bwd = features[current_node - 1] @ goal_feat if current_node > 0 else -1
    # prefer forward; return LEFT/RIGHT from graph topology if available
    return "FORWARD" if fwd > bwd else "BACKWARD"
```

---

### Option C: Retrain Action Predictor with Better Labels (Best accuracy, needs time)

The current training objective `(feat_i, feat_j, hist_i) → action_i` is asking the model to predict "what step was taken at frame i" given "I'm at i and want to reach j." For small gaps (1–10 frames), this is sensible: the action taken at i is what starts you toward j. For large gaps (30–100 frames), the label is misleading.

**Fix the training data**:
1. Use **only short-range pairs** (gap 1–15 frames) — these have reliable action labels
2. Add **multi-maze training** — the model must generalise to unseen textures
3. Make **goal distribution match inference**: for each sample, randomly either use `features[j]` directly or use the average of `features[j-2:j+2]` as the goal, so the model sees fused-style goals at training time
4. Add **class balancing**: log the label distribution — if the exploration has 70% FORWARD steps, the model will predict FORWARD for everything. Use class-weighted cross-entropy or oversample turns.

**Retrain command**:
```bash
python scripts/train_action_predictor.py \
    --data-dir training_data/ \
    --device cuda \
    --epochs 100 \
    --subsample 1
```

---

### Option D: Replace Action Predictor with a Relative-Bearing Regressor

Instead of classifying FORWARD/LEFT/RIGHT/BACKWARD, train a model to predict the **angle to the goal** (0–360°, modulo current heading). Then convert angle to action.

This is more geometrically meaningful and would generalise better across mazes. Requires:
- Recording absolute heading during exploration (not currently in `data_info.json`)
- More complex training pipeline

**Not recommended for the current competition timeline.**

---

## 4. Minimum Viable Fix List (Priority Order for Competition)

Do these in order — each is independently testable:

### Step 1 — Wire stuck detection (30 min)
In `player.py` `act()`, before the localize block:
```python
self.planner.stuck_buffer.append(self._current_feat)
recovery = self.planner.check_stuck()
if recovery:
    self._action_vote.clear()   # clear vote buffer during recovery
    return _ACT_MAP[recovery]
```

### Step 2 — Fix path_index (20 min)
In `player.py` `act()`, change `_get_first_path_action` to start from `p.path_index` and increment it:
```python
def _get_first_path_action(self) -> str | None:
    p = self.planner
    if not p.current_path or len(p.current_path) < 2:
        return None
    while p.path_index < len(p.current_path) - 1:
        a, b = p.current_path[p.path_index], p.current_path[p.path_index + 1]
        edge = self.nav_graph.get_edge_action(a, b)
        p.path_index += 1
        if edge in _ACT_MAP:
            return edge
    return None
```
Note: when a new plan is made, `path_index = 0` — keep that reset.

### Step 3 — Lower check-in thresholds (5 min)
In `config.py`:
```python
checkin_sim_threshold: float = 0.82     # was 0.92
checkin_confidence_needed: int = 5      # was 8
temporal_bwd_weight: float = 4.0        # was 10.0
```

### Step 4 — Fix visual-edge path following (30 min)
In `player.py`, when the path's next hop is a visual shortcut, set a temporary waypoint:
```python
def _get_next_waypoint_feat(self) -> np.ndarray | None:
    """If next edge is visual, return that node's feature as a mini-goal."""
    p = self.planner
    if not p.current_path or len(p.current_path) < 2:
        return None
    a, b = p.current_path[0], p.current_path[1]
    if self.nav_graph.get_edge_action(a, b) == "VISUAL":
        return self.features[b]
    return None
```
Then in the action-predictor fallback:
```python
waypoint_feat = self._get_next_waypoint_feat()
goal_for_predictor = waypoint_feat if waypoint_feat is not None else p.goal_feature_fused
idx = self.action_predictor.predict_action(
    self._current_feat, goal_for_predictor, list(self.action_history), self.device,
)
```

### Step 5 — Fix action predictor goal distribution (15 min)
In `player.py` `act()`, when calling the action predictor, pass the single best-matching goal view instead of the fused average:
```python
if self.planner.goal_features:
    sims = [float(np.dot(self._current_feat, gf)) for gf in self.planner.goal_features]
    best_goal_feat = self.planner.goal_features[int(np.argmax(sims))]
else:
    best_goal_feat = self.planner.goal_feature_fused

idx = self.action_predictor.predict_action(
    self._current_feat, best_goal_feat, list(self.action_history), self.device,
)
```

### Step 6 — Retrain action predictor with short-range only + multi-maze (2–4 hours)
In `train_action_predictor.py`, change the ranges list:
```python
ranges = [(1, 8), (8, 20)]   # remove the (30, 100) long-range range
# Long-range labels are noise: the local action at frame i has nothing to
# do with the overall direction to a goal 50+ frames away.
```
Then retrain on all available mazes:
```bash
python scripts/train_action_predictor.py --data-dir training_data/ --device cuda --epochs 100
```

---

## 5. File-Level Change Map

```
player.py
├── __init__: no change needed (anti-shake already added)
├── act():
│   ├── ADD: stuck_buffer.append + check_stuck() call (Bug 1)
│   ├── CHANGE: action predictor uses best single-view goal not fused (Bug 5)
│   └── KEEP: replan throttle + majority-vote smoother (already fixed)
├── _get_first_path_action():
│   └── CHANGE: start from path_index, increment it (Bug 3)
└── ADD: _get_next_waypoint_feat() helper (Bug 2)

config.py
├── checkin_sim_threshold: 0.92 → 0.82
├── checkin_confidence_needed: 8 → 5
└── temporal_bwd_weight: 10.0 → 4.0

train_action_predictor.py
└── ActionPairDataset: ranges = [(1,8),(8,20)] not [(1,10),(10,30),(30,100)]

(optional, after testing)
action_predictor.py: no architecture change needed yet — fix training first
graph.py: no change needed — visual shortcuts are fine in the index, just handled better during execution
```

---

## 6. Quick Sanity Tests

After each fix, test in isolation:

```bash
# Test 1: Does check_stuck() fire correctly?
# Manually stop the robot (put it against a wall) and observe recovery burst in logs

# Test 2: Does path_index advance?
# Add print(f"path_index={p.path_index}") in _get_first_path_action and watch it count up

# Test 3: Does check-in fire?
# Navigate near the goal manually and check if CHECKIN appears in logs

# Test 4: Action predictor accuracy before/after retraining
# python scripts/train_action_predictor.py --single-maze ...
# check val_acc: should be > 0.65 with short-range only
```

---

## 7. Key Invariants to Preserve

1. **Feature normalisation**: All features MUST be L2-normalised before FAISS or cosine dot products. `PlaceFeatureExtractor` does this internally — do not accidentally strip normalisation.

2. **numpy laundering**: Any tensor returned from MPS/CUDA must go through `np.array(x, copy=True)` before passing to FAISS. The helper `to_numpy()` in `utils.py` does this — use it everywhere.

3. **BatchNorm in eval mode**: `feat_model.eval()` must be called before any single-image inference. If you accidentally call it in train mode with batch_size=1, BatchNorm will crash or give wrong results.

4. **Cache invalidation**: The cache hash in `utils.py` includes the projection head's mtime. If you retrain and save a new model, the cache auto-invalidates. Don't manually delete cache if the model file changed — let the hash handle it.

5. **Competition maze is different**: Do not assume the robot will localise well. The path planner and stuck recovery must work even when localisation returns a wrong node.
