<p align="center">
  <h1 align="center">R-VLFM: Relational Vision-Language Frontier Maps for Zero-Shot Semantic Navigation</h1>
  <h3 align="center">
    <a href="https://scholar.google.com/citations?user=guiqu3wAAAAJ">Rongali Sai Bhargav</a>
    &nbsp;·&nbsp;
    Built on <a href="https://naoki.io/portfolio/vlfm.html">VLFM</a> by Yokoyama et al. (ICRA 2024)
  </h3>
  <p align="center">
    <a href="https://github.com/Saibhargav1208/R-VLFM">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
    </a>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/built%20with-Python3-red.svg" />
    </a>
    <a href="https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct">
      <img src="https://img.shields.io/badge/VLM-Qwen2--VL--2B-blue.svg" />
    </a>
  </p>
</p>

---

## What is R-VLFM?

**R-VLFM** extends [VLFM (ICRA 2024)](https://arxiv.org/abs/2312.03275) to handle **relational object goals** — a capability that VLFM and all known follow-up works do not address.

| | VLFM | R-VLFM |
|---|---|---|
| Goal specification | `"cup"` | `"the cup on the kitchen counter"` |
| Finds | Any cup anywhere | The specific cup in the specified location |
| VLM backbone | BLIP-2 (Jan 2023) | Qwen2-VL-2B (2024) |
| Frontier scoring | Single cosine similarity | Two-stage: coarse score + relational VQA |
| Goal parser | None | LLM-based relation extractor |

### Why this matters

Real-world deployments (hospitals, warehouses, homes) require **precise** object navigation. A nurse robot that finds *any* medicine bottle instead of *the specific bottle on the blue shelf* is not useful — or safe. R-VLFM bridges that gap without any task-specific training.

---

## Architecture

```
User input: "find the red cup on the kitchen counter"
                         |
                         v
              +---------------------+
              |   Goal Parser        |  Qwen2-VL-2B
              |  object  = "cup"     |  (one-time at episode start)
              |  relation = "on the  |
              |  kitchen counter"    |
              +----------+----------+
                         |
          +--------------v------------------------------+
          |            STAGE 1  (every step)            |
          |  Frontier scoring via Qwen2-VL-2B           |
          |  Q: "How likely does this scene lead        |
          |      toward a cup?"  ->  score in [0,1]    |
          |  Builds value map (same as VLFM)            |
          |  CLIP pre-filter -> top-3 frontiers         |
          +--------------+------------------------------+
                         |  top-3 candidates
          +--------------v------------------------------+
          |            STAGE 2  (decision time)         |  <- NEW
          |  Relational verification via Qwen2-VL-2B   |
          |  Q: "Is there a red cup on a kitchen        |
          |      counter in this image?"  ->  score     |
          |  Final = 0.4 x s1  +  0.6 x s2             |
          +--------------+------------------------------+
                         |  best frontier
                         v
                   Navigate there
```

---

## New Files

| File | Description |
|---|---|
| `vlfm/vlm/qwen_vl.py` | Qwen2-VL-2B wrapper. Replaces BLIP-2. Three methods: `score_frontier`, `verify_relation`, `parse_goal`. Drop-in compatible: has `.cosine()` method. |
| `vlfm/policy/r_vlfm_policy.py` | `RVLFMPolicy` — extends `ITMPolicyV2`. Adds two-stage scoring and relational goal parsing. |
| `vlfm/policy/habitat_policies.py` | Added `HabitatRVLFMPolicy` (registered with `@baseline_registry`). |
| `config/experiments/r_vlfm_objectnav_hm3d.yaml` | Drop-in Habitat config. Only change vs. VLFM: `policy.name: "HabitatRVLFMPolicy"`. |
| `scripts/launch_r_vlfm_servers.sh` | Launches all servers. Replaces BLIP2ITM pane with Qwen2-VL-2B. |

---

## Installation

### 1. Base environment (same as VLFM)
```bash
conda create -n r_vlfm python=3.9 -y
conda activate r_vlfm

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install -e .[habitat]
git clone https://github.com/WongKinYiu/yolov7.git
```

### 2. R-VLFM additional dependencies
```bash
pip install transformers>=4.45.0 accelerate qwen-vl-utils
```

Qwen2-VL-2B weights (~4.5GB) are downloaded automatically on first server launch.

### 3. GPU requirements

| Component | VRAM |
|---|---|
| Qwen2-VL-2B (float16) | ~5 GB |
| GroundingDINO + SAM + YOLO | ~3 GB |
| **Total** | **~8 GB** |

---

## Dataset and Weights

Download HM3D dataset and weights following the [original VLFM instructions](https://github.com/bdaiinstitute/vlfm).

Required weights in `data/`:
- `mobile_sam.pt`
- `groundingdino_swint_ogc.pth`
- `yolov7-e6e.pt`
- `pointnav_weights.pth`

---

## Running R-VLFM

### Step 1 — Launch model servers
```bash
chmod +x scripts/launch_r_vlfm_servers.sh
./scripts/launch_r_vlfm_servers.sh
```

Wait ~90 seconds for all models to load. Monitor with:
```bash
tmux attach-session -t r_vlfm_servers_<id>
```

### Step 2 — Evaluate on HM3D
```bash
python -m vlfm.run --config-name r_vlfm_objectnav_hm3d
```

### Step 3 — Evaluate on MP3D
```bash
python -m vlfm.run \
  --config-name r_vlfm_objectnav_hm3d \
  habitat.dataset.data_path=data/datasets/objectnav/mp3d/val/val.json.gz
```

### Step 4 — Kill servers when done
```bash
tmux kill-session -t r_vlfm_servers_<id>
```

---

## Configuration

The only required change vs. the original VLFM config:

```yaml
rl:
  policy:
    name: "HabitatRVLFMPolicy"   # was "HabitatITMPolicyV2"
    qwenvl_port: 12190
```

Stage-2 hyperparameters (in `vlfm/policy/r_vlfm_policy.py`):

| Parameter | Default | Description |
|---|---|---|
| `TOP_K_VERIFY` | 3 | Number of Stage-1 frontiers re-verified by Stage-2 |
| `STAGE1_MIN_SCORE` | 0.20 | Minimum Stage-1 score to enter Stage-2 |
| `STAGE1_WEIGHT` | 0.4 | Stage-1 blend weight in final score |
| `STAGE2_WEIGHT` | 0.6 | Stage-2 blend weight in final score |

---

## Comparison with VLFM

| | VLFM | R-VLFM |
|---|---|---|
| VLM | BLIP-2 (2023) | Qwen2-VL-2B (2024) |
| Goal type | Simple noun | Simple + relational |
| Frontier scoring | Single-stage cosine | Two-stage: coarse + relational verify |
| Goal parsing | None | Qwen2-VL LLM parser |
| VRAM | ~6 GB | ~8 GB |
| Code changes | — | 2 new files, 1 modified, 1 new config |

For simple goals (e.g. `"chair"`), R-VLFM automatically skips Stage-2 and behaves identically to VLFM with a stronger VLM backbone. Zero regression on standard benchmarks.

---

## Citation

```bibtex
@misc{rongali2025rvlfm,
  title   = {R-VLFM: Relational Vision-Language Frontier Maps for Zero-Shot Semantic Navigation},
  author  = {Rongali, Sai Bhargav},
  year    = {2025},
  url     = {https://github.com/Saibhargav1208/R-VLFM}
}

@inproceedings{yokoyama2024vlfm,
  title     = {VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation},
  author    = {Naoki Yokoyama and Sehoon Ha and Dhruv Batra and Jiuguang Wang and Bernadette Bucher},
  booktitle = {International Conference on Robotics and Automation (ICRA)},
  year      = {2024}
}
```

---

## Contact

**Rongali Sai Bhargav** — Honda R&D | IIT Bombay (B.Tech 2024)
[Google Scholar](https://scholar.google.com/citations?user=guiqu3wAAAAJ) · [GitHub](https://github.com/Saibhargav1208)

Interested in collaboration on robot navigation, VLMs, or graph-based scene understanding? Open an issue or reach out directly.

---

*R-VLFM is built on [VLFM](https://github.com/bdaiinstitute/vlfm) (MIT License) by the Boston Dynamics AI Institute. All original VLFM code is intact — only `habitat_policies.py` and `__init__.py` were extended, never modified.*
